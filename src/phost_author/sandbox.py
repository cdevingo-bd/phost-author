"""Run untrusted Python in a bubblewrap sandbox with limits. What it does and does not protect.

Enforced (when ``bwrap`` is available, reported as ``sandbox="bwrap"``):
  * the filesystem is read-only, the user's home directory and /tmp are replaced by empty private ones, and only the
    paths the interpreter needs are mounted back, read-only;
  * no network (own network namespace), own pid and ipc namespaces, the process tree dies with the parent;
  * a wall-clock limit (the whole tree is killed), a CPU-time limit, a limit on the size of any one file, a limit on
    the total growth of the scratch directory (polled, so a fast writer can overshoot by a fraction of a second of
    writing), no core dumps;
  * one writable scratch directory (the kernel cache), nothing else.

Not enforced:
  * the GPU is shared. A submission can use GPU memory, run long kernels (the wall limit stops it) or fault the device;
  * memory is not limited (an address-space limit breaks CUDA);
  * the verdict is computed in the same process as the submission. A submission that is written to tamper with the
    checker can forge a pass. This sandbox protects the machine, not the integrity of a competition: a leaderboard
    needs the judge in a separate process from the code it judges.

Without ``bwrap`` nothing is isolated, and ``run`` refuses unless ``allow_unsandboxed=True``.
"""

from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field


class SandboxUnavailable(RuntimeError):
    pass


# Warp compiles a submission's kernels with NVRTC, which needs to create files of this size. Measured on Warp 1.16 with a
# warm cache and a one-line kernel: a 64 MB per-file limit fails the compile, 128 MB succeeds. A larger kernel was not
# measured, which is why the default is far above this floor. A policy below it could never run any kernel.
MIN_COMPILE_FILE_MB = 128


@dataclass(frozen=True)
class Policy:
    wall_seconds: float = 300.0
    cpu_seconds: int = 240
    max_file_mb: int = 1024               # any one file (RLIMIT_FSIZE); at least MIN_COMPILE_FILE_MB
    scratch_mb: int = 2048                # growth of the scratch directory in total, all files together
    network: bool = False
    extra_read_only: tuple[str, ...] = ()
    scratch: str | None = None            # writable directory (kernel cache); a private temp dir when None
    warm_cache: str | None = None         # a trusted Warp kernel cache, copied into the scratch before each run

    def __post_init__(self):
        if self.max_file_mb < MIN_COMPILE_FILE_MB:
            raise ValueError(f"max_file_mb={self.max_file_mb} is below {MIN_COMPILE_FILE_MB}, the smallest per-file limit "
                             "under which Warp's compiler can build a kernel (a submission could never run)")
        if self.scratch_mb < self.max_file_mb:
            raise ValueError(f"scratch_mb={self.scratch_mb} is below max_file_mb={self.max_file_mb}: one file could not fit")


@dataclass
class Result:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    sandbox: str                           # "bwrap" or "none"
    seconds: float
    quota_exceeded: bool = False           # the scratch directory grew past policy.scratch_mb and the tree was killed
    extra: dict = field(default_factory=dict)


_POLL_SECONDS = 0.25


def _tree_bytes(path: str) -> int:
    """Bytes in every regular file under ``path``."""
    total = 0
    for root, _, names in os.walk(path):
        for name in names:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def available() -> bool:
    return shutil.which("bwrap") is not None


def _needed_paths() -> list[tuple[str, str]]:
    """(real path, path as the interpreter names it) for directories a private home or /tmp would hide.

    A path may be a symlink (a build cache, an editable install): the real directory is mounted at the name the
    interpreter uses, so ``sys.path`` and ``PYTHONPATH`` mean the same inside.
    """
    home, hidden = os.path.expanduser("~"), ["/tmp"]
    candidates = [sys.prefix, os.path.dirname(sys.executable), *[p for p in sys.path if p]]
    out: dict[str, str] = {}
    for given in candidates:
        given = os.path.abspath(given)
        real = os.path.realpath(given)
        if not os.path.isdir(real):
            continue
        if any(p.startswith(root + os.sep) for p in (given, real) for root in [home, *hidden]):
            out[given] = real
    return [(real, given) for given, real in out.items()]


def _env(scratch_inside: str) -> dict:
    keep = {k: os.environ[k] for k in ("PATH", "LD_LIBRARY_PATH", "LANG", "LC_ALL", "CUDA_VISIBLE_DEVICES") if k in os.environ}
    keep.update(HOME=f"{scratch_inside}/home", WARP_CACHE_PATH=f"{scratch_inside}/warp", PYTHONDONTWRITEBYTECODE="1",
                PYTHONPATH=os.pathsep.join(p for p in sys.path if p), PYTHONNOUSERSITE="1", TMPDIR="/tmp")
    return keep


def command(argv: list[str], policy: Policy, scratch: str) -> tuple[list[str], dict]:
    """The bwrap command line and environment for ``argv``."""
    home = os.path.expanduser("~")
    cmd = ["bwrap", "--ro-bind", "/", "/", "--dev-bind", "/dev", "/dev", "--proc", "/proc",
           "--tmpfs", "/tmp", "--tmpfs", home]
    for real, given in [*_needed_paths(), *[(os.path.realpath(p), os.path.abspath(p)) for p in policy.extra_read_only]]:
        cmd += ["--ro-bind", real, given]
    cmd += ["--bind", scratch, "/mnt", "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--die-with-parent",
            "--new-session", "--chdir", "/tmp"]
    if not policy.network:
        cmd += ["--unshare-net"]
    return cmd + argv, _env("/mnt")


def _limits(policy: Policy):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (policy.cpu_seconds, policy.cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (policy.max_file_mb << 20,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return apply


def run(argv: list[str], *, stdin: str = "", policy: Policy = Policy(), allow_unsandboxed: bool = False) -> Result:
    """Run ``argv`` (an interpreter command) under the policy. Never raises for a failing or hostile child."""
    sandboxed = available()
    if not sandboxed and not allow_unsandboxed:
        raise SandboxUnavailable("bwrap is not installed; refusing to run untrusted code without isolation")
    scratch = policy.scratch or tempfile.mkdtemp(prefix="phost_sandbox_")
    os.makedirs(os.path.join(scratch, "home"), exist_ok=True)
    if policy.warm_cache and os.path.isdir(policy.warm_cache):
        # a copy, never a shared mount: what a submission writes to its cache must not reach the next submission
        # Warp looks for <cache path>/<warp version>, and the warm cache directory is named for that version
        shutil.copytree(policy.warm_cache, os.path.join(scratch, "warp", os.path.basename(policy.warm_cache.rstrip(os.sep))),
                        dirs_exist_ok=True)
    try:
        if sandboxed:
            cmd, env = command(argv, policy, scratch)
        else:
            cmd, env = argv, _env(scratch)
        baseline = _tree_bytes(scratch)
        started = time.perf_counter()
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   env=env, preexec_fn=_limits(policy), start_new_session=True)
        timed_out = quota_exceeded = False
        pending: str | None = stdin
        while True:
            try:
                out, err = process.communicate(pending, timeout=_POLL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                pending = None                                   # the input was handed over on the first call
                timed_out = time.perf_counter() - started > policy.wall_seconds
                quota_exceeded = _tree_bytes(scratch) - baseline > policy.scratch_mb << 20
                if timed_out or quota_exceeded:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    out, err = process.communicate()
                    break
        return Result(process.returncode, out, err, timed_out, "bwrap" if sandboxed else "none",
                      time.perf_counter() - started, quota_exceeded)
    finally:
        if policy.scratch is None:
            shutil.rmtree(scratch, ignore_errors=True)
