"""The sandbox: what it blocks, what it limits, and that it says so when it cannot isolate."""

import os
import shutil
import subprocess
import sys
import time
import uuid

import pytest

from phost_author import sandbox
from phost_author.sandbox import Policy, SandboxUnavailable

pytestmark = pytest.mark.skipif(not sandbox.available(), reason="bwrap is not installed")

PY = [sys.executable, "-s", "-c"]


def run(code, **policy):
    return sandbox.run([*PY, code], policy=Policy(**{"wall_seconds": 60, **policy}))


def test_a_normal_program_runs_and_is_reported_as_sandboxed():
    result = run("print(6*7)")
    assert result.returncode == 0 and result.stdout.strip() == "42" and result.sandbox == "bwrap"


def test_stdin_reaches_the_program():
    result = sandbox.run([*PY, "import sys; print(sys.stdin.read().upper())"], stdin="hello")
    assert result.stdout.strip() == "HELLO"


def test_the_network_is_blocked():
    result = run("import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=3)")
    assert result.returncode != 0 and ("OSError" in result.stderr or "unreachable" in result.stderr)


def test_the_users_files_are_not_visible_and_writes_do_not_escape():
    home = os.path.expanduser("~")
    secret = os.path.join(home, f".phost_canary_{uuid.uuid4().hex}")
    mark = os.path.join(home, f".phost_written_{uuid.uuid4().hex}")
    with open(secret, "w") as stream:
        stream.write("secret")
    try:
        result = run(f"""
import os
try:
    print('READ', open({secret!r}).read())
except OSError as e:
    print('read blocked', type(e).__name__)
open({mark!r}, 'w').write('x')            # lands in the sandbox's private home, not the real one
print('write done')
""")
        assert "secret" not in result.stdout and "read blocked FileNotFoundError" in result.stdout, result.stdout + result.stderr
        assert not os.path.exists(mark)
    finally:
        os.remove(secret)


def test_the_system_is_read_only():
    result = run("""
try:
    open('/etc/phost_x', 'w')
    print('WRITTEN')
except OSError as e:
    print('blocked', type(e).__name__)
""")
    assert "blocked" in result.stdout and "WRITTEN" not in result.stdout


def test_the_environment_is_scrubbed(monkeypatch):
    monkeypatch.setenv("PHOST_SECRET_TOKEN", "hunter2")
    result = run("import os; print(os.environ.get('PHOST_SECRET_TOKEN', 'absent'))")
    assert result.stdout.strip() == "absent"


def test_the_wall_clock_limit_kills_the_whole_tree():
    marker = f"phost_marker_{uuid.uuid4().hex}"
    started = time.perf_counter()
    result = sandbox.run([*PY, f"import subprocess, time\nsubprocess.Popen(['sleep', '300', '{marker}'])\nwhile True: pass"],
                         policy=Policy(wall_seconds=3, cpu_seconds=100))
    assert result.timed_out and time.perf_counter() - started < 15
    time.sleep(0.5)
    assert subprocess.run(["pgrep", "-f", marker], capture_output=True).returncode == 1       # nothing left running


def test_the_cpu_limit_stops_a_busy_loop_before_the_wall_clock():
    started = time.perf_counter()
    result = run("while True: pass", cpu_seconds=2, wall_seconds=60)
    assert not result.timed_out and result.returncode != 0 and time.perf_counter() - started < 20


def test_the_file_size_limit_is_enforced(tmp_path):
    limit = sandbox.MIN_COMPILE_FILE_MB
    result = sandbox.run([*PY, f"open('/mnt/big', 'wb').write(b'x' * ({limit + 32} << 20))"],
                         policy=Policy(max_file_mb=limit, scratch_mb=limit, scratch=str(tmp_path)))
    assert result.returncode != 0 and (not (tmp_path / "big").exists() or (tmp_path / "big").stat().st_size <= limit << 20)


def test_a_policy_below_the_compiler_floor_is_refused():
    """Under it Warp's compiler cannot build any kernel, so every submission would fail for a reason that looks like its own."""
    with pytest.raises(ValueError, match="smallest per-file limit"):
        Policy(max_file_mb=sandbox.MIN_COMPILE_FILE_MB - 1)
    Policy(max_file_mb=sandbox.MIN_COMPILE_FILE_MB, scratch_mb=sandbox.MIN_COMPILE_FILE_MB)      # the floor itself is fine


def test_a_scratch_quota_smaller_than_one_file_is_refused():
    with pytest.raises(ValueError, match="one file could not fit"):
        Policy(max_file_mb=512, scratch_mb=256)


def test_many_small_files_are_stopped_by_the_scratch_quota_not_the_per_file_limit(tmp_path):
    """Each file is far under the per-file limit; together they would fill the disk."""
    writer = ("import os, time\n"
              "i = 0\n"
              "while True:\n"
              "    open(f'/mnt/f{i}', 'wb').write(b'x' * (8 << 20)); i += 1; time.sleep(0.02)\n")
    started = time.perf_counter()
    result = sandbox.run([*PY, writer], policy=Policy(max_file_mb=128, scratch_mb=128, wall_seconds=60, scratch=str(tmp_path)))
    assert result.quota_exceeded and not result.timed_out and result.returncode != 0
    assert time.perf_counter() - started < 30
    written = sum(f.stat().st_size for f in tmp_path.iterdir() if f.name.startswith("f"))
    assert 128 << 20 < written < 400 << 20                      # stopped soon after the quota, not at the wall clock


def test_a_program_under_the_quota_is_not_flagged(tmp_path):
    result = sandbox.run([*PY, "open('/mnt/ok', 'wb').write(b'x' * (4 << 20))"],
                         policy=Policy(max_file_mb=128, scratch_mb=128, scratch=str(tmp_path)))
    assert result.returncode == 0 and not result.quota_exceeded


def test_a_warm_cache_copied_in_does_not_count_against_the_quota(tmp_path):
    warm = tmp_path / "warm" / "1.0.0"
    warm.mkdir(parents=True)
    (warm / "cache.bin").write_bytes(b"x" * (200 << 20))                 # larger than the quota on its own
    result = sandbox.run([*PY, "print('ran')"], policy=Policy(max_file_mb=128, scratch_mb=128, warm_cache=str(warm),
                                                              scratch=str(tmp_path / "scratch")))
    assert result.returncode == 0 and not result.quota_exceeded


def test_the_scratch_directory_is_writable_and_survives_when_given(tmp_path):
    sandbox.run([*PY, "open('/mnt/kept', 'w').write('yes')"], policy=Policy(scratch=str(tmp_path)))
    assert (tmp_path / "kept").read_text() == "yes"


def test_a_warm_cache_is_copied_not_shared(tmp_path):
    warm, scratch = tmp_path / "1.2.3", tmp_path / "scratch"                 # named for a Warp version, like the real one
    warm.mkdir(); scratch.mkdir()
    (warm / "kernel.bin").write_text("trusted")
    sandbox.run([*PY, "print(open('/mnt/warp/1.2.3/kernel.bin').read()); open('/mnt/warp/1.2.3/kernel.bin', 'w').write('poisoned'); open('/mnt/warp/1.2.3/new', 'w')"],
                policy=Policy(scratch=str(scratch), warm_cache=str(warm)))
    assert (warm / "kernel.bin").read_text() == "trusted" and not (warm / "new").exists()      # the original is untouched
    assert (scratch / "warp" / "1.2.3" / "kernel.bin").read_text() == "poisoned"


def test_it_refuses_to_run_without_isolation_unless_told(monkeypatch):
    monkeypatch.setattr(sandbox, "available", lambda: False)
    with pytest.raises(SandboxUnavailable):
        sandbox.run([*PY, "print(1)"])
    result = sandbox.run([*PY, "print(1)"], allow_unsandboxed=True)
    assert result.sandbox == "none" and result.stdout.strip() == "1"
