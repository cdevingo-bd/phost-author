"""Check a user-written runtime against its stage's contract, in a sandbox.

A submission is Python source that defines a class (``Runtime`` by default) with the stage's runtime methods, for
example ``solve(system)`` and ``metrics()`` for ``newton.linear_solve``. Stages whose contract says
``reference="impl"`` also need a module-level ``reference(inputs, params)`` that returns the float64 outputs.
The same ``check_impl`` that gates any implementation judges it; the result says how it was run.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping

from . import sandbox
from .worker import PREFIX


def default_warm_cache() -> str | None:
    """The user's Warp kernel cache for the installed Warp version, if there is one (trusted: written by the parent)."""
    try:
        from importlib.metadata import version
        path = os.path.join(os.path.expanduser("~"), ".cache", "warp", version("warp-lang"))
    except Exception:
        return None
    return path if os.path.isdir(path) else None


@dataclass(frozen=True)
class Submission:
    source: str
    stage: str = "newton.linear_solve"
    entry: str = "Runtime"
    params: Mapping[str, Any] = field(default_factory=dict)


def check_submission(submission: Submission, *, scene: str = "trampoline8:impact", worlds: int = 2,
                     policy: sandbox.Policy = sandbox.Policy(), allow_unsandboxed: bool = False) -> dict:
    """Returns ``{ok, verdict, error, sandbox, seconds, ...}``. ``ok`` is True only if the check ran and passed.

    A hostile or broken submission yields ``ok: False`` with a reason; it does not raise.
    """
    if policy.warm_cache is None:
        policy = dataclasses.replace(policy, warm_cache=default_warm_cache())
    request = dict(source=submission.source, stage=submission.stage, entry=submission.entry,
                   params=dict(submission.params), scene=scene, worlds=worlds)
    result = sandbox.run([sys.executable, "-s", "-m", "phost_author.worker"], stdin=json.dumps(request), policy=policy,
                         allow_unsandboxed=allow_unsandboxed)
    report: dict[str, Any] = dict(ok=False, verdict=None, error=None, sandbox=result.sandbox,
                                  seconds=round(result.seconds, 2), timed_out=result.timed_out,
                                  quota_exceeded=result.quota_exceeded,
                                  integrity="the verdict is computed in the submission's own process")
    if result.timed_out:
        report["error"] = f"timed out after {policy.wall_seconds:.0f} s"
        return report
    if result.quota_exceeded:
        report["error"] = f"wrote more than {policy.scratch_mb} MB to its scratch directory"
        return report
    lines = [line for line in result.stdout.splitlines() if line.startswith(PREFIX)]
    if not lines:
        report["error"] = f"the worker exited with code {result.returncode} and no result"
        report["stderr"] = result.stderr[-2000:]
        return report
    payload = json.loads(lines[-1][len(PREFIX):])
    if "error" in payload:
        report["error"] = payload["error"]
        report["traceback"] = payload.get("traceback", "")
        report["stderr"] = result.stderr[-3000:]                 # compiler messages land here
        return report
    report["verdict"] = payload["verdict"]
    report["ok"] = bool(payload["verdict"]["ok"])
    return report
