"""A submitted kernel is checked on the host of the scene's own engine, so it needs neither MuJoCo nor a GPU: here, the NumPy
chain. Needs bwrap (the sandbox), nothing else."""

import pytest

from phost_author import sandbox
from phost_author.sandbox import Policy
from phost_author.submit import Submission, check_submission

pytestmark = pytest.mark.skipif(not sandbox.available(), reason="needs bwrap")

POLICY = Policy(wall_seconds=120, cpu_seconds=100)

DENSE = '''
"""A dense solve of the effective system, in NumPy."""
import numpy as np


class Runtime:
    def __init__(self, host, params):
        pass

    def solve(self, system):
        h = np.asarray(system.h, dtype=np.float64)
        rhs = np.asarray(system.rhs, dtype=np.float64)
        n = system.qacc.shape[1]
        for w in range(h.shape[0]):
            upper = np.triu(h[w, :n, :n])
            system.qacc[w, :] = np.linalg.solve(upper + np.triu(upper, 1).T, rhs[w, :n]).astype(system.qacc.dtype)

    def metrics(self):
        return {}
'''

IGNORES_THE_COUPLING = DENSE.replace("np.linalg.solve(upper + np.triu(upper, 1).T, rhs[w, :n])", "rhs[w, :n] / np.diag(upper)")


def submit(source):
    return check_submission(Submission(source, "effective.solve", "Runtime", {}), scene="chain32:pluck", worlds=2, policy=POLICY)


def test_a_correct_kernel_is_checked_on_the_chain_engines_host():
    report = submit(DENSE)
    assert report["error"] is None and report["ok"], report
    assert report["scene"] == "chain32:pluck" and report["engine"] == "chain" and report["sandbox"] == "bwrap"


def test_a_wrong_kernel_fails_the_same_check_on_the_chain():
    report = submit(IGNORES_THE_COUPLING)
    assert report["error"] is None and not report["ok"], report
    assert any("max_relative_residual" in f or "max_native_difference" in f for f in report["verdict"]["failures"])
