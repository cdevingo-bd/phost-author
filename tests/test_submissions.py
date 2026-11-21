"""User-written Newton solvers, submitted as source, judged in the sandbox on the real MuJoCo Warp host."""

import importlib.resources
import os
import uuid

import pytest

from phost_author import sandbox
from phost_author.sandbox import Policy
from phost_author.submit import Submission, check_submission

try:
    import mujoco_warp  # noqa: F401
    import warp as wp
    HAVE = wp.is_cuda_available() and sandbox.available()
except Exception:                                    # pragma: no cover
    HAVE = False

pytestmark = pytest.mark.skipif(not HAVE, reason="needs mujoco_warp, CUDA and bwrap")

POLICY = Policy(wall_seconds=240, cpu_seconds=200)


def template() -> str:
    return importlib.resources.files("phost_author").joinpath("templates/newton_solver_template.py").read_text()


@pytest.fixture(scope="module", autouse=True)
def reviewed():
    from phost.hosts import mjwarp
    from phost import GuardError
    try:
        mjwarp.MJWarpHost.stages  # noqa: B018
        from phost import source_hash
        if source_hash(mjwarp._solver_module()._cholesky_factorize_solve) not in mjwarp.REVIEWED_HOOKS:
            raise GuardError("unreviewed")
    except GuardError:
        pytest.skip("this mujoco_warp build is not in the reviewed tables")


def submit(source, **kw):
    return check_submission(Submission(source), policy=POLICY, **kw)


def with_solve_body(body: str) -> str:
    """A runtime class with a custom solve(); the setup is the template's."""
    return template().split("class Runtime:")[0] + f'''
class Runtime:
    def __init__(self, host, params):
        self.nv = int(host.model.nv)
        self.host = host
        self.factor = wp.zeros((int(host.data.nworld), self.nv, self.nv), dtype=float, device=host.device)
        self.y = wp.zeros((int(host.data.nworld), self.nv), dtype=float, device=host.device)

    def solve(self, system):
{body}

    def metrics(self):
        return {{"status": 0, "fallbacks": 0}}
'''


def test_a_correct_user_solver_passes_the_gate():
    report = submit(template())
    assert report["ok"], report
    assert report["sandbox"] == "bwrap"
    assert report["verdict"]["metrics"]["max_relative_residual"] < 1e-4
    assert report["verdict"]["failures"] == []


def test_a_solver_that_ignores_the_coupling_fails_on_accuracy():
    wrong = template().replace(
        "            t = h[w, j, i]\n", "            t = float(0.0)\n")           # drops every off-diagonal entry of H
    assert wrong != template()
    report = submit(wrong)
    assert report["verdict"] is not None, report["error"]
    assert not report["ok"]
    assert any("max_relative_residual" in f for f in report["verdict"]["failures"]), report["verdict"]


def test_a_solver_that_modifies_its_input_fails():
    meddler = with_solve_body('''        wp.launch(cholesky_solve, dim=system.active.shape[0],
                  inputs=[system.h, system.grad, system.active, self.nv, self.factor, self.y],
                  outputs=[system.search, system.search_dot, system.newton_decrement], device=system.h.device)
        wp.launch(poke, dim=1, inputs=[system.grad], device=system.h.device)''').replace(
        "class Runtime:", '''@wp.kernel
def poke(grad: wp.array2d[float]):
    grad[0, 0] = grad[0, 0] + 1.0


class Runtime:''', 1)
    report = submit(meddler)
    assert not report["ok"] and any("modified" in f for f in report["verdict"]["failures"]), report


def test_a_solver_that_writes_inactive_worlds_fails():
    writer = with_solve_body('''        wp.launch(cholesky_solve, dim=system.active.shape[0],
                  inputs=[system.h, system.grad, system.active, self.nv, self.factor, self.y],
                  outputs=[system.search, system.search_dot, system.newton_decrement], device=system.h.device)
        wp.launch(scribble, dim=system.active.shape[0], inputs=[], outputs=[system.search_dot], device=system.h.device)''').replace(
        "class Runtime:", '''@wp.kernel
def scribble(search_dot: wp.array[float]):
    search_dot[wp.tid()] = 0.0


class Runtime:''', 1)
    report = submit(writer)
    assert not report["ok"] and any("inactive" in f for f in report["verdict"]["failures"]), report


def test_a_crashing_solver_is_reported_not_raised():
    report = submit(with_solve_body("        raise RuntimeError('boom from user code')"))
    assert not report["ok"] and "boom from user code" in report["error"]


def test_a_missing_entry_point_is_reported():
    report = submit("x = 1\n")
    assert not report["ok"] and "no 'Runtime'" in report["error"]


def test_a_syntax_error_is_reported():
    report = submit("def broken(:\n")
    assert not report["ok"] and "SyntaxError" in report["error"]


def test_a_hanging_solver_is_stopped_by_the_wall_clock():
    hang = with_solve_body("        while True:\n            pass")
    report = check_submission(Submission(hang), policy=Policy(wall_seconds=60, cpu_seconds=100))
    assert not report["ok"] and report["timed_out"]


def test_a_solver_cannot_reach_the_network_or_the_users_files():
    secret = os.path.join(os.path.expanduser("~"), f".phost_canary_{uuid.uuid4().hex}")
    with open(secret, "w") as stream:
        stream.write("secret")
    try:
        spy = with_solve_body(f'''        import socket
        try:
            socket.create_connection(("1.1.1.1", 53), timeout=2)
            raise AssertionError("NETWORK REACHED")
        except OSError:
            pass
        try:
            open({secret!r}).read()
            raise AssertionError("SECRET READ")
        except FileNotFoundError:
            pass
        raise RuntimeError("isolation held")''')
        report = submit(spy)
    finally:
        os.remove(secret)
    assert not report["ok"] and "isolation held" in report["error"], report          # both attempts were blocked


def test_the_report_says_where_the_verdict_was_computed():
    assert "own process" in submit(template())["integrity"]
