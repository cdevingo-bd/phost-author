"""Typed equations beyond one scalar law: several outputs, arrays and reductions, and a typed solver step.

Everything runs on the fake host (NumPy), judged by the same ``check_impl`` as any implementation; the GPU file runs the same
texts on MuJoCo Warp.
"""

import ast

import numpy as np
import pytest

from phost import CapabilityError, ContractError, Registry, Selection, Session
from phost.impls import register_builtin
from phost.stages import body_wrench as bw
from phost.stages import effective_solve as es
from phost.stages import newton_linear as nl
from phost.stages import probe_state as ps
from phost.stages import register_all
from phost.testing.contract import check_impl
from phost.testing.fake import FakeHost
from phost_author import ExpressionError, expr, expression_impls, laws, program
from phost_author.backends import BACKENDS, backend_for, register_backend


@pytest.fixture
def registry():
    reg = Registry()
    register_all(reg)
    register_builtin(reg)
    expression_impls.register(reg)
    return reg


def run(registry, stage, **params):
    return check_impl(FakeHost(nv=12, world_count=3, nu=3), Selection(stage, f"{stage}/expression", params), registry)


# ---- the grammar: arrays, reductions, indices ----------------------------------------------------------

SCOPE = expr.Scope(scalars=("time",), vectors={"qpos": "nq", "qvel": "nv", "qacc": "nv"}, reductions=True, subscripts=True)


def axis(text, scope=SCOPE):
    return expr.parse_in(text, scope).axis


def test_an_expression_has_a_type_that_follows_its_names():
    assert axis("time + 1") is None and axis("qvel") == "nv" and axis("qvel*qvel + qacc") == "nv" and axis("2*qpos") == "nq"
    assert axis("sum(qvel*qvel)") is None and axis("norm(qvel) + qpos[0]") is None
    assert axis("dot(qvel, qacc)") is None and axis("where(qvel > 0, qvel, -qvel)") == "nv"


@pytest.mark.parametrize("text,message", [
    ("qpos + qvel", "cannot combine an array along 'nq' with one along 'nv'"),
    ("dot(qpos, qvel)", "cannot combine"),
    ("sum(time)", "needs an array"),
    ("sum(1.0)", "needs an array"),
    ("sum(qvel, qvel)", "takes 1 argument"),
    ("dot(qvel)", "takes 2 arguments"),
    ("qvel[-1]", "whole number"),
    ("qvel[1.5]", "whole number"),
    ("qvel[time]", "whole number"),
    ("qvel[0:2]", "whole number"),
    ("time[0]", "only name.number. is allowed"),
    ("(qvel + 1)[0]", "only name.number. is allowed"),
    ("A @ qvel", "no matrix"),
    ("qvel.real", "Attribute"),
    ("lambda: 1", "Lambda"),
])
def test_array_expressions_that_make_no_sense_are_refused_with_a_reason(text, message):
    with pytest.raises(ExpressionError, match=message):
        expr.parse_in(text, SCOPE)


def test_reductions_and_indices_are_off_unless_the_stage_allows_them():
    plain = expr.Scope(scalars=("q",))
    for text in ("sum(q)", "q[0]"):
        with pytest.raises(ExpressionError):
            expr.parse_in(text, plain)


def test_the_matrix_is_only_ever_applied_to_a_named_array():
    scope = expr.Scope(scalars=("n",), vectors={"x": "n"}, reductions=True, matrix="A", matrix_axis="n")
    assert expr.parse_in("A @ x", scope).axis == "n"
    for text, message in (("A", "can only be used as"), ("A @ (x + 1)", "needs the name of an array"), ("x @ x", "only A @ name"),
                          ("A @ A", "needs the name"), ("A @ y", "unknown name|needs the name")):
        with pytest.raises(ExpressionError, match=message):
            expr.parse_in(text, scope)


def test_reductions_evaluate_like_numpy_over_the_last_axis_and_broadcast_per_world():
    rng = np.random.default_rng(1)
    v, q = rng.standard_normal((4, 7)), rng.standard_normal((4, 5))
    values = dict(qvel=v, qpos=q, time=np.ones((4, 1)))
    ev = lambda text: expr.evaluate(expr.parse_in(text, SCOPE), values)               # noqa: E731
    assert np.allclose(ev("sum(qvel)"), v.sum(1, keepdims=True)) and np.allclose(ev("mean(qvel)"), v.mean(1, keepdims=True))
    assert np.allclose(ev("norm(qvel)"), np.linalg.norm(v, axis=1, keepdims=True))
    assert np.allclose(ev("maxabs(qvel)"), np.abs(v).max(1, keepdims=True))
    assert np.allclose(ev("amax(qvel) - amin(qvel)"), (v.max(1) - v.min(1))[:, None])
    assert np.allclose(ev("qpos[3]*time"), q[:, 3:4]) and np.allclose(ev("dot(qvel, qvel)"), (v * v).sum(1, keepdims=True))
    assert np.allclose(ev("sum(qvel - mean(qvel))"), 0.0, atol=1e-12)                # a reduction inside a reduction
    with pytest.raises(ExpressionError, match="only 5 entries"):
        ev("qpos[9]")


# ---- the text a person types for a form --------------------------------------------------------------------

def test_a_single_output_law_may_be_bare_or_assigned(registry):
    form = registry.equation("control.law")
    bare = laws.parse_pointwise("a*(b - length)", form)
    assert list(bare) == ["ctrl"] and laws.parse_pointwise("ctrl = a*(b - length)", form)["ctrl"].key == bare["ctrl"].key
    assert laws.parse_pointwise("ctrl = a*length; ", form)


def test_a_law_with_several_outputs_takes_one_line_each_and_the_rest_are_zero(registry):
    form = registry.equation(bw.NAME)
    parsed = laws.parse_pointwise("fx = -c*vx\nfz = -9.81*a; tz = c*wz", form)
    assert list(parsed) == ["fx", "fz", "tz"]


@pytest.mark.parametrize("text,message", [
    ("-c*vx", "write one line per output"),
    ("fq = 1", "not an output of this stage"),
    ("fx = 1\nfx = 2", "assigned twice"),
    ("fx, fy = 1, 2", "one name on the left"),
    ("fx += 1", "AugAssign is not allowed"),
    ("import os", "Import is not allowed"),
    ("fx = vq", "unknown name 'vq'"),
    ("fx = ", "syntax error"),
    ("", "empty"),
    ("fx = 1\n" * 17, "limit is 16"),
])
def test_bad_multi_output_text_is_refused_with_a_reason(registry, text, message):
    with pytest.raises(ExpressionError, match=message):
        laws.parse_pointwise(text, registry.equation(bw.NAME))


def test_channels_are_labelled_by_their_line_or_by_position(registry):
    form, contract = registry.equation(ps.NAME), registry.contract(ps.NAME)
    parsed = laws.parse_channels("speed = norm(qvel)\nsum(qacc)\nq0 = qpos[0]", form, contract)
    assert [label for label, _ in parsed] == ["speed", "c1", "q0"]


@pytest.mark.parametrize("text,message", [
    ("qvel", "array along 'nv'"),
    ("x = qpos + qvel", "cannot combine"),
    ("x1 = 1\nx1 = 2", "labelled 'x1'"),
    ("a = 1", "already uses"),
    ("qvel = 1", "already uses"),
    ("time = 1", "already uses"),
    ("_x = 1", "already uses"),
])
def test_bad_channels_are_refused_with_a_reason(registry, text, message):
    with pytest.raises(ExpressionError, match=message):
        laws.parse_channels(text, registry.equation(ps.NAME), registry.contract(ps.NAME))


def test_an_index_outside_the_model_is_refused_once_the_sizes_are_known(registry):
    form, contract = registry.equation(ps.NAME), registry.contract(ps.NAME)
    parsed = laws.parse_channels("far = qvel[30]", form, contract)
    laws.check_indices(parsed, {"nv": 31}, contract, form)
    with pytest.raises(ContractError, match=r"qvel\[30\] is out of range, qvel has 12 entries"):
        laws.check_indices(parsed, {"nv": 12}, contract, form)


# ---- a typed solver step -----------------------------------------------------------------------------

def spd(n, seed=0, condition=50.0):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return ((q * np.geomspace(1.0, condition, n)) @ q.T)[None], rng.standard_normal((1, n))


def solve(init, step, iterations, n=8, **constants):
    compiled = program.compile_program(init, step)
    a, b = spd(n)
    given = {c: constants.get(c, 0.0) for c in expr.CONSTANTS}
    return program.run(compiled, a, b, given, iterations)[0], np.linalg.solve(a[0], b[0])


def test_conjugate_gradients_written_as_a_program_is_exact_in_n_passes():
    x, truth = solve(program.DEFAULT_INIT, program.DEFAULT_STEP, 8)
    assert np.allclose(x, truth, atol=1e-8)


def test_a_damped_jacobi_step_is_one_line_and_improves_with_passes():
    errors = [np.linalg.norm(solve("", "x = x + a*(rhs - A@x)/diag", k, a=0.5)[0] - solve("", "x = x + a*(rhs - A@x)/diag", 1, a=0.5)[1])
              for k in (1, 20, 200)]
    assert errors[0] > errors[1] > errors[2]


def test_k_and_n_are_available_to_a_program():
    x, _ = solve("", "x = x + (k + 1)*rhs/n/diag", 1)
    a, b = spd(8)
    assert np.allclose(x, b[0] / 8 / np.diag(a[0]))


@pytest.mark.parametrize("init,step,message", [
    ("", "", "step is empty"),
    ("", "y = rhs", "never assigns x"),
    ("", "x = rhs\nx = 1.0", "x is the solution and must be an array|was an array"),
    ("", "x = dot(rhs, rhs)", "must be an array"),
    ("rhs = 1", "x = rhs", "part of the language"),
    ("", "a = 2\nx = rhs", "part of the language"),
    ("", "A = rhs", "part of the language"),
    ("", "x = A", "can only be used as"),
    ("", "x = ghost + rhs", "unknown name 'ghost'"),
    ("", "x = rhs\ny = 1.0\ny = rhs", "was a number and this makes it an array"),
    ("", "x = rhs\ny = rhs\ny = sum(rhs)", "was an array and this makes it a number"),
    ("", "rhs", "write  name = expression"),
    ("", "x = rhs\n_h = 1", "not a usable name"),
    ("", "x = rhs\n" + "\n".join(f"v{i} = rhs" for i in range(13)), "limit is 12|at most 12"),
])
def test_a_bad_program_is_refused_with_the_line_and_the_reason(init, step, message):
    with pytest.raises(ExpressionError, match=message):
        program.compile_program(init, step)


def test_a_name_must_be_set_before_a_pass_uses_it():
    with pytest.raises(ExpressionError, match="unknown name 'carry'"):
        program.compile_program("", "x = x + carry*rhs\ncarry = 1.0")
    assert program.compile_program("carry = 1.0", "x = x + carry*rhs\ncarry = 0.5*carry")


# ---- the implementations, judged as any other ---------------------------------------------------------

def test_a_wrench_with_a_few_outputs_passes_and_unwritten_components_are_zero(registry):
    host = FakeHost(nv=12, world_count=3)
    text = "fx = -c*vx\nfy = -c*vy\nfz = -c*vz - a\ntz = -0.1*c*wz"
    verdict = check_impl(host, Selection(bw.NAME, f"{bw.NAME}/expression", dict(expr=text, c=0.7, a=0.5)), registry)
    assert verdict.ok, verdict.failures
    with Session(host, [Selection(bw.NAME, f"{bw.NAME}/expression", dict(expr=text, c=0.7, a=0.5))], registry):
        host.drive(bw.NAME, seed=2)
        wrench = host.data.xfrc_applied
        assert not wrench[..., 3:5].any() and wrench[..., 5].any() and wrench[..., 2].any()


def test_a_wrench_that_writes_nothing_is_the_zero_wrench(registry):
    impl = registry.impl(f"{bw.NAME}/expression")
    out = impl.reference(dict(xpos=np.zeros((2, 3, 3)), cvel=np.ones((2, 3, 6)), time=np.zeros(2)),
                         impl.resolve_params(dict(expr="fx = 0")))
    assert out["wrench"].shape == (2, 3, 6) and not out["wrench"].any()


def test_a_probe_of_typed_channels_passes_and_reports_its_width(registry):
    text = "speed = norm(qvel)\nenergy = 0.5*sum(qvel*qvel)\nfirst = qpos[0]\nworst = maxabs(qacc)\nspread = amax(qvel) - amin(qvel)"
    host = FakeHost(nv=12, world_count=3)
    verdict = check_impl(host, Selection(ps.NAME, f"{ps.NAME}/expression", dict(expr=text)), registry)
    assert verdict.ok, verdict.failures
    with Session(host, [Selection(ps.NAME, f"{ps.NAME}/expression", dict(expr=text))], registry) as session:
        assert session.runtimes[ps.NAME].width == 5


def test_a_probe_that_indexes_past_the_model_is_refused_when_it_is_installed(registry):
    host = FakeHost(nv=12, world_count=3)
    with pytest.raises(ContractError, match=r"qvel\[400\] is out of range, qvel has 12 entries"):
        with Session(host, [Selection(ps.NAME, f"{ps.NAME}/expression", dict(expr="far = qvel[400]"))], registry):
            pass
    with Session(host, [Selection(ps.NAME, f"{ps.NAME}/expression", dict(expr="last = qvel[11]"))], registry):
        pass


@pytest.mark.parametrize("stage", [es.NAME, nl.NAME])
def test_the_default_solver_step_passes_its_own_tolerance_on_both_solves(registry, stage):
    verdict = run(registry, stage, iterations=40, tolerance=1e-3)
    assert verdict.ok, verdict.failures
    assert verdict.metrics["max_relative_residual"] < 1e-3 and verdict.metrics["max_program_error"] < 1e-6


@pytest.mark.parametrize("stage", [es.NAME, nl.NAME])
def test_a_solver_that_has_not_converged_fails_the_accuracy_it_claimed_not_the_exact_gate(registry, stage):
    verdict = run(registry, stage, iterations=1, tolerance=1e-6)
    assert not verdict.ok
    assert any("max_relative_residual" in f and "exceeds 1e-06" in f for f in verdict.failures)
    loose = run(registry, stage, iterations=1, tolerance=1.0)
    assert not any("max_relative_residual" in f for f in loose.failures)          # claiming less is allowed; it is ranked on it


def test_a_diverging_program_fails_as_not_finite(registry):
    verdict = run(registry, es.NAME, init="", step="x = x + 10*(rhs - A@x)/diag", iterations=200, tolerance=1.0)
    assert not verdict.ok and any("not finite" in f or "exceeds" in f for f in verdict.failures)


def test_the_solver_is_judged_as_approximate_in_the_registry(registry):
    for stage in (es.NAME, nl.NAME):
        impl = registry.impl(f"{stage}/expression")
        assert impl.swap_class.value == "approximate" and impl.title == "Typed solver step"
        assert [m.name for m in impl.metrics] == ["max_program_error"]
    assert registry.impl(f"{bw.NAME}/expression").swap_class is None


def test_newton_leaves_inactive_worlds_alone_and_derives_its_other_outputs(registry):
    host = FakeHost(nv=12, world_count=4)
    selection = Selection(nl.NAME, f"{nl.NAME}/expression", dict(iterations=40, tolerance=1e-3))
    verdict = check_impl(host, selection, registry, calls=[{"active": [1, 0, 1, 1]}])
    assert verdict.ok, verdict.failures


# ---- the backend boundary ----------------------------------------------------------------------------------

class Wrong:
    """A backend whose solver is off by a little: the residual might still look fine, the program error does not."""

    name = "wrong"

    def __init__(self):
        self.inner = backend_for("numpy")

    def pointwise(self, plan):
        return self.inner.pointwise(plan)

    def channels(self, plan):
        return self.inner.channels(plan)

    def iteration(self, plan):
        run = self.inner.iteration(plan)

        def go(system, constants):
            run(system, constants)
            system.qacc[...] *= 1.0 + 1e-2
        return go


def test_a_new_backend_is_one_registration_and_its_code_is_judged_against_the_typed_program(registry):
    register_backend(Wrong(), replace=True)
    try:
        host = FakeHost(nv=12, world_count=3)
        host.backend = "wrong"
        verdict = check_impl(host, Selection(es.NAME, f"{es.NAME}/expression", dict(iterations=40, tolerance=0.5)), registry)
        assert verdict.metrics["max_program_error"] > 1e-3
        assert not verdict.ok and any("max_program_error" in f for f in verdict.failures)
    finally:
        BACKENDS.pop("wrong", None)


def test_an_unknown_backend_is_a_clear_refusal_and_a_duplicate_name_is_refused():
    with pytest.raises(CapabilityError, match="no typed-equation backend for host arrays of kind 'torch'"):
        backend_for("torch")
    with pytest.raises(ValueError, match="already registered"):
        register_backend(type("Dup", (), dict(name="numpy"))())


# ---- the preview a person sees while typing --------------------------------------------------------------

def test_the_form_description_tells_a_person_what_they_can_type(registry):
    pointwise = expression_impls.describe_form(registry, bw.NAME)
    assert pointwise["kind"] == "pointwise" and pointwise["outputs"] == ["fx", "fy", "fz", "tx", "ty", "tz"]
    assert "wx" in pointwise["inputs"] and pointwise["slope"] is None
    probe = expression_impls.describe_form(registry, ps.NAME)
    assert probe["arrays"] == {"qpos": "nq", "qvel": "nv", "qacc": "nv"} and "norm" in probe["reductions"]
    solver = expression_impls.describe_form(registry, nl.NAME)
    assert solver["outputs"] == ["x"] and solver["default_step"] == program.DEFAULT_STEP
    with pytest.raises(ContractError, match="no typed-equation stage"):
        expression_impls.describe_form(registry, "nope")


def test_a_wrench_preview_has_a_curve_for_every_output(registry):
    out = expression_impls.preview(registry, bw.NAME, "fx = -a*vx\nfz = b", dict(a=2.0, b=3.0), sweep="vx", lo=-1, hi=1, n=3)
    assert out["kind"] == "pointwise" and out["sweep"] == "vx"
    assert out["outputs"]["fx"] == [2.0, 0.0, -2.0] and out["outputs"]["fz"] == [3.0] * 3 and out["outputs"]["ty"] == [0.0] * 3
    assert out["y"] == out["outputs"]["fx"] and "derivative" not in out


def test_a_probe_preview_evaluates_each_channel_on_a_sample_state(registry):
    out = expression_impls.preview(registry, ps.NAME, "n2 = dot(qvel, qvel)\nfirst = qpos[0]", size=6, seed=3)
    assert out["kind"] == "channels" and [c["label"] for c in out["channels"]] == ["n2", "first"] and out["finite"]
    rng = np.random.default_rng(3)
    qpos, qvel = rng.standard_normal(6), rng.standard_normal(6)
    assert out["channels"][0]["value"] == pytest.approx(float(qvel @ qvel)) and out["channels"][1]["value"] == pytest.approx(qpos[0])
    big = expression_impls.preview(registry, ps.NAME, "far = qvel[20]", size=6)
    assert big["sample"]["entries"] == 21                                   # the sample grows to fit the index typed


def test_a_solver_preview_shows_the_residual_after_every_pass(registry):
    out = expression_impls.preview(registry, es.NAME, init=program.DEFAULT_INIT, step=program.DEFAULT_STEP, iterations=30,
                                   size=10, condition=100.0)
    assert out["kind"] == "iteration" and len(out["residuals"]) == 30 and out["start"] == pytest.approx(1.0)
    assert out["final"] < 1e-6 and out["solution_error"] < 1e-5 and out["finite"]
    slow = expression_impls.preview(registry, es.NAME, init="", step="x = x + a*(rhs - A@x)/diag", iterations=5,
                                    constants=dict(a=0.5), size=10, condition=1e4)
    assert slow["final"] > 1e-2                                             # Jacobi on a stiff system: visibly not there yet


@pytest.mark.parametrize("kwargs", [dict(size=1), dict(size=65), dict(iterations=0), dict(iterations=201), dict(condition=0.5)])
def test_solver_previews_refuse_sizes_that_would_be_slow_or_meaningless(registry, kwargs):
    with pytest.raises(ContractError, match="a solver preview needs"):
        expression_impls.preview(registry, es.NAME, **kwargs)


def test_every_registered_typed_stage_has_exactly_one_expression_impl(registry):
    stages = registry.equation_stages()
    assert len(stages) == 6
    for stage in stages:
        assert [i.id for i in registry.impls_for(stage) if i.kind.value == "expression"] == [f"{stage}/expression"]
    assert ast.parse(program.DEFAULT_STEP) and ast.parse(program.DEFAULT_INIT)
