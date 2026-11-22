"""Typed equations as stage implementations: contract checks, then a simulation changed by an equation."""

import numpy as np
import pytest

from phost import ContractError, Registry, Selection, Session
from phost.impls import register_builtin
from phost.stages import control_law as cl
from phost.stages import dof_passive_force as pf
from phost.stages import register_all
from phost.testing.contract import check_impl
from phost.testing.fake import FakeHost
from phost_author import ExpressionError, expression_impls

PF, CL = f"{pf.NAME}/expression", f"{cl.NAME}/expression"


@pytest.fixture
def registry():
    reg = Registry()
    register_all(reg)
    register_builtin(reg)
    expression_impls.register(reg)
    return reg


@pytest.fixture
def host():
    return FakeHost(nv=9, world_count=3, nu=3)


def selection(stage_impl, expr, **constants):
    stage = stage_impl.rsplit("/", 1)[0]
    return Selection(stage, stage_impl, dict(expr=expr, **constants))


EQUATIONS = [
    (PF, "-a*(q - b) - c*v", dict(a=3.0, b=0.2, c=0.4)),
    (PF, "-a*q - b*q**3 + c*sin(d*time)", dict(a=2.0, b=5.0, c=0.7, d=3.0)),
    (PF, "where(q > 0, -a*q, -b*q) - c*tanh(v)", dict(a=1.0, b=4.0, c=0.3)),
    (PF, "clamp(-a*q, -b, b)", dict(a=10.0, b=0.5)),
    (CL, "a*(b - length) - c*velocity", dict(a=5.0, b=0.4, c=0.3)),
    (CL, "clamp(a*sin(b*time) - c*length, -1, 1)", dict(a=0.8, b=2.0, c=1.0)),
]


@pytest.mark.parametrize("impl,text,constants", EQUATIONS)
def test_equations_pass_the_stage_contract_on_the_fake_host(host, registry, impl, text, constants):
    verdict = check_impl(host, selection(impl, text, **constants), registry)
    assert verdict.ok, verdict.failures
    assert verdict.metrics["max_law_error"] < 1e-6


def test_the_reference_is_the_typed_equation(registry):
    impl = registry.impl(PF)
    q = np.array([[0.5, -1.0, 2.0]])
    inputs = dict(q=q, v=np.zeros((1, 3)), time=np.array([2.0]), scalar=np.array([1, 0, 1], np.int32))
    out = impl.reference(inputs, impl.resolve_params(dict(expr="a*q + b*time", a=3.0, b=10.0)))["force"]
    assert np.allclose(out, [[1.5 + 20.0, 0.0, 6.0 + 20.0]])        # the masked DOF gets 0


def test_a_bad_equation_is_refused_before_any_simulation(registry):
    impl = registry.impl(PF)
    for text in ("__import__('os')", "q.real", "x + 1", "q +", ""):
        with pytest.raises(ExpressionError):
            impl.resolve_params(dict(expr=text))
    with pytest.raises(ContractError):                                # a name that belongs to the other stage
        registry.impl(CL).resolve_params(dict(expr="q"))


def test_an_equation_that_produces_nan_fails_the_gate(host, registry):
    verdict = check_impl(host, selection(PF, "log(q - 100.0)"), registry)
    assert not verdict.ok and any("not finite" in f for f in verdict.failures)


def test_constants_change_the_result_without_changing_the_equation(host, registry):
    outs = []
    for a in (1.0, 2.0):
        with Session(host, [selection(PF, "-a*q", a=a)], registry):
            host.drive(pf.NAME)
        outs.append(host.data.qfrc_passive.copy())
    native = FakeHost(nv=9, world_count=3, nu=3)
    native.drive(pf.NAME)
    q = host.data.qpos[:, :9]
    scalar = host.model.scalar.astype(bool)
    assert np.allclose(outs[1] - native.data.qfrc_passive, np.where(scalar, -2.0 * q, 0.0), atol=1e-5)
    assert np.allclose(outs[0] - native.data.qfrc_passive, np.where(scalar, -1.0 * q, 0.0), atol=1e-5)


def test_warp_source_has_the_expected_shape(registry):
    from phost_author.backends.warp_backend import pointwise_source
    impl = registry.impl(PF)
    plan = expression_impls.pointwise_plan(registry.contract(pf.NAME), registry.equation(pf.NAME),
                                           impl.resolve_params(dict(expr="-a*q - c*v", implicit=False)))
    source, args = pointwise_source(plan)
    assert "def kernel(p_q: wp.array2d[float], p_v: wp.array2d[float], p_scalar: wp.array[int], a: float" in source
    assert "        p_force[w, i] = (((-a) * p_q[w, i]) - (c * p_v[w, i]))" in source
    assert "    else:\n        p_force[w, i] = 0.0" in source                # the mask: no force on non-scalar DOFs
    assert args == dict(inputs=["q", "v", "scalar"], outputs=["force"])        # only what the equation reads is passed
    compile(source, "<kernel>", "exec")


# -- damping of a typed law ---------------------------------------------------------------------------

def test_a_typed_damper_integrates_like_the_models_own_damper(registry):
    """'-c*v' differentiates to c, and the host integrates that implicitly, as it does model damping."""
    c = 0.8
    own, native = FakeHost(damping=0.0), FakeHost(damping=c)
    on = native.model.scalar.astype(bool)
    with Session(native, [Selection(pf.NAME, "dof.passive_force/native")], registry):
        want = native.integrate(seed=4)[:, on]
    with Session(own, [selection(PF, "-a*v", a=c)], registry):
        got = own.integrate(seed=4)[:, on]
    assert np.allclose(got, want, rtol=1e-5, atol=1e-6)


def test_a_nonlinear_typed_law_reports_its_state_dependent_damping(registry):
    """-a v |v| has damping 2 a |v|: not a constant, so only the symbolic derivative gets it right."""
    host = FakeHost()
    verdict = check_impl(host, selection(PF, "-a*v*abs(v)", a=0.6), registry)
    assert verdict.ok, verdict.failures
    assert verdict.metrics["max_damping_error"] < 1e-5
    impl = registry.impl(PF)
    v = np.array([[0.5, -2.0, 1.0]])
    inputs = dict(q=np.zeros((1, 3)), v=v, time=np.zeros(1), scalar=np.ones(3, np.int32))
    damping = impl.reference(inputs, impl.resolve_params(dict(expr="-a*v*abs(v)", a=0.6)))["damping"]
    assert np.allclose(damping, 2 * 0.6 * np.abs(v))


def test_implicit_false_reports_no_damping(registry):
    impl = registry.impl(PF)
    inputs = dict(q=np.zeros((1, 3)), v=np.ones((1, 3)), time=np.zeros(1), scalar=np.ones(3, np.int32))
    params = impl.resolve_params(dict(expr="-a*v", a=0.5, implicit=False))
    assert not impl.reference(inputs, params)["damping"].any()
    assert check_impl(FakeHost(), Selection(pf.NAME, PF, dict(expr="-a*v", a=0.5, implicit=False)), registry).ok


def test_the_control_law_has_no_implicit_parameter(registry):
    assert "implicit" not in [p.name for p in registry.impl(CL).params]
    assert registry.impl(CL).extensions_optional == ()


# ---- preview: what an equation does, without a simulation ----

def test_preview_samples_the_law_along_the_first_variable(registry):
    from phost_author.expression_impls import preview
    out = preview(registry, "dof.passive_force", "-a*q", dict(a=3.0), lo=-1, hi=1, n=5)
    assert out["variables"] == ["q", "v", "time"] and out["sweep"] == "q"
    assert out["x"] == [-1, -0.5, 0, 0.5, 1] and out["y"] == [3, 1.5, -0, -1.5, -3]
    assert out["finite"] and out["key"]


def test_preview_holds_the_other_variables_where_asked(registry):
    from phost_author.expression_impls import preview
    out = preview(registry, "dof.passive_force", "q*v", sweep="q", at=dict(v=2.0), lo=0, hi=1, n=3)
    assert out["y"] == [0, 1.0, 2.0]


def test_preview_gives_the_symbolic_damping_of_a_force_law(registry):
    from phost_author.expression_impls import preview
    out = preview(registry, "dof.passive_force", "-a*v - b*v**3", dict(a=2.0, b=1.0), sweep="v", lo=-1, hi=1, n=3)
    assert out["derivative"]["of"] == "v" and "v" in out["derivative"]["text"]
    # law = -a v - b v^3, so damping = -d law/dv = a + 3 b v^2 = 2 + 3 v^2
    assert out["derivative"]["damping"] == pytest.approx([5.0, 2.0, 5.0])


def test_a_control_law_has_no_derivative_entry(registry):
    from phost_author.expression_impls import preview
    assert "derivative" not in preview(registry, "control.law", "-a*length", dict(a=1.0), lo=0, hi=1, n=3)


def test_preview_reports_a_law_that_leaves_the_finite_numbers(registry):
    from phost_author.expression_impls import preview
    assert preview(registry, "dof.passive_force", "1/q", lo=-1, hi=1, n=3)["finite"] is False


@pytest.mark.parametrize("kwargs,message", [
    (dict(stage="nope", text="q"), "no typed-equation stage"),
    (dict(stage="dof.passive_force", text="q", sweep="zz"), "cannot sweep"),
    (dict(stage="dof.passive_force", text="q", lo=1, hi=1), "lo < hi"),
    (dict(stage="dof.passive_force", text="q", n=1), "lo < hi"),
    (dict(stage="dof.passive_force", text="q", lo=float("nan")), "lo < hi"),
])
def test_preview_refuses_bad_requests(registry, kwargs, message):
    from phost_author.expression_impls import preview
    with pytest.raises(ContractError, match=message):
        preview(registry, **kwargs)


def test_preview_refuses_an_unacceptable_equation_with_the_parser_message(registry):
    from phost_author.expression_impls import preview
    with pytest.raises(ExpressionError):
        preview(registry, "dof.passive_force", "__import__('os')")
