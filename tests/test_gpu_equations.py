"""Typed equations on the real MuJoCo Warp host. Needs a reviewed build, CUDA and MuJoCo."""

import numpy as np
import pytest

try:
    import mujoco_warp  # noqa: F401
    import warp as wp
    HAVE = wp.is_cuda_available()
except Exception:                                    # pragma: no cover
    HAVE = False

pytestmark = pytest.mark.skipif(not HAVE, reason="needs mujoco_warp and CUDA")

if HAVE:
    from phost import GuardError, Registry, Selection
    from phost.impls import controllers, forces, register_builtin
    from phost.stages import control_law as cl
    from phost.stages import dof_passive_force as pf
    from phost.stages import register_all
    from phost.testing.contract import check_impl
    from phost_author import expression_impls

    from support import QPOS, cpu_truth, error, make_host, run_gpu

PF, CL = f"{pf.NAME}/expression", f"{cl.NAME}/expression"


@pytest.fixture
def registry():
    reg = Registry()
    register_all(reg)
    register_builtin(reg)
    expression_impls.register(reg)
    return reg


@pytest.fixture(scope="module", autouse=True)
def reviewed():
    host = make_host()[0]
    try:
        for name in (pf.NAME, cl.NAME):
            host.stages()[name].point.guards["target"].check()
    except GuardError:
        pytest.skip("this mujoco_warp build is not in the reviewed tables")


EQUATIONS = [
    (PF, "-a*(q - b) - c*v", dict(a=3.0, b=0.2, c=0.4)),
    (PF, "-a*q - b*q**3 + c*sin(d*time)", dict(a=2.0, b=5.0, c=0.7, d=3.0)),
    (PF, "where(q > 0, -a*q, -b*q) - c*tanh(v)", dict(a=1.0, b=4.0, c=0.3)),
    (PF, "clamp(-a*q, -b, b)", dict(a=10.0, b=0.5)),
    (CL, "a*(b - length) - c*velocity", dict(a=5.0, b=0.4, c=0.3)),
    (CL, "clamp(a*sin(b*time) - c*length, -1, 1)", dict(a=0.8, b=2.0, c=1.0)),
]


@pytest.mark.parametrize("impl,text,constants", EQUATIONS)
def test_generated_kernels_pass_the_contract_on_the_real_host(registry, impl, text, constants):
    host = make_host()[0]
    stage = impl.rsplit("/", 1)[0]
    verdict = check_impl(host, Selection(stage, impl, dict(expr=text, **constants)), registry)
    assert verdict.ok, verdict.failures
    assert verdict.metrics["max_law_error"] < 1e-5            # float32 kernel against the float64 evaluation of the same text


def test_a_typed_spring_changes_the_simulation_to_what_mujoco_computes(registry):
    """The text '-a*q' becomes a kernel; the motion it produces matches MuJoCo with that force applied explicitly."""
    a = 6.0
    truth = cpu_truth(150, force=lambda q, v, t: -a * q)
    typed = run_gpu(150, [Selection(pf.NAME, PF, dict(expr="-a*q", a=a))], registry)
    plain = run_gpu(150)
    assert error(typed, truth) < 1e-5
    assert error(plain, truth) > 100 * max(error(typed, truth), 1e-6)         # and it is not the unforced motion


def test_a_typed_nonlinear_time_dependent_law_matches_mujoco(registry):
    a, b, c, w = 4.0, 30.0, 2.0, 5.0
    truth = cpu_truth(200, force=lambda q, v, t: -a * q - b * q**3 + c * np.sin(w * t))
    typed = run_gpu(200, [Selection(pf.NAME, PF, dict(expr="-a*q - b*q**3 + c*sin(d*time)", a=a, b=b, c=c, d=w))], registry)
    assert error(typed, truth) < 5e-5


def test_the_expression_pd_equals_the_builtin_pd(registry):
    typed = run_gpu(150, [Selection(cl.NAME, CL, dict(expr="a*(b - length) - c*velocity", a=4.0, b=0.2, c=0.3))], registry)
    built = run_gpu(150, [Selection(cl.NAME, controllers.PD_ID, dict(kp=4.0, kd=0.3, target=0.2))], registry)
    assert error(typed, built) < 1e-6


def test_changing_a_constant_does_not_recompile(registry):
    from phost_author import expr
    import sys
    run_gpu(5, [Selection(pf.NAME, PF, dict(expr="-a*q", a=1.0))], registry)
    modules = [m for m in sys.modules if m.startswith("phost_expr_")]
    run_gpu(5, [Selection(pf.NAME, PF, dict(expr="-a*q", a=7.0))], registry)
    assert [m for m in sys.modules if m.startswith("phost_expr_")] == modules       # same module, new constant


def test_a_typed_damper_lands_where_the_models_own_damper_does(registry):
    """'-a*v' reports its derivative, so Euler integrates it implicitly. It must be as close to MuJoCo's native joint
    damper as MJWarp's own native damper is, where a plain force is 3e-4 away at this damping and timestep."""
    import mujoco
    from support import XML
    c = 0.9
    damped_xml = XML.replace('axis="0 1 0"', f'axis="0 1 0" damping="{c}"').replace('axis="0 0 1"', f'axis="0 0 1" damping="{c}"')
    model = mujoco.MjModel.from_xml_string(damped_xml)
    data = mujoco.MjData(model)
    data.qvel[6:9] = [1.5, -1.0, 0.8]
    start = data.qvel.copy()
    for _ in range(150):
        mujoco.mj_step(model, data)
    truth = data.qpos.copy()

    def gpu(selections, xml=XML):
        cm = mujoco.MjModel.from_xml_string(xml)
        cd = mujoco.MjData(cm)
        cd.qvel[:] = start
        mujoco.mj_forward(cm, cd)
        import mujoco_warp as mjw
        from phost import Session
        from phost.testing.mjwarp import DrivableMJWarpHost
        with wp.ScopedDevice("cuda:0"):
            m = mjw.put_model(cm)
            d = mjw.put_data(cm, cd, nworld=1, nconmax=16, njmax=128)
            host = DrivableMJWarpHost(m, d, cpu_model=cm, cpu_data=cd)
            with Session(host, selections, registry):
                for _ in range(150):
                    mjw.step(m, d)
            wp.synchronize()
            return d.qpos.numpy()[0].astype(np.float64)

    typed = gpu([Selection(pf.NAME, PF, dict(expr="-a*v", a=c))])
    explicit = gpu([Selection(pf.NAME, PF, dict(expr="-a*v", a=c, implicit=False))])
    control = gpu([], xml=damped_xml)
    assert error(typed, truth) <= 3 * error(control, truth) + 1e-6, (error(typed, truth), error(control, truth))
    assert error(explicit, truth) > 10 * max(error(typed, truth), 1e-6)         # the explicit law really is further away
