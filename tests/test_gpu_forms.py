"""The typed forms on the real MuJoCo Warp host: several outputs into a vector-typed port, reductions over the state, and a
typed solver step whose generated kernel is judged against the same program in float64. Needs a reviewed build and CUDA."""

import numpy as np
import pytest

try:
    import mujoco_warp  # noqa: F401
    import warp as wp
    HAVE = wp.is_cuda_available()
except Exception:                                    # pragma: no cover
    HAVE = False

if not HAVE:                    # a module-level skip: the decorators below name things only a GPU build imports
    pytest.skip("needs mujoco_warp and CUDA", allow_module_level=True)

if HAVE:
    from phost import GuardError, Registry, Selection, Session
    from phost.impls import register_builtin
    from phost.stages import body_wrench as bw
    from phost.stages import effective_solve as es
    from phost.stages import newton_linear as nl
    from phost.stages import probe_state as ps
    from phost.stages import register_all
    from phost.testing.contract import check_impl
    from phost_author import expression_impls
    from phost_author.backends.warp_backend import layout_of
    from phost_author.worker import build_host

    from support import make_host


@pytest.fixture(scope="module")
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
        host.stages()[bw.NAME].point.guards["target"].check()
    except GuardError:
        pytest.skip("this mujoco_warp build is not in the reviewed tables")


def lab_host(stage, registry, worlds=2):
    from phost_lab import default_scenes
    from phost_lab.opportunity import scene_for
    scenes = default_scenes()
    return build_host(scenes, scene_for(stage, scenes, registry), worlds)


WRENCH = "fx = -c*vx\nfy = -c*vy\nfz = -c*vz - a\ntx = -0.1*c*wx\nty = -0.1*c*wy\ntz = -0.1*c*wz"


def test_a_typed_wrench_into_the_hosts_vector_typed_arrays_matches_its_law_and_the_hosts_effect(registry):
    host = make_host()[0]
    verdict = check_impl(host, Selection(bw.NAME, f"{bw.NAME}/expression", dict(expr=WRENCH, c=0.7, a=0.5)), registry)
    assert verdict.ok, verdict.failures
    assert verdict.metrics["max_law_error"] < 1e-5 and verdict.metrics["max_effect_error"] < 1e-4


def test_the_kernel_is_written_for_how_the_host_stores_the_port(registry):
    host = make_host()[0]
    with Session(host, [Selection(bw.NAME, f"{bw.NAME}/expression", dict(expr="fx = -c*vx", c=1.0))], registry) as session:
        host.drive(bw.NAME, seed=0)
        system = host.data
        assert layout_of(system.cvel).composite == "wp.spatial_vectorf" and layout_of(system.xpos).composite == "wp.vec3f"
        wrench = host.data.xfrc_applied.numpy()
        assert wrench[..., 0].any() and not wrench[..., 1:].any()                    # the other components were written as 0


def test_a_typed_wrench_moves_the_trajectory_and_a_zero_one_does_not(registry):
    from support import run_gpu
    plain = run_gpu(60)
    drag = run_gpu(60, [Selection(bw.NAME, f"{bw.NAME}/expression", dict(expr="fz = -a", a=3.0))], registry)
    nothing = run_gpu(60, [Selection(bw.NAME, f"{bw.NAME}/expression", dict(expr="fz = 0"))], registry)
    assert np.abs(drag - plain).max() > 1e-3 and np.abs(nothing - plain).max() < 1e-6


def test_typed_channels_over_the_state_pass_on_the_real_host(registry):
    host = make_host()[0]
    text = "speed = norm(qvel)\nenergy = 0.5*sum(qvel*qvel)\nfirst = qpos[0]\nworst = maxabs(qacc)\nspread = amax(qvel) - amin(qvel)"
    verdict = check_impl(host, Selection(ps.NAME, f"{ps.NAME}/expression", dict(expr=text)), registry)
    assert verdict.ok, verdict.failures
    assert verdict.metrics["max_law_error"] < 1e-5


def test_a_probe_index_past_the_model_is_refused_at_install_on_the_real_host(registry):
    from phost import ContractError
    host = make_host()[0]
    with pytest.raises(ContractError, match="is out of range"):
        with Session(host, [Selection(ps.NAME, f"{ps.NAME}/expression", dict(expr="far = qvel[500]"))], registry):
            pass


@pytest.mark.parametrize("stage", [es.NAME, nl.NAME])
def test_a_typed_conjugate_gradient_pass_judged_against_the_same_program_in_float64(registry, stage):
    host = lab_host(stage, registry)
    verdict = check_impl(host, Selection(stage, f"{stage}/expression", dict(iterations=60, tolerance=0.05)), registry)
    assert verdict.ok, verdict.failures
    assert verdict.metrics["max_relative_residual"] < 0.05 and verdict.metrics["max_program_error"] < 1e-3


def test_a_one_pass_jacobi_step_fails_the_accuracy_it_claimed_on_the_real_host(registry):
    host = lab_host(es.NAME, registry)
    verdict = check_impl(host, Selection(es.NAME, f"{es.NAME}/expression",
                                         dict(init="", step="x = x + a*(rhs - A@x)/diag", iterations=1, tolerance=1e-6, a=0.5)),
                         registry)
    assert not verdict.ok and any("max_relative_residual" in f for f in verdict.failures)
    assert verdict.metrics["max_program_error"] < 1e-3                  # the kernel does what was typed; what was typed is not enough
