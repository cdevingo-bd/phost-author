"""Typed equations on JAX arrays: the same texts, the same trees, the JAX emitter, eager and inside ``jax.jit``."""

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="needs jax")

from phost import Registry, Selection, Session
from phost.hosts.chain import ChainHost, make_model
from phost.hosts.jaxchain import JaxChainHost
from phost.impls import register_builtin
from phost.stages import control_law as cl
from phost.stages import dof_passive_force as pf
from phost.stages import effective_solve as es
from phost.stages import probe_state as ps
from phost.stages import register_all
from phost.testing.contract import check_impl
from phost_author import expression_impls
from phost_author.backends import BACKENDS, backend_for


@pytest.fixture
def registry():
    reg = Registry()
    register_all(reg)
    register_builtin(reg)
    expression_impls.register(reg)
    return reg


def make(cls=JaxChainHost, n=24, nu=3, worlds=3, **kw):
    return cls(make_model(n, nu, **kw), worlds=worlds)


def typed(stage, **params):
    return Selection(stage, f"{stage}/expression", params)


def pluck(host, at=6, size=0.1):
    host.load(np.zeros(host.model.nv))
    host.data.x[:, at] = size


def test_the_jax_emitter_is_registered_beside_numpy_and_warp():
    assert backend_for("jax").name == "jax" and backend_for("numpy").name == "numpy" and {"numpy", "jax"} <= set(BACKENDS)


@pytest.mark.parametrize("selection", [
    typed(pf.NAME, expr="-a*q - b*v + c*sin(time)", a=3.0, b=0.4, c=0.2),
    typed(cl.NAME, expr="a*(0.3 - length) - b*velocity", a=5.0, b=0.3),
    typed(ps.NAME, expr="peak = maxabs(qvel); tip = qpos[23]; energy = sum(qvel*qvel); lead = qacc[0]"),
    typed(es.NAME, iterations=40, tolerance=1e-3),
])
def test_a_typed_law_for_every_stage_the_chain_offers_passes_its_contract_on_jax_arrays(registry, selection):
    verdict = check_impl(make(), selection, registry)
    assert verdict.ok, verdict.failures


def test_the_jax_emitter_computes_what_the_numpy_one_does_from_the_same_text(registry):
    text = dict(expr="-a*q - b*v + c*sin(time)", a=3.0, b=0.4, c=0.2)
    ours = check_impl(make(), typed(pf.NAME, **text), registry)
    theirs = check_impl(make(ChainHost), typed(pf.NAME, **text), registry)
    assert ours.ok and theirs.ok
    assert abs(ours.metrics["max_law_error"] - theirs.metrics["max_law_error"]) < 1e-5


def test_a_typed_solver_on_jax_arrays_reproduces_the_hosts_banded_solve_over_a_run(registry):
    native, jax_host = make(ChainHost), make()
    for host in (native, jax_host):
        pluck(host)
    with Session(jax_host, [typed(es.NAME, iterations=40, tolerance=1e-3)], registry):
        for _ in range(60):
            jax_host.step()
    for _ in range(60):
        native.step()
    assert np.allclose(np.asarray(jax_host.data.x), native.data.x, atol=2e-5)


def test_a_typed_law_is_compiled_into_the_step_and_a_typed_probe_reads_inside_it(registry):
    host = make()
    pluck(host)
    law = typed(pf.NAME, expr="-a*q - b*v", a=2.0, b=1.5)
    probe = typed(ps.NAME, expr="tip = qpos[6]; speed = norm(qvel)")
    eager = make()
    pluck(eager)
    with Session(eager, [law], registry):
        for _ in range(40):
            eager.step()
    with Session(host, [law, probe], registry) as session:
        host.step()
        pluck(host)
        session.reset_metrics()
        step = jax.jit(host.pure_step)
        state = host.get_state()
        for _ in range(40):
            state = step(state)
        host.set_state(state)
        calls = session.metrics()
        value = np.asarray(host.buffer("probe/2", (host.data.nworld, 2)))
    assert np.allclose(np.asarray(host.data.x), np.asarray(eager.data.x), atol=1e-6)
    assert calls[pf.NAME]["calls"] == 40 and calls[ps.NAME]["calls"] == 40
    assert np.isfinite(value).all() and (value[:, 1] > 0).all()


def test_a_typed_probe_index_past_the_chain_is_refused_at_install_on_jax_arrays(registry):
    from phost import ContractError
    with pytest.raises(ContractError, match=r"qpos\[9\] is out of range"):
        with Session(make(n=8, nu=2), [typed(ps.NAME, expr="far = qpos[9]")], registry):
            pass
