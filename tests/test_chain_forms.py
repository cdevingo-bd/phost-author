"""Typed equations on a host that is not MuJoCo-shaped: the NumPy spring-mass chain.

The same texts, the same compiler and the same backend, on a host whose arrays are called ``x``, ``v`` and ``a`` and whose
effective system is a band: if the typed forms depended on the simulator they were first written for, this would fail.
"""

import numpy as np
import pytest

from phost import ContractError, Registry, Selection, Session
from phost.hosts.chain import ChainHost, make_model
from phost.impls import register_builtin
from phost.stages import control_law as cl
from phost.stages import dof_passive_force as pf
from phost.stages import effective_solve as es
from phost.stages import probe_state as ps
from phost.stages import register_all
from phost.testing.contract import check_impl
from phost_author import expression_impls


@pytest.fixture
def registry():
    reg = Registry()
    register_all(reg)
    register_builtin(reg)
    expression_impls.register(reg)
    return reg


def chain(n=24, nu=3, worlds=3, **kw):
    return ChainHost(make_model(n, nu, **kw), worlds=worlds)


def typed(stage, **params):
    return Selection(stage, f"{stage}/expression", params)


def pluck(host, at=6, size=0.1):
    host.load(np.zeros(host.model.nv))
    host.data.x[:, at] = size


def test_the_host_is_numpy_and_the_typed_forms_compile_for_it():
    assert chain().backend == "numpy"


@pytest.mark.parametrize("selection", [
    typed(pf.NAME, expr="-a*q - b*v + c*sin(time)", a=3.0, b=0.4, c=0.2),
    typed(cl.NAME, expr="a*(0.3 - length) - b*velocity", a=5.0, b=0.3),
    typed(ps.NAME, expr="peak = maxabs(qvel); tip = qpos[23]; energy = sum(qvel*qvel); lead = qacc[0]"),
    typed(es.NAME, iterations=40, tolerance=1e-3),
])
def test_a_typed_law_for_every_stage_the_chain_offers_passes_its_contract(registry, selection):
    verdict = check_impl(chain(), selection, registry)
    assert verdict.ok, verdict.failures


def test_a_typed_probe_reads_the_chains_own_arrays_after_each_forward_pass(registry):
    host = chain()
    pluck(host)
    with Session(host, [typed(ps.NAME, expr="tip = qpos[6]; speed = norm(qvel); when = time")], registry) as session:
        host.step()
        host.step()
        host.forward()                                         # the probe reads what this pass sees
        value = host.buffer("probe/3", (host.data.nworld, 3)).copy()
        calls = session.metrics()[ps.NAME]["calls"]
    d = host.data
    assert np.allclose(value[:, 0], d.x[:, 6], atol=1e-6)
    assert np.allclose(value[:, 1], np.linalg.norm(d.v.astype(np.float64), axis=1), rtol=1e-5)
    assert np.allclose(value[:, 2], d.t, atol=1e-6) and (value[:, 2] > 0).all()
    assert calls == 3


def test_a_probe_index_past_the_chain_is_refused_at_install(registry):
    host = chain(n=8, nu=2)
    with pytest.raises(ContractError, match=r"qpos\[9\] is out of range, qpos has 8 entries"):
        with Session(host, [typed(ps.NAME, expr="far = qpos[9]")], registry):
            pass


def test_a_typed_conjugate_gradient_solve_reproduces_the_hosts_banded_solve_over_a_run(registry):
    native, typed_host = chain(), chain()
    pluck(native)
    pluck(typed_host)
    with Session(typed_host, [typed(es.NAME, iterations=60, tolerance=1e-4)], registry):
        for _ in range(100):
            typed_host.step()
    for _ in range(100):
        native.step()
    assert np.allclose(typed_host.data.x, native.data.x, atol=2e-5), np.abs(typed_host.data.x - native.data.x).max()


def test_a_typed_solve_that_stops_too_early_fails_the_accuracy_it_claimed(registry):
    verdict = check_impl(chain(), typed(es.NAME, init="", step="x = x + a*(rhs - A@x)/diag", iterations=1, tolerance=1e-6), registry)
    assert not verdict.ok


def test_a_typed_law_that_reports_its_slope_integrates_like_the_chains_own_dashpot(registry):
    c = 0.8
    own, native = chain(damping=0.0), chain(damping=c)
    for host in (own, native):
        pluck(host)
    with Session(native, [typed(pf.NAME, expr="-a*q", a=0.3)], registry):
        for _ in range(40):
            native.step()
    with Session(own, [typed(pf.NAME, expr="-a*q - b*v", a=0.3, b=c)], registry):
        for _ in range(40):
            own.step()
    assert np.allclose(own.data.x, native.data.x, rtol=1e-4, atol=1e-6), np.abs(own.data.x - native.data.x).max()


def test_a_typed_controller_drives_only_the_driven_particles(registry):
    host = chain(nu=2)
    host.load(np.zeros(host.model.nv))
    with Session(host, [typed(cl.NAME, expr="a*(1.0 - length) - b*velocity", a=40.0, b=2.0)], registry):
        for _ in range(60):
            host.step()
    x = host.data.x[0]
    assert x[0] > 0.1 and x[1] > 0.05                      # pulled toward 1 through the driven particles
    assert abs(x[-1]) < abs(x[0])                          # the rest of the chain follows, attenuated


def test_the_stages_a_chain_does_not_have_are_not_installable(registry):
    from phost.stages import body_wrench as bw
    with pytest.raises(Exception, match="no stage 'body.wrench'"):
        with Session(chain(), [typed(bw.NAME, expr="fx = -a*vx")], registry):
            pass
