"""The JAX emitter: evaluates the checked trees with ``jax.numpy`` in float32 and replaces the output port's array.

The same trees the NumPy and Warp emitters read, so a typed law runs on a JAX host unchanged. Nothing here uses Python control
flow on array values, so the function is traceable: a host that jits its step (``phost.hosts.jaxchain``) compiles the law with
it. Ports are ``phost.ports.Ref`` cells (a JAX array cannot be written in place); an output is written by assigning to its cell.
Needs ``jax``, imported only when a law is first run.
"""

from __future__ import annotations

from phost import CapabilityError

from .. import expr, program
from ..backends import ChannelsPlan, IterationPlan, PointwisePlan, register_backend


def _jnp():
    import jax.numpy as jnp
    return jnp


def _array(value):
    from phost.ports import unwrap
    return _jnp().asarray(unwrap(value), _jnp().float32)


def _write(port, value) -> None:
    from phost.ports import Ref
    if not isinstance(port, Ref):
        raise CapabilityError("a JAX port must be a phost.ports.Ref: an immutable array cannot be written into")
    port[...] = value


def _constants(constants) -> dict:
    jnp = _jnp()
    return {c: jnp.asarray(float(constants[c]), jnp.float32) for c in expr.CONSTANTS}


class _Jax:
    name = "jax"

    def pointwise(self, plan: PointwisePlan):
        form = plan.form
        first = next(iter(form.outputs))

        def run(system, constants):
            jnp = _jnp()
            values = {}
            for n in form.inputs:
                array = _array(getattr(system, n.port))
                values[n.name] = array[..., n.column] if n.column is not None else array[:, None] if array.ndim == 1 else array
            values.update(_constants(constants))
            width = tuple(getattr(system, first.port).shape[:2])
            mask = _array(getattr(system, form.mask)).astype(bool) if form.mask else None

            def law(tree):
                out = jnp.broadcast_to(expr.evaluate(tree, values, jnp.float32, jnp), width)
                return jnp.where(mask, out, 0.0) if mask is not None else out

            blocks: dict[str, dict[int, object]] = {}
            for n in form.outputs:
                value = law(plan.laws[n.name]) if n.name in plan.laws else jnp.zeros(width, jnp.float32)
                if n.column is None:
                    _write(getattr(system, n.port), value)
                else:
                    blocks.setdefault(n.port, {})[n.column] = value
            for port, columns in blocks.items():
                target = getattr(system, port)
                size = int(target.shape[-1])
                _write(target, jnp.stack([columns.get(c, jnp.zeros(width, jnp.float32)) for c in range(size)], axis=-1))
            if plan.slopes:
                damping = getattr(system, form.slope.port, None)
                if damping is not None:
                    (tree,) = plan.slopes.values()
                    _write(damping, form.slope.sign * law(tree))
        return run

    def channels(self, plan: ChannelsPlan):
        form = plan.form

        def run(system, constants):
            jnp = _jnp()
            values = {}
            for n in form.inputs:
                array = _array(getattr(system, n.port))
                values[n.name] = array[:, None] if array.ndim == 1 else array
            values.update(_constants(constants))
            out = getattr(system, form.output)
            worlds = int(out.shape[0])
            columns = [jnp.broadcast_to(expr.evaluate(law, values, jnp.float32, jnp), (worlds, 1))[:, 0] for _, law in plan.channels]
            _write(out, jnp.stack(columns, axis=1))
        return run

    def iteration(self, plan: IterationPlan):
        form = plan.form

        def run(system, constants):
            jnp = _jnp()
            out = getattr(system, form.solution)
            n = int(out.shape[1])
            h, rhs = _array(getattr(system, form.matrix)), _array(getattr(system, form.rhs))[:, :n]
            live = _array(getattr(system, form.active)) != 0 if form.active else jnp.ones(h.shape[0], bool)
            a = program.symmetric(h, n, jnp, jnp.float32)
            a = jnp.where(live[:, None, None], a, jnp.eye(n, dtype=jnp.float32))      # an inactive world's matrix may hold anything
            b = jnp.where(live[:, None], rhs, 0.0)
            x = program.run(plan.program, a, b, constants, plan.iterations, xp=jnp, dtype=jnp.float32)
            _write(out, jnp.where(live[:, None], form.sign * x, _array(out)))
            for port, quantity in form.derived:
                value = jnp.sum(x * x, axis=1) if quantity == "square_norm" else jnp.sum(b * x, axis=1)
                _write(getattr(system, port), jnp.where(live, value, _array(getattr(system, port))))
        return run


register_backend(_Jax())
