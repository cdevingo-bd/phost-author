"""The NumPy emitter: evaluates the checked trees directly, in float64, and casts to the port's dtype on write.

It is also the reference semantics: every other backend is judged against what this one computes from the same trees.
"""

from __future__ import annotations

import numpy as np

from .. import expr, program
from . import ChannelsPlan, IterationPlan, PointwisePlan, register_backend


def _array(value):
    """A float64 numpy view of a port's array (a device array is copied to the host)."""
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value, np.float64)


def _write(port, value) -> None:
    """Assign into a port in place, whatever array type it is, casting to its dtype."""
    with np.errstate(all="ignore"):                          # a diverging law overflows float32: the check reports it
        if hasattr(port, "assign"):
            port.assign(np.asarray(value).astype(port.numpy().dtype))
        else:
            port[...] = np.asarray(value).astype(port.dtype)


def pointwise_values(plan: PointwisePlan, system, constants) -> dict:
    """The float64 value of every input name of the form, shaped (world, element) or (world, 1), with the constants."""
    values = {}
    for n in plan.form.inputs:
        array = _array(getattr(system, n.port))
        values[n.name] = array[..., n.column] if n.column is not None else array[:, None] if array.ndim == 1 else array
    values.update({c: np.float64(constants[c]) for c in expr.CONSTANTS})
    return values


class _Numpy:
    name = "numpy"

    def pointwise(self, plan: PointwisePlan):
        form = plan.form
        first = next(iter(form.outputs))

        def run(system, constants):
            values = pointwise_values(plan, system, constants)
            width = _array(getattr(system, first.port)).shape[:2]
            mask = _array(getattr(system, form.mask)).astype(bool) if form.mask else None

            def law(tree):
                out = np.broadcast_to(expr.evaluate(tree, values), width).astype(np.float64)
                return np.where(mask, out, 0.0) if mask is not None else out

            ports: dict[str, np.ndarray] = {}
            for n in form.outputs:
                port = getattr(system, n.port)
                if n.column is None:
                    _write(port, law(plan.laws[n.name]) if n.name in plan.laws else np.zeros(width))
                else:
                    current = ports.get(n.port)
                    if current is None:
                        current = ports[n.port] = np.zeros(_array(port).shape)
                    current[..., n.column] = law(plan.laws[n.name]) if n.name in plan.laws else 0.0
            for name, block in ports.items():
                _write(getattr(system, name), block)
            if plan.slopes:
                damping = getattr(system, form.slope.port, None)
                if damping is not None:
                    (tree,) = plan.slopes.values()
                    _write(damping, form.slope.sign * law(tree))
        return run

    def channels(self, plan: ChannelsPlan):
        form, contract = plan.form, plan.contract

        def run(system, constants):
            values = {}
            for n in form.inputs:
                array = _array(getattr(system, n.port))
                values[n.name] = array[:, None] if array.ndim == 1 else array
            values.update({c: np.float64(constants[c]) for c in expr.CONSTANTS})
            out = getattr(system, form.output)
            block = np.zeros(_array(out).shape)
            for column, (_, law) in enumerate(plan.channels):
                block[:, column] = np.broadcast_to(expr.evaluate(law, values), (block.shape[0], 1))[:, 0]
            _write(out, block)
        return run

    def iteration(self, plan: IterationPlan):
        form = plan.form

        def run(system, constants):
            out = getattr(system, form.solution)
            n = out.shape[1]
            h, rhs = _array(getattr(system, form.matrix)), _array(getattr(system, form.rhs))[:, :n]
            live = _array(getattr(system, form.active)) != 0 if form.active else np.ones(h.shape[0], bool)
            a = program.symmetric(h, n)
            a[~live] = np.eye(n)                              # an inactive world's matrix may hold anything
            b = np.where(live[:, None], rhs, 0.0)
            x = program.run(plan.program, a, b, constants, plan.iterations)
            _write_where(out, form.sign * x, live)
            for port, quantity in form.derived:
                value = np.sum(x * x, axis=1) if quantity == "square_norm" else np.sum(b * x, axis=1)
                _write_where(getattr(system, port), value, live)
        return run


def _write_where(port, value, live) -> None:
    current = _array(port)
    current[live] = np.asarray(value)[live]
    _write(port, current)


register_backend(_Numpy())
