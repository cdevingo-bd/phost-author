"""Implementations of the pointwise stages whose law is a typed equation.

``dof.passive_force/expression`` and ``control.law/expression`` take the equation as the ``expr`` parameter and up to
eight free constants ``a`` .. ``h``. The float64 reference evaluates the same checked tree, so the stage's contract
check judges the generated kernel against exactly what was typed.

    dof.passive_force: names q, v, time        -> force on each hinge or slide DOF
    control.law:       names length, velocity, time -> ctrl of each actuator
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from phost import ContractError, Impl, ImplKind, ParamSpec, Registry
from phost.impls import controllers, forces
from phost.stages import control_law as cl
from phost.stages import dof_passive_force as pf

from . import expr

KERNEL_HEAD = "import warp as wp\n\n\n"


@dataclass(frozen=True)
class Target:
    stage: str
    variables: tuple[str, ...]
    output: str                       # port the equation writes
    kernel_signature: str             # kernel arguments before the constants and the output
    emitted: Mapping[str, str]        # name -> kernel expression for that name
    masked: bool                      # write 0 where scalar == 0
    applicable: Any
    advisory: Any


TARGETS = {
    pf.NAME: Target(
        pf.NAME, ("q", "v", "time"), "force",
        "q: wp.array2d[float], v: wp.array2d[float], time: wp.array[float], scalar: wp.array[int]",
        {"q": "q[w, i]", "v": "v[w, i]", "time": "time[w]"}, True, forces._applicable, forces._advisory),
    cl.NAME: Target(
        cl.NAME, ("length", "velocity", "time"), "ctrl",
        "length: wp.array2d[float], velocity: wp.array2d[float], time: wp.array[float]",
        {"length": "length[w, i]", "velocity": "velocity[w, i]", "time": "time[w]"}, False,
        controllers._applicable, controllers._advisory),
}


def kernel_source(target: Target, expression: expr.Expression) -> str:
    """The Warp module for one equation. Constants are arguments, so changing them does not recompile."""
    emitted = dict(target.emitted, **{c: c for c in expr.CONSTANTS})
    constants = ", ".join(f"{c}: float" for c in expr.CONSTANTS)
    body = expr.to_warp(expression, emitted)
    lines = [KERNEL_HEAD + "@wp.kernel",
             f"def kernel({target.kernel_signature}, {constants}, out: wp.array2d[float]):",
             "    w, i = wp.tid()"]
    if target.masked:
        lines += ["    if scalar[i] == 1:", f"        out[w, i] = {body}", "    else:", "        out[w, i] = 0.0"]
    else:
        lines += [f"    out[w, i] = {body}"]
    return "\n".join(lines) + "\n"


def _parse(target: Target, params: Mapping[str, Any]) -> expr.Expression:
    return expr.parse(params["expr"], target.variables)


def _reference(target: Target):
    def reference(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict:
        expression = _parse(target, params)
        shape = np.asarray(inputs["time"]).shape[0], np.asarray(inputs[target.variables[0]]).shape[1]
        values = {name: np.asarray(inputs[name], np.float64) for name in target.variables if name != "time"}
        values["time"] = np.asarray(inputs["time"], np.float64)[:, None]
        values.update({c: params[c] for c in expr.CONSTANTS})
        out = np.broadcast_to(expr.evaluate(expression, values), shape).astype(np.float64)
        if target.masked:
            out = np.where(np.asarray(inputs["scalar"]).astype(bool), out, 0.0)
        return {target.output: out}
    return reference


class _Runtime:
    def __init__(self, target: Target, host, params):
        self.target, self.params, self.traces = target, dict(params), 0
        self.expression = _parse(target, params)
        self.kernel = None
        if host.backend == "warp":
            name = f"phost_expr_{target.stage.replace('.', '_')}_{self.expression.key}"
            self.kernel = expr.build_module(name, kernel_source(target, self.expression)).kernel

    def apply(self, system):
        t = self.target
        out = getattr(system, t.output)
        if self.kernel is None:                                   # numpy backend
            values = {name: np.asarray(getattr(system, name), np.float64) for name in t.variables if name != "time"}
            values["time"] = np.asarray(system.time, np.float64)[:, None]
            values.update({c: self.params[c] for c in expr.CONSTANTS})
            law = np.broadcast_to(expr.evaluate(self.expression, values), out.shape)
            if t.masked:
                law = np.where(np.asarray(system.scalar).astype(bool), law, 0.0)
            out[:] = law.astype(out.dtype)
        else:
            import warp as wp
            wp.launch(self.kernel, dim=out.shape, inputs=self._arguments(system), outputs=[out], device=out.device)
        self.traces += 1

    def _arguments(self, system):
        t = self.target
        head = [system.q, system.v, system.time, system.scalar] if t.masked else \
            [system.length, system.velocity, system.time]
        return head + [self.params[c] for c in expr.CONSTANTS]

    def metrics(self):
        return dict(traces=self.traces)


def _impl(target: Target) -> Impl:
    def check(params):
        _parse(target, params)

    return Impl(
        id=f"{target.stage}/expression", stage=target.stage, stage_version=1, version="1",
        title="Typed equation", kind=ImplKind.EXPRESSION,
        doc=(f"The law is the equation in `expr`, over {', '.join(target.variables)} and the constants a..h "
             f"(set as parameters, so changing them does not recompile). Functions: {', '.join(sorted(expr.FUNCTIONS))}."),
        params=(ParamSpec("expr", "str", "0.0", "the equation"),
                *[ParamSpec(c, "float", 0.0, f"constant {c}") for c in expr.CONSTANTS]),
        fit="Pointwise laws. Not for laws that couple DOFs or keep state.",
        applicable=target.applicable, advisory=target.advisory, check_params=check,
        factory=lambda host, params, t=target: _Runtime(t, host, params), reference=_reference(target))


def register(registry: Registry) -> None:
    for target in TARGETS.values():
        registry.add_impl(_impl(target))
