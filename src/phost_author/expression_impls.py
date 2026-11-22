"""Implementations of the pointwise stages whose law is a typed equation.

``dof.passive_force/expression`` and ``control.law/expression`` take the equation as the ``expr`` parameter and up to
eight free constants ``a`` .. ``h``. The float64 reference evaluates the same checked tree, so the stage's contract
check judges the generated kernel against exactly what was typed.

    dof.passive_force: names q, v, time        -> force on each hinge or slide DOF
    control.law:       names length, velocity, time -> ctrl of each actuator

A passive force law also reports ``damping``, its derivative with respect to ``v`` taken symbolically from the typed
tree, so that a host whose integrator treats joint damping implicitly does the same for the typed law (parameter
``implicit``, on by default).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from phost import ContractError, Impl, ImplKind, ParamSpec, Registry
from phost.impls import controllers, forces
from phost.stages import control_law as cl
from phost.stages import dof_passive_force as pf

from . import expr
from .modules import load_source

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
    derivative_of: str | None = None  # variable whose derivative of the law is reported as ``damping`` (negated)


TARGETS = {
    pf.NAME: Target(
        pf.NAME, ("q", "v", "time"), "force",
        "q: wp.array2d[float], v: wp.array2d[float], time: wp.array[float], scalar: wp.array[int]",
        {"q": "q[w, i]", "v": "v[w, i]", "time": "time[w]"}, True, forces._applicable, forces._advisory,
        derivative_of="v"),
    cl.NAME: Target(
        cl.NAME, ("length", "velocity", "time"), "ctrl",
        "length: wp.array2d[float], velocity: wp.array2d[float], time: wp.array[float]",
        {"length": "length[w, i]", "velocity": "velocity[w, i]", "time": "time[w]"}, False,
        controllers._applicable, controllers._advisory),
}


def kernel_source(target: Target, expression: expr.Expression, with_damping: bool = False) -> str:
    """The Warp module for one equation. Constants are arguments, so changing them does not recompile.

    With ``with_damping`` the kernel also writes ``damping[w, i] = -d law / d v``, derived from the same tree.
    """
    emitted = dict(target.emitted, **{c: c for c in expr.CONSTANTS})
    constants = ", ".join(f"{c}: float" for c in expr.CONSTANTS)
    outputs = "out: wp.array2d[float]" + (", damping: wp.array2d[float]" if with_damping else "")
    body = expr.to_warp(expression, emitted)
    slope = expr.to_warp(expr.differentiate(expression, target.derivative_of), emitted) if with_damping else ""
    lines = [KERNEL_HEAD + "@wp.kernel", f"def kernel({target.kernel_signature}, {constants}, {outputs}):",
             "    w, i = wp.tid()"]
    if target.masked:
        lines += ["    if scalar[i] == 1:", f"        out[w, i] = {body}"]
        lines += [f"        damping[w, i] = -({slope})"] if with_damping else []
        lines += ["    else:", "        out[w, i] = 0.0"]
        lines += ["        damping[w, i] = 0.0"] if with_damping else []
    else:
        lines += [f"    out[w, i] = {body}"]
    return "\n".join(lines) + "\n"


def _parse(target: Target, params: Mapping[str, Any]) -> expr.Expression:
    return expr.parse(params["expr"], target.variables)


def _values(target: Target, source, constants: Mapping[str, Any]) -> dict:
    """Float64 values of the names an expression of this target uses, from a system or a dict of inputs."""
    get = source.get if isinstance(source, Mapping) else lambda name: getattr(source, name)
    values = {name: np.asarray(get(name), np.float64) for name in target.variables if name != "time"}
    values["time"] = np.asarray(get("time"), np.float64)[:, None]
    values.update({c: constants[c] for c in expr.CONSTANTS})
    return values


def _evaluate(target: Target, expression: expr.Expression, source, constants, shape, mask) -> np.ndarray:
    law = np.broadcast_to(expr.evaluate(expression, _values(target, source, constants)), shape).astype(np.float64)
    return np.where(np.asarray(mask).astype(bool), law, 0.0) if target.masked else law


def _reference(target: Target):
    def reference(inputs: Mapping[str, Any], params: Mapping[str, Any]) -> dict:
        expression = _parse(target, params)
        shape = np.asarray(inputs["time"]).shape[0], np.asarray(inputs[target.variables[0]]).shape[1]
        mask = inputs.get("scalar")
        out = {target.output: _evaluate(target, expression, inputs, params, shape, mask)}
        if target.derivative_of:
            slope = expr.differentiate(expression, target.derivative_of)
            out["damping"] = (-_evaluate(target, slope, inputs, params, shape, mask)) if params["implicit"] else np.zeros(shape)
        return out
    return reference


class _Runtime:
    def __init__(self, target: Target, host, params):
        self.target, self.params = target, dict(params)
        self.expression = _parse(target, params)
        self.slope = expr.differentiate(self.expression, target.derivative_of) \
            if target.derivative_of and params["implicit"] else None
        self.kernel = self._scratch = None
        if host.backend == "warp":
            source = kernel_source(target, self.expression, with_damping=self.slope is not None)
            self.kernel = load_source(source, f"phost_expr_{target.stage.replace('.', '_')}").kernel

    def apply(self, system):
        t = self.target
        out = getattr(system, t.output)
        damping = getattr(system, "damping", None) if self.slope is not None else None
        if self.kernel is None:                                   # numpy backend
            mask = getattr(system, "scalar", None)
            out[:] = _evaluate(t, self.expression, system, self.params, out.shape, mask).astype(out.dtype)
            if damping is not None:
                damping[:] = (-_evaluate(t, self.slope, system, self.params, out.shape, mask)).astype(damping.dtype)
        else:
            import warp as wp
            outputs = [out]
            if self.slope is not None:
                if damping is None:                               # the host does not take it; the kernel still writes it
                    if self._scratch is None:
                        self._scratch = wp.zeros(out.shape, dtype=float, device=out.device)
                    damping = self._scratch
                outputs.append(damping)
            wp.launch(self.kernel, dim=out.shape, inputs=self._arguments(system), outputs=outputs, device=out.device)

    def _arguments(self, system):
        t = self.target
        head = [system.q, system.v, system.time, system.scalar] if t.masked else \
            [system.length, system.velocity, system.time]
        return head + [self.params[c] for c in expr.CONSTANTS]

    def metrics(self):
        return {}


def _impl(target: Target) -> Impl:
    def check(params):
        _parse(target, params)

    implicit = (ParamSpec("implicit", "bool", True, "report the law's own damping so the host integrates it implicitly "
                          "where it can; False applies the law explicitly"),) if target.derivative_of else ()
    return Impl(
        id=f"{target.stage}/expression", stage=target.stage, stage_version=1, version="2",
        title="Typed equation", kind=ImplKind.EXPRESSION,
        doc=(f"The law is the equation in `expr`, over {', '.join(target.variables)} and the constants a..h "
             f"(set as parameters, so changing them does not recompile). Functions: {', '.join(sorted(expr.FUNCTIONS))}."),
        params=(ParamSpec("expr", "str", "0.0", "the equation"),
                *[ParamSpec(c, "float", 0.0, f"constant {c}") for c in expr.CONSTANTS], *implicit),
        fit="Pointwise laws. Not for laws that couple DOFs or keep state.",
        extensions_optional=(forces.IMPLICIT,) if target.derivative_of else (),
        applicable=target.applicable, advisory=target.advisory, check_params=check,
        factory=lambda host, params, t=target: _Runtime(t, host, params), reference=_reference(target))


def preview(stage: str, text: str, constants: Mapping[str, float] | None = None, sweep: str | None = None,
            at: Mapping[str, float] | None = None, lo: float = -1.0, hi: float = 1.0, n: int = 101) -> dict:
    """What an equation does, without a simulation: check it and sample it along one variable.

    ``sweep`` (default: the stage's first variable) runs from ``lo`` to ``hi`` over ``n`` points while the other
    variables stay at the values in ``at`` (default 0). Returns the variables, the sample points, the law's values,
    and for a force law the symbolic derivative with respect to ``v`` as text and its sampled damping. An
    unacceptable equation raises ``ExpressionError`` with the message a person should see.
    """
    if stage not in TARGETS:
        raise ContractError(f"no typed-equation stage {stage!r}; known: {sorted(TARGETS)}")
    target = TARGETS[stage]
    sweep = sweep or target.variables[0]
    if sweep not in target.variables:
        raise ContractError(f"cannot sweep {sweep!r}; the variables are {list(target.variables)}")
    if not (2 <= n <= 2001) or not np.isfinite([lo, hi]).all() or not lo < hi:
        raise ContractError("a preview needs lo < hi (finite) and between 2 and 2001 points")
    values = {name: (at or {}).get(name, 0.0) for name in target.variables}
    given = {c: float((constants or {}).get(c, 0.0)) for c in expr.CONSTANTS}
    expression = expr.parse(text, target.variables)
    xs = np.linspace(lo, hi, n)
    env = {**values, **given, sweep: xs}
    out: dict[str, Any] = dict(variables=list(target.variables), sweep=sweep, x=xs.tolist(), key=expression.key,
                               y=np.broadcast_to(expr.evaluate(expression, env), xs.shape).tolist())
    if target.derivative_of:
        slope = expr.differentiate(expression, target.derivative_of)
        out["derivative"] = dict(of=target.derivative_of, text=ast.unparse(slope.tree.body),
                                 damping=np.broadcast_to(-expr.evaluate(slope, env), xs.shape).tolist())
    out["finite"] = bool(np.isfinite(out["y"]).all())
    return out


def register(registry: Registry) -> None:
    for target in TARGETS.values():
        registry.add_impl(_impl(target))
