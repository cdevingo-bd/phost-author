"""Implementations of every stage whose law can be typed, built from the form the stage declares (``phost.equation``).

There is no table of stages here. ``register(registry)`` adds a ``<stage>/expression`` implementation for each stage in the
registry that declares an equation form; a stage added to phost with a form is typeable with no change to this package.

    Pointwise   dof.passive_force, control.law, body.wrench   one law per element; ``expr`` is one expression, or one
                                                              ``name = expression`` line per output the stage writes
    Channels    probe.state                                   one line per measured column, over arrays and reductions
    Iteration   effective.solve, newton.linear_solve          ``init`` and ``step`` lines, repeated ``iterations`` times

The float64 reference of an implementation evaluates the same checked trees as its code does, so the stage's contract check
judges the generated kernel against exactly what was typed. A typed solver is approximate: it declares itself so (an
``Impl.swap_class``), claims a ``tolerance`` on the relative residual, and is held to the claim; the check also reports
``max_program_error``, the distance between what the generated code computed and what the typed program means in float64.

How a tree becomes code is the host's array backend's business (``phost_author.backends``): NumPy and Warp ship, and a
backend registered under another name (JAX, ...) serves hosts whose ports are that kind of array.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from phost import (Channels, ContractError, Impl, ImplKind, Iteration, MetricSpec, ParamSpec, Pointwise, Registry, StageContract,
                   SwapClass, Tolerance)

from . import expr, laws, program
from .backends import ChannelsPlan, IterationPlan, PointwisePlan, backend_for

CONSTANT_PARAMS = tuple(ParamSpec(c, "float", 0.0, f"constant {c}") for c in expr.CONSTANTS)
PROGRAM_ERROR = MetricSpec("max_program_error", "1", "lower", "gauge",
                           "max |output - the typed program run in float64| / max |that|: whether the generated code computes "
                           "what was typed, apart from whether what was typed is any good")
PROGRAM_ERROR_GATE = 1e-3
HOST_SOLVE_GATE = 1.0


# ---- plans: the text, checked --------------------------------------------------------------------------

def pointwise_plan(contract: StageContract, form: Pointwise, params: Mapping[str, Any]) -> PointwisePlan:
    texts = laws.parse_pointwise(params["expr"], form)
    slopes = {}
    if form.slope is not None and params.get(form.slope.param, True):
        (output,) = form.outputs
        if output.name in texts:
            slopes[output.name] = expr.differentiate(texts[output.name], form.slope.of)
    return PointwisePlan(contract, form, texts, slopes)


def channels_plan(contract: StageContract, form: Channels, params: Mapping[str, Any]) -> ChannelsPlan:
    return ChannelsPlan(contract, form, tuple(laws.parse_channels(params["expr"], form, contract)))


def iteration_plan(contract: StageContract, form: Iteration, params: Mapping[str, Any]) -> IterationPlan:
    return IterationPlan(contract, form, program.compile_program(params["init"], params["step"]), int(params["iterations"]))


def make_plan(contract: StageContract, form, params: Mapping[str, Any]):
    return {Pointwise: pointwise_plan, Channels: channels_plan, Iteration: iteration_plan}[type(form)](contract, form, params)


def _constants(params: Mapping[str, Any]) -> dict:
    return {c: float(params[c]) for c in expr.CONSTANTS}


# ---- the runtime ---------------------------------------------------------------------------------------

class _Runtime:
    """What a host calls once per execution of the stage; its entry method (``apply``, ``solve``, ``observe``) runs the backend."""

    def __init__(self, contract: StageContract, form, params: Mapping[str, Any], host):
        self.contract, self.form, self.params = contract, form, dict(params)
        self.plan = make_plan(contract, form, params)
        self._run = getattr(backend_for(host.backend), form.kind)(self.plan)
        self._constants = _constants(params)
        self._checked = False
        if isinstance(form, Channels):
            self.width = len(self.plan.channels)
            self._check_against_the_model(host)

    def _check_against_the_model(self, host) -> None:
        """An index past the end of an array is refused when the probe is installed, from what the host says about its model."""
        facts = host.facts()
        sizes = {"nv": facts.nv, **{k: v for k, v in facts.extra.items() if isinstance(v, int) and not isinstance(v, bool)}}
        laws.check_indices(self.plan.channels, sizes, self.contract, self.form)

    def _execute(self, system) -> None:
        if not self._checked:
            self._check_sizes(system)
            self._checked = True
        self._run(system, self._constants)

    def _check_sizes(self, system) -> None:
        """Indices in a channel must be inside the arrays of this model: known only once the stage first runs."""
        if not isinstance(self.form, Channels):
            return
        sizes: dict[str, int] = {}
        for n in self.form.inputs:
            shape = tuple(getattr(system, n.port).shape)
            if len(shape) == 2:
                axis = str(self.contract.port(n.port).shape[1]).rstrip("+")
                if sizes.setdefault(axis, shape[1]) != shape[1]:
                    raise ContractError(f"{n.name} has {shape[1]} entries but another array along {axis!r} has {sizes[axis]}")
        laws.check_indices(self.plan.channels, sizes, self.contract, self.form)

    def metrics(self) -> dict:
        return {}


def _runtime_class(contract: StageContract):
    return type("TypedLaw", (_Runtime,), {contract.entry: _Runtime._execute})


# ---- the float64 reference -----------------------------------------------------------------------------

def _dims(contract: StageContract, inputs: Mapping[str, Any], extra: Mapping[str, int]) -> dict:
    """Bind the contract's dimensions from the input arrays (padded dimensions bind to the unpadded size ``extra`` gives)."""
    bound = dict(extra)
    for port in contract.ports:
        if port.name in inputs:
            for dim, size in zip(port.shape, np.shape(inputs[port.name])):
                if isinstance(dim, str) and not dim.endswith("+"):
                    bound.setdefault(dim, int(size))
    return bound


def _system(contract: StageContract, inputs: Mapping[str, Any], dims: Mapping[str, int], fill: float = 0.0):
    system = SimpleNamespace(**{k: np.asarray(v) for k, v in inputs.items()})
    for port in contract.ports:
        if port.role.value != "in":
            shape = tuple(dims[d.rstrip("+")] if isinstance(d, str) else d for d in port.shape)
            setattr(system, port.name, np.full(shape, fill, np.float64))
    return system


def _reference(contract: StageContract, form):
    def reference(inputs: Mapping[str, Any], params: Mapping[str, Any], dims: Mapping[str, int] | None = None) -> dict:
        plan = make_plan(contract, form, params)
        extra = dict(dims or {})
        if isinstance(form, Channels):
            extra["width"] = len(plan.channels)
        bound = _dims(contract, inputs, extra)
        system = _system(contract, inputs, bound, fill=np.nan if isinstance(form, Iteration) else 0.0)
        getattr(backend_for("numpy"), form.kind)(plan)(system, _constants(params))
        return {port.name: getattr(system, port.name) for port in contract.ports if port.role.value != "in"}
    return reference


# ---- the implementation --------------------------------------------------------------------------------

def _params(contract: StageContract, form) -> tuple[ParamSpec, ...]:
    if isinstance(form, Pointwise):
        names = ", ".join(n.name for n in form.outputs)
        text = ("the equation" if len(form.outputs) == 1 else f"one line per output: {names}; one not written is 0")
        implicit = (ParamSpec(form.slope.param, "bool", True,
                              "report the law's own damping so the host integrates it implicitly where it can; False applies "
                              "the law explicitly"),) if form.slope else ()
        return (ParamSpec("expr", "str", "0.0" if len(form.outputs) == 1 else "", text), *CONSTANT_PARAMS, *implicit)
    if isinstance(form, Channels):
        return (ParamSpec("expr", "str", "peak = maxabs(qvel)", "one line per measured column: label = expression"),
                *CONSTANT_PARAMS)
    return (ParamSpec("init", "str", program.DEFAULT_INIT, "lines run once before the passes; x is the running solution"),
            ParamSpec("step", "str", program.DEFAULT_STEP, "lines run on every pass, in order"),
            ParamSpec("iterations", "int", 20, "passes of step", min=1, max=program.MAX_ITERATIONS),
            ParamSpec("tolerance", "float", 1e-2, "the relative residual this program claims to reach; the check holds it to "
                      "that", min=1e-7, max=1.0),
            *(ParamSpec(c, "float", 1.0 if c == "a" else 0.0, f"constant {c}") for c in expr.CONSTANTS))


def _doc(contract: StageContract, form) -> str:
    consts = "the constants a..h (set as parameters, so changing them does not recompile)"
    functions = f"Functions: {', '.join(sorted(expr.FUNCTIONS))}."
    if isinstance(form, Pointwise):
        names = ", ".join(n.name for n in form.inputs)
        return (f"The law is the equation in `expr`, over {names} and {consts}. {functions} "
                f"It writes {', '.join(n.name for n in form.outputs)}.")
    if isinstance(form, Channels):
        names = ", ".join(n.name for n in form.inputs)
        return (f"One line per measured column, `label = expression`, over {names} and {consts}. The arrays are summed, "
                f"normed or picked from with {', '.join(sorted(expr.REDUCTIONS))} and name[0]. {functions}")
    return ("A linear solve written as a program: `init` once, then `step` for `iterations` passes. It sees rhs (right-hand side), "
            "diag (diagonal of A), `A @ name`, n, k (the pass), " + consts + ". The answer is x. Approximate: it claims a "
            "relative residual `tolerance` and is held to it.")


_FIT = {Pointwise: "Pointwise laws. Not for laws that couple elements or keep state.",
        Channels: "Measurements of the whole state per world. Read-only: the simulation does not change.",
        Iteration: "Trying a solver idea quickly. A generated solver is for exploring, not for speed: the opportunity report "
                   "says what the stage is worth."}


def _iteration_gates(contract: StageContract):
    measured = {m.name for m in contract.metrics}

    def gates(params: Mapping[str, Any]) -> tuple[Tolerance, ...]:
        out = [Tolerance("max_relative_residual", float(params["tolerance"]), "the accuracy this program claims"),
               Tolerance("max_program_error", PROGRAM_ERROR_GATE, "float32 code against the typed program in float64")]
        if "max_native_difference" in measured:
            out.append(Tolerance("max_native_difference", HOST_SOLVE_GATE,
                                 "an approximate solve must be near the host's own answer; its accuracy is ranked on the "
                                 "trajectory, not gated here"))
        return tuple(out)
    return gates


def build_impl(contract: StageContract, form) -> Impl:
    def check(params):
        make_plan(contract, form, params)

    iteration = isinstance(form, Iteration)
    runtime = _runtime_class(contract)
    return Impl(
        id=f"{contract.name}/expression", stage=contract.name, stage_version=contract.version, version="3",
        title="Typed solver step" if iteration else "Typed equation", kind=ImplKind.EXPRESSION,
        doc=_doc(contract, form), params=_params(contract, form), fit=_FIT[type(form)],
        metrics=(PROGRAM_ERROR,) if iteration else (),
        extensions_optional=("implicit_damping",) if getattr(form, "slope", None) else (),
        applicable=form.applicable, advisory=form.advisory, check_params=check,
        factory=lambda host, params: runtime(contract, form, params, host), reference=_reference(contract, form),
        swap_class=SwapClass.APPROXIMATE if iteration else None,
        tolerances=_iteration_gates(contract) if iteration else None)


def register(registry: Registry) -> None:
    """Add a typed-equation implementation for every stage in ``registry`` that declares an equation form."""
    for stage in registry.equation_stages():
        registry.add_impl(build_impl(registry.contract(stage), registry.equation(stage)))


# ---- preview: what a typed law does, without a simulation ----------------------------------------------

def _form_of(registry: Registry, stage: str):
    if stage not in registry.equation_stages():
        raise ContractError(f"no typed-equation stage {stage!r}; known: {sorted(registry.equation_stages())}")
    return registry.contract(stage), registry.equation(stage)


def describe_form(registry: Registry, stage: str) -> dict:
    """What a person typing for ``stage`` needs to know: the form's kind, its names and what it writes."""
    contract, form = _form_of(registry, stage)
    out: dict[str, Any] = dict(stage=stage, kind=form.kind, constants=list(expr.CONSTANTS))
    if isinstance(form, Pointwise):
        out.update(inputs=[n.name for n in form.inputs], outputs=[n.name for n in form.outputs],
                   slope=form.slope.of if form.slope else None)
    elif isinstance(form, Channels):
        scope = laws.channels_scope(form, contract)
        out.update(inputs=[n.name for n in form.inputs], arrays={k: v for k, v in scope.vectors.items()},
                   functions=sorted(expr.FUNCTIONS), reductions=sorted(expr.REDUCTIONS))
    else:
        out.update(inputs=["rhs", "diag", "A", "n", "k"], outputs=["x"], functions=sorted(expr.FUNCTIONS),
                   reductions=sorted(expr.REDUCTIONS), default_init=program.DEFAULT_INIT, default_step=program.DEFAULT_STEP)
    return out


def preview(registry: Registry, stage: str, text: str | None = None, constants: Mapping[str, float] | None = None,
            sweep: str | None = None, at: Mapping[str, float] | None = None, lo: float = -1.0, hi: float = 1.0, n: int = 101,
            *, init: str | None = None, step: str | None = None, iterations: int = 20, size: int = 12,
            condition: float = 100.0, seed: int = 0) -> dict:
    """What a typed law does, without a simulation: check it and sample it.

    Pointwise: ``sweep`` (default: the stage's first input) runs from ``lo`` to ``hi`` over ``n`` points while the other
    inputs stay at ``at`` (default 0); the result has each output's values, and for a force law the symbolic derivative.
    Channels: each channel's value on a sample state. Iteration: the program run on a random symmetric positive definite system
    of ``size`` unknowns with the given ``condition`` number, and its relative residual after every pass. An unacceptable
    text raises ``ExpressionError`` with the message a person should see.
    """
    contract, form = _form_of(registry, stage)
    if isinstance(form, Pointwise):
        return _preview_pointwise(contract, form, text or "", constants, sweep, at, lo, hi, n)
    if isinstance(form, Channels):
        return _preview_channels(contract, form, text or "", constants, size, seed)
    return _preview_iteration(contract, form, init if init is not None else program.DEFAULT_INIT,
                              step if step is not None else program.DEFAULT_STEP, constants, iterations, size, condition, seed)


def _preview_pointwise(contract, form, text, constants, sweep, at, lo, hi, n) -> dict:
    names = [i.name for i in form.inputs]
    sweep = sweep or names[0]
    if sweep not in names:
        raise ContractError(f"cannot sweep {sweep!r}; the variables are {names}")
    if not (2 <= n <= 2001) or not np.isfinite([lo, hi]).all() or not lo < hi:
        raise ContractError("a preview needs lo < hi (finite) and between 2 and 2001 points")
    texts = laws.parse_pointwise(text, form)
    values = {name: (at or {}).get(name, 0.0) for name in names}
    given = {c: float((constants or {}).get(c, 0.0)) for c in expr.CONSTANTS}
    xs = np.linspace(lo, hi, n)
    env = {**values, **given, sweep: xs}
    outputs = {name: np.broadcast_to(expr.evaluate(texts[name], env) if name in texts else 0.0, xs.shape).tolist()
               for name in (o.name for o in form.outputs)}
    first = form.outputs[0].name
    out: dict[str, Any] = dict(kind="pointwise", variables=names, outputs=outputs, sweep=sweep, x=xs.tolist(),
                               key="/".join(texts[k].key for k in texts), y=outputs[first])
    if form.slope and len(form.outputs) == 1 and first in texts:
        slope = expr.differentiate(texts[first], form.slope.of)
        out["derivative"] = dict(of=form.slope.of, text=ast.unparse(slope.tree.body),
                                 damping=np.broadcast_to(form.slope.sign * expr.evaluate(slope, env), xs.shape).tolist())
    out["finite"] = bool(np.isfinite([v for ys in outputs.values() for v in ys]).all())
    return out


def _preview_channels(contract, form, text, constants, size, seed) -> dict:
    channels = laws.parse_channels(text, form, contract)
    scope = laws.channels_scope(form, contract)
    top = max([node.slice.value for _, law in channels for node in ast.walk(law.tree) if isinstance(node, ast.Subscript)] + [0])
    count = max(int(size), top + 1)
    rng = np.random.default_rng(seed)
    values: dict[str, Any] = {name: rng.standard_normal((1, count)) for name in scope.vectors}
    values.update({name: np.array([[1.0]]) for name in scope.scalars})
    values.update({c: float((constants or {}).get(c, 0.0)) for c in expr.CONSTANTS})
    results = [dict(label=label, text=law.text, value=float(np.broadcast_to(expr.evaluate(law, values), (1, 1))[0, 0]))
               for label, law in channels]
    return dict(kind="channels", sample=dict(entries=count, seed=seed, time=1.0), channels=results,
                finite=bool(np.isfinite([r["value"] for r in results]).all()),
                key="/".join(law.key for _, law in channels))


def _spd(n: int, condition: float, rng) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigenvalues = np.geomspace(1.0, max(float(condition), 1.0), n)
    a = (q * eigenvalues) @ q.T
    return (a + a.T) / 2


def _preview_iteration(contract, form, init, step, constants, iterations, size, condition, seed) -> dict:
    compiled = program.compile_program(init, step)
    n, iterations = int(size), int(iterations)
    if not (2 <= n <= 64) or not 1 <= iterations <= program.MAX_ITERATIONS or not 1.0 <= float(condition) <= 1e8:
        raise ContractError("a solver preview needs 2 to 64 unknowns, 1 to 200 passes and a condition number from 1 to 1e8")
    rng = np.random.default_rng(seed)
    a, b = _spd(n, condition, rng)[None], rng.standard_normal((1, n))
    given = {c: float((constants or {}).get(c, 1.0 if c == "a" else 0.0)) for c in expr.CONSTANTS}
    direct = np.linalg.solve(a[0], b[0])
    scale = float(np.linalg.norm(b[0]))
    history: list[float] = []

    def residual(x) -> float:
        return float(np.linalg.norm(a[0] @ x[0] - b[0]) / scale)

    start = residual(program.run(compiled, a, b, given, 0))
    final = program.run(compiled, a, b, given, iterations, on_pass=lambda k, x: history.append(residual(x)))
    error = float(np.linalg.norm(final[0] - direct) / np.linalg.norm(direct))
    return dict(kind="iteration", unknowns=n, condition=float(condition), start=start, residuals=history,
                final=history[-1], solution_error=error, finite=bool(np.isfinite(history).all()),
                key=compiled.step[0].law.key if compiled.step else "")
