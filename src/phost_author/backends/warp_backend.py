"""The Warp emitter: source text for a kernel, from the same checked trees the NumPy emitter evaluates.

Only fixed tables and numbers that were parsed and checked appear in the generated source, so nothing a person typed reaches
the compiler as code. Constants ``a``..``h`` are kernel arguments (changing them does not recompile). Kernel arguments are
named ``p_<port>`` so a port can never collide with a constant or a local.

``pointwise``  one thread per (world, element).
``channels``   one thread per world; a reduction is a loop over the axis.
``iteration``  one thread per world runs the whole program, its arrays in a scratch ``(world, slot, n)`` buffer. This is the
               simple form, correct for any program and fast only for small systems: a person trying methods wants an answer,
               and the opportunity report says what the stage is worth before anyone tunes it.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass

from .. import expr
from ..modules import load_source
from . import ChannelsPlan, IterationPlan, PointwisePlan, register_backend

HEAD = "import warp as wp\n\n\n"
INF = "3.0e38"


def _annotation(rank: int, element: str) -> str:
    return f"wp.array[{element}]" if rank == 1 else f"wp.array{rank}d[{element}]"


def _array_type(port) -> str:
    return _annotation(len(port.shape), "int" if port.dtype.startswith("int") else "float")


@dataclass(frozen=True)
class Layout:
    """How a host stores a port. A contract describes ``cvel`` as ``(world, body, 6)`` floats; a host may keep it as a
    ``(world, body)`` array of 6-vectors. ``composite`` is the vector type's name then, and the kernel indexes ``a[w, i][k]``
    and writes a whole vector at once."""

    annotation: str
    composite: str | None = None


def plain_layout(port) -> Layout:
    """The layout the contract itself describes (floats and ints, one array axis per port axis)."""
    return Layout(_array_type(port))


def layout_of(array) -> Layout:
    """The layout of a Warp array as it is."""
    import warp as wp
    dtype = array.dtype
    if dtype is wp.float32:
        return Layout(_annotation(array.ndim, "float"))
    if dtype is wp.int32:
        return Layout(_annotation(array.ndim, "int"))
    name = f"wp.{dtype.__name__}"
    return Layout(_annotation(array.ndim, name), name)


class Gen:
    """Turns a checked tree into Warp code. A reduction becomes a loop whose result is a temporary scalar, written to ``lines``
    before the expression that uses it; so ``code = gen.expr(tree)`` must be called before the line that uses ``code`` is emitted.

    ``scalar(name)`` and ``vector(name, index)`` give the code for a name; ``matvec(name, index)`` the code of entry ``index`` of
    ``A @ name``; ``size(axis)`` the code of an axis length.
    """

    def __init__(self, lines, depth, *, scalar, vector, size, vector_names, axis_of, matvec=None):
        self.lines, self.depth, self.scalar, self.vector, self.size, self.matvec = lines, depth, scalar, vector, size, matvec
        self.vector_names, self.axis_of = frozenset(vector_names), axis_of
        self.count = 0

    def emit(self, text: str) -> None:
        self.lines.append("    " * self.depth + text)

    def fresh(self, stem: str) -> str:
        self.count += 1
        return f"{stem}{self.count}"

    def expr(self, node, index=None, axes=None) -> str:
        axes = axes or {}

        def go(node, index) -> str:
            if isinstance(node, ast.Constant):
                return repr(float(node.value))
            if isinstance(node, ast.Name):
                if node.id == "pi":
                    return repr(math.pi)
                return self.vector(node.id, index) if node.id in self.vector_names else self.scalar(node.id)
            if isinstance(node, ast.UnaryOp):
                return f"({'-' if isinstance(node.op, ast.USub) else '+'}{go(node.operand, index)})"
            if isinstance(node, ast.BinOp):
                if isinstance(node.op, ast.MatMult):
                    return self.matvec(node.right.id, index)
                if isinstance(node.op, ast.Pow):
                    return f"wp.pow({go(node.left, index)}, {go(node.right, index)})"
                return f"({go(node.left, index)} {expr.BINARY[type(node.op)]} {go(node.right, index)})"
            if isinstance(node, ast.Compare):
                return f"({go(node.left, index)} {expr.COMPARE[type(node.ops[0])]} {go(node.comparators[0], index)})"
            if isinstance(node, ast.Subscript):
                return self.vector(node.value.id, node.slice.value)
            name = node.func.id
            if name in expr.REDUCTIONS:
                return self.reduce(name, node, go)
            return f"wp.{expr.FUNCTIONS[name].warp}({', '.join(go(a, index) for a in node.args)})"
        return go(node, index)

    def reduce(self, name, node, go) -> str:
        """Emit the loop for a reduction (the arguments' own reductions come out first) and return its result."""
        axis = self.axis_of(node.args)
        j, acc = self.fresh("j"), self.fresh("r")
        bodies = [go(a, j) for a in node.args]
        init = {"amax": f"-{INF}", "amin": INF}.get(name, "0.0")
        first = bodies[0]
        update = {"sum": lambda: f"{acc} += {first}", "mean": lambda: f"{acc} += {first}",
                  "amax": lambda: f"{acc} = wp.max({acc}, {first})", "amin": lambda: f"{acc} = wp.min({acc}, {first})",
                  "maxabs": lambda: f"{acc} = wp.max({acc}, wp.abs({first}))",
                  "norm": lambda: f"{acc} += ({first}) * ({first})",
                  "dot": lambda: f"{acc} += ({first}) * ({bodies[1]})"}[name]()
        size = self.size(axis)
        self.emit(f"{acc} = float({init})")
        self.emit(f"for {j} in range({size}):")
        self.lines.append("    " * (self.depth + 1) + update)
        if name == "mean":
            return f"({acc} / float({size}))"
        return f"wp.sqrt({acc})" if name == "norm" else acc


def _vector_axis(vectors):
    """The axis a reduction's arguments run along: the first array name or matrix product found in them."""
    def axis_of(args):
        for argument in args:
            for node in ast.walk(argument):
                if isinstance(node, ast.Name) and node.id in vectors:
                    return vectors[node.id]
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                    return vectors[node.right.id]
        return None
    return axis_of


# ---- pointwise ----------------------------------------------------------------------------------------

def pointwise_source(plan: PointwisePlan, layouts=None) -> tuple[str, dict]:
    """The kernel source and its argument plan: ``{"inputs": [ports], "outputs": [ports]}`` (constants follow the inputs).

    ``layouts`` maps a port name to its ``Layout`` on the host the kernel is for (default: as the contract describes it).
    """
    form, contract = plan.form, plan.contract
    used = set()
    for tree in (*plan.laws.values(), *plan.slopes.values()):
        used |= tree.names
    in_ports: list[str] = []
    for n in form.inputs:
        if n.name in used and n.port not in in_ports:
            in_ports.append(n.port)
    if form.mask:
        in_ports.append(form.mask)
    out_ports: list[str] = []
    for n in form.outputs:
        if n.port not in out_ports:
            out_ports.append(n.port)
    if plan.slopes and form.slope.port not in out_ports:
        out_ports.append(form.slope.port)
    layout = lambda port: (layouts or {}).get(port) or plain_layout(contract.port(port))      # noqa: E731

    def ref(n):
        rank = len(contract.port(n.port).shape)
        if rank == 1:
            return f"p_{n.port}[w]"
        if n.column is None:
            return f"p_{n.port}[w, i]"
        return f"p_{n.port}[w, i][{n.column}]" if layout(n.port).composite else f"p_{n.port}[w, i, {n.column}]"

    emitted = {n.name: ref(n) for n in form.inputs if n.name in used} | {c: c for c in expr.CONSTANTS}
    signature = [f"p_{p}: {layout(p).annotation}" for p in in_ports] + [f"{c}: float" for c in expr.CONSTANTS] + \
                [f"p_{p}: {layout(p).annotation}" for p in out_ports]
    lines = [HEAD + "@wp.kernel", f"def kernel({', '.join(signature)}):", "    w, i = wp.tid()"]
    assign = []                                   # (statement, zero statement) for every output the kernel writes
    whole: dict[str, list] = {}                   # composite port -> its components in order
    for n in form.outputs:
        tree = plan.laws.get(n.name)
        code = expr.to_warp(tree, emitted) if tree is not None else "0.0"
        if n.column is not None and layout(n.port).composite:
            whole.setdefault(n.port, ["0.0"] * contract.port(n.port).shape[-1])[n.column] = code
            continue
        target = f"p_{n.port}[w, i, {n.column}]" if n.column is not None else f"p_{n.port}[w, i]"
        assign.append((f"{target} = {code}", f"{target} = 0.0"))
    for port, parts in whole.items():
        ctor = layout(port).composite
        assign.append((f"p_{port}[w, i] = {ctor}({', '.join(parts)})", f"p_{port}[w, i] = {ctor}({', '.join(['0.0'] * len(parts))})"))
    if plan.slopes:
        (tree,) = plan.slopes.values()
        body, sign = expr.to_warp(tree, emitted), form.slope.sign
        text = f"-({body})" if sign == -1 else f"({body})" if sign == 1 else f"{sign!r} * ({body})"
        assign.append((f"p_{form.slope.port}[w, i] = {text}", f"p_{form.slope.port}[w, i] = 0.0"))
    if form.mask:
        lines.append(f"    if p_{form.mask}[i] == 1:")
        lines += [f"        {a}" for a, _ in assign]
        lines.append("    else:")
        lines += [f"        {z}" for _, z in assign]
    else:
        lines += [f"    {a}" for a, _ in assign]
    return "\n".join(lines) + "\n", dict(inputs=in_ports, outputs=out_ports)


# ---- channels -----------------------------------------------------------------------------------------

def channels_source(plan: ChannelsPlan, scope) -> tuple[str, dict]:
    form, contract = plan.form, plan.contract
    used = set()
    for _, law in plan.channels:
        used |= law.names
    names = [n for n in form.inputs if n.name in used]
    ports = [n.port for n in names]
    axes = {}                                      # axis name -> first port it was read from
    for n in names:
        if n.name in scope.vectors:
            axes.setdefault(scope.vectors[n.name], n.port)
    by_name = {n.name: n for n in names}
    signature = [f"p_{p}: {_array_type(contract.port(p))}" for p in ports] + [f"{c}: float" for c in expr.CONSTANTS] + \
                [f"n_{axis}: int" for axis in axes] + [f"p_{form.output}: {_array_type(contract.port(form.output))}"]
    lines = [HEAD + "@wp.kernel", f"def kernel({', '.join(signature)}):", "    w = wp.tid()"]
    g = Gen(lines, 1,
             scalar=lambda name: f"p_{by_name[name].port}[w]" if name in by_name else name,
             vector=lambda name, index: f"p_{by_name[name].port}[w, {index}]",
             size=lambda axis: f"n_{axis}", vector_names=scope.vectors, axis_of=_vector_axis(scope.vectors))
    for column, (_, law) in enumerate(plan.channels):
        code = g.expr(law.tree.body)
        g.emit(f"p_{form.output}[w, {column}] = {code}")
    return "\n".join(lines) + "\n", dict(inputs=ports, axes=axes)


# ---- iteration ----------------------------------------------------------------------------------------

MATVEC = '''@wp.func
def matvec(a: wp.array3d[float], vec: wp.array3d[float], w: int, slot: int, i: int, n: int) -> float:
    s = float(0.0)
    for j in range(n):
        s += a[w, wp.min(i, j), wp.max(i, j)] * vec[w, slot, j]
    return s


'''


def iteration_source(plan: IterationPlan) -> tuple[str, dict]:
    form, program = plan.form, plan.program
    slots = {"rhs": 0, "diag": 1}
    for name in program.vectors:
        slots[name] = len(slots)
    tmp, count = len(slots), len(slots) + 1
    outputs = [form.solution] + [port for port, _ in form.derived]
    signature = ["p_a: wp.array3d[float]", "p_rhs: wp.array2d[float]"] + (["p_active: wp.array[int]"] if form.active else []) + \
                ["vec: wp.array3d[float]"] + [f"{c}: float" for c in expr.CONSTANTS] + ["n: int", "iterations: int"] + \
                ["p_out: wp.array2d[float]"] + [f"p_derived{i}: wp.array[float]" for i in range(len(form.derived))]
    lines = [HEAD + MATVEC + "@wp.kernel", f"def kernel({', '.join(signature)}):", "    w = wp.tid()"]
    if form.active:
        lines += ["    if p_active[w] == 0:", "        return"]
    lines += ["    for i in range(n):", f"        vec[w, {slots['rhs']}, i] = p_rhs[w, i]", f"        vec[w, {slots['diag']}, i] = p_a[w, i, i]"]
    lines += [f"        vec[w, {slots[name]}, i] = 0.0" for name in program.vectors]
    lines += [f"    s_{name} = float(0.0)" for name in program.scalars]
    lines += ["    s_k = float(0.0)"]

    axis_of = _vector_axis({name: "n" for name in (*slots, )})

    def statements(block, depth):
        g = Gen(lines, depth,
                 scalar=lambda name: {"n": "float(n)", "k": "s_k"}.get(name, name if name in expr.CONSTANTS else f"s_{name}"),
                 vector=lambda name, index: f"vec[w, {slots[name]}, {index}]",
                 size=lambda axis: "n", vector_names=slots, axis_of=axis_of,
                 matvec=lambda name, index: f"matvec(p_a, vec, w, {slots[name]}, {index}, n)")
        pad = "    " * depth
        for statement in block:
            tree, target = statement.law.tree.body, statement.target
            code = g.expr(tree, "i")
            if statement.law.axis is None:
                lines.append(f"{pad}s_{target} = {code}")
                continue
            reads_itself = any(isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult) and node.right.id == target
                               for node in ast.walk(tree))
            slot = slots[target]
            if reads_itself:                       # entry i of A @ x needs all of x as it was: build the new one aside
                lines.extend([f"{pad}for i in range(n):", f"{pad}    vec[w, {tmp}, i] = {code}",
                              f"{pad}for i in range(n):", f"{pad}    vec[w, {slot}, i] = vec[w, {tmp}, i]"])
            else:
                lines.extend([f"{pad}for i in range(n):", f"{pad}    vec[w, {slot}, i] = {code}"])

    statements(program.init, 1)
    lines += ["    for k in range(iterations):", "        s_k = float(k)"]
    statements(program.step, 2)
    x = slots["x"]
    sign = form.sign
    lines += ["    for i in range(n):", f"        p_out[w, i] = {sign!r} * vec[w, {x}, i]"]
    for index, (_, quantity) in enumerate(form.derived):
        other = f"vec[w, {x}, i]" if quantity == "square_norm" else "p_rhs[w, i]"
        lines += ["    q = float(0.0)", "    for i in range(n):", f"        q += vec[w, {x}, i] * {other}",
                  f"    p_derived{index}[w] = q"]
    return "\n".join(lines) + "\n", dict(slots=count, outputs=outputs)


# ---- the backend --------------------------------------------------------------------------------------

def _constants(constants):
    return [float(constants[c]) for c in expr.CONSTANTS]


class _Warp:
    name = "warp"

    def pointwise(self, plan: PointwisePlan):
        """Built on first use: how a host stores a port (floats, or vectors) is known from the arrays, not the contract."""
        import warp as wp
        built: dict = {}
        scratch: dict = {}
        contract, form = plan.contract, plan.form

        def build(system):
            _, args = pointwise_source(plan)
            layouts = {}
            for port in (*args["inputs"], *args["outputs"]):
                array = getattr(system, port, None)
                layouts[port] = layout_of(array) if array is not None else plain_layout(contract.port(port))
            source, args = pointwise_source(plan, layouts)
            return load_source(source, f"phost_law_{contract.name.replace('.', '_')}").kernel, args

        def run(system, constants):
            if "kernel" not in built:
                built["kernel"], built["args"] = build(system)
            kernel, args = built["kernel"], built["args"]
            outs = []
            for port in args["outputs"]:
                array = getattr(system, port)
                if array is None:                                  # the host does not take the slope: the kernel still writes it
                    first = getattr(system, args["outputs"][0])
                    array = scratch.get(port)
                    if array is None:
                        array = scratch[port] = wp.zeros(first.shape[:2], dtype=float, device=first.device)
                outs.append(array)
            inputs = [getattr(system, p) for p in args["inputs"]] + _constants(constants)
            wp.launch(kernel, dim=tuple(outs[0].shape[:2]), inputs=inputs, outputs=outs, device=outs[0].device)
        return run

    def channels(self, plan: ChannelsPlan):
        import warp as wp
        from ..laws import channels_scope
        scope = channels_scope(plan.form, plan.contract)
        source, args = channels_source(plan, scope)
        kernel = load_source(source, f"phost_law_{plan.contract.name.replace('.', '_')}").kernel

        def run(system, constants):
            out = getattr(system, plan.form.output)
            sizes = [int(getattr(system, port).shape[1]) for port in args["axes"].values()]
            inputs = [getattr(system, p) for p in args["inputs"]] + _constants(constants) + sizes
            wp.launch(kernel, dim=out.shape[0], inputs=inputs, outputs=[out], device=out.device)
        return run

    def iteration(self, plan: IterationPlan):
        import warp as wp
        form = plan.form
        source, args = iteration_source(plan)
        kernel = load_source(source, f"phost_law_{plan.contract.name.replace('.', '_')}").kernel
        scratch = []

        def run(system, constants):
            out = getattr(system, form.solution)
            worlds, n = out.shape
            if not scratch or tuple(scratch[0].shape) != (worlds, args["slots"], n):
                scratch[:] = [wp.zeros((worlds, args["slots"], n), dtype=float, device=out.device)]
            inputs = [getattr(system, form.matrix), getattr(system, form.rhs)] + \
                     ([getattr(system, form.active)] if form.active else []) + scratch + _constants(constants) + \
                     [int(n), int(plan.iterations)]
            outputs = [out] + [getattr(system, port) for port, _ in form.derived]
            wp.launch(kernel, dim=worlds, inputs=inputs, outputs=outputs, device=out.device)
        return run


register_backend(_Warp())
