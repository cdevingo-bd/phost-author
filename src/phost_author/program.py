"""A typed linear-solver step: a short program, repeated, that turns ``A x = b`` into an answer.

A person types two blocks of lines, ``name = expression``:

    init   runs once. ``x`` is the running solution; if init does not assign it, it starts at zero.
    step   runs ``iterations`` times, in order. Every name keeps its value from one pass to the next.

The names a program sees are ``rhs`` (the right-hand side) and ``diag`` (the diagonal of ``A``), arrays along the unknowns; ``A`` itself,
usable only as ``A @ name``; ``n`` (the number of unknowns), ``k`` (which pass this is, from 0), the constants ``a`` .. ``h``
and ``pi``. Everything else a program names is its own, scalar or array according to what it was assigned. Arrays work entry
by entry; ``dot``, ``sum``, ``norm``, ``maxabs``, ``mean``, ``amax`` and ``amin`` collapse an array to a number. The program's ``x``
at the end is the answer.

A damped Jacobi step is ``x = x + a * (rhs - A @ x) / diag``. Conjugate gradients with a Jacobi preconditioner is eight lines
(``DEFAULT_*`` below). Nothing a person types is executed as Python: the lines are parsed into the same checked trees as
every other typed equation, and backends turn the trees into code.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from . import expr
from .laws import _assignment, _statements, expression_of

AXIS = "n"
INPUTS = {"rhs": AXIS, "diag": AXIS}
RESERVED_NAMES = frozenset({"rhs", "diag", "A", "n", "k"}) | expr.RESERVED
MAX_ITERATIONS, MAX_NAMES, MAX_STATEMENTS = 200, 12, 12

DEFAULT_INIT = """x = 0*rhs
r = rhs
z = r/diag
p = z
rz = dot(r, z)"""

DEFAULT_STEP = """Ap = A @ p
alpha = rz / max(dot(p, Ap), 1e-30)
x = x + alpha*p
r = r - alpha*Ap
z = r/diag
rz_new = dot(r, z)
p = z + (rz_new / max(rz, 1e-30))*p
rz = rz_new"""


@dataclass(frozen=True)
class Statement:
    target: str
    law: expr.Expression


@dataclass(frozen=True)
class Program:
    init: tuple[Statement, ...]
    step: tuple[Statement, ...]
    vectors: tuple[str, ...]            # the program's own arrays, ``x`` first (``b`` and ``d`` are inputs)
    scalars: tuple[str, ...]            # the program's own scalars

    @property
    def names(self) -> tuple[str, ...]:
        return (*self.vectors, *self.scalars)

    def uses_matrix(self) -> bool:
        return any("A" in s.law.names for s in (*self.init, *self.step))


def _block(text: str, vectors: dict, scalars: list, label: str) -> tuple[Statement, ...]:
    """Check one block in order; ``vectors`` and ``scalars`` grow as names are first assigned."""
    out = []
    if not text.strip():
        return ()
    for stmt in _statements(text):
        target, value = _assignment(stmt)
        if target is None:
            raise expr.ExpressionError(f"{label}: write  name = expression  (a line alone does nothing)")
        if target in RESERVED_NAMES:
            raise expr.ExpressionError(f"{label}: {target!r} is part of the language and cannot be assigned")
        if not target.isidentifier() or target.startswith("_"):
            raise expr.ExpressionError(f"{label}: {target!r} is not a usable name")
        scope = expr.Scope(scalars=("n", "k", *scalars), vectors=dict(vectors), reductions=True, matrix="A", matrix_axis=AXIS)
        try:
            law = expression_of(value, scope)
        except expr.ExpressionError as error:
            raise expr.ExpressionError(f"{label}, {target} = ...: {error}") from None
        kind = law.axis
        if target == "x" and kind is None:
            raise expr.ExpressionError(f"{label}: x is the solution and must be an array, not one number")
        known = vectors.get(target) if target in vectors else (None if target in scalars else False)
        if known is not False and known != kind:
            raise expr.ExpressionError(f"{label}: {target} was {'an array' if target in vectors else 'a number'} and this makes it "
                                       f"{'an array' if kind else 'a number'}")
        if known is False:
            if len(vectors) + len(scalars) - len(INPUTS) >= MAX_NAMES:
                raise expr.ExpressionError(f"a program may name at most {MAX_NAMES} things of its own")
            (vectors.__setitem__(target, kind) if kind else scalars.append(target))
        out.append(Statement(target, law))
    if len(out) > MAX_STATEMENTS:
        raise expr.ExpressionError(f"{label} has {len(out)} lines; the limit is {MAX_STATEMENTS}")
    return tuple(out)


def compile_program(init: str, step: str) -> Program:
    """Parse and check both blocks. Raises ``ExpressionError`` naming the line and what is wrong."""
    vectors: dict[str, str] = {**INPUTS, "x": AXIS}              # x always exists: zero unless init says otherwise
    scalars: list[str] = []
    init_block = _block(init, vectors, scalars, "init")
    step_block = _block(step, vectors, scalars, "step")
    if not step_block:
        raise expr.ExpressionError("step is empty: nothing would change x")
    if not any(s.target == "x" for s in (*init_block, *step_block)):
        raise expr.ExpressionError("the program never assigns x, so the answer would be zero")
    own = tuple(n for n in vectors if n not in INPUTS)
    return Program(init_block, step_block, own, tuple(scalars))


def symmetric(h, n: int, xp=None, dtype=None):
    """The full symmetric matrices of ``h`` (worlds, >= n, >= n), rebuilt from the upper triangle (the rest is undefined)."""
    import numpy as np
    xp, dtype = xp if xp is not None else np, dtype or np.float64
    upper = xp.triu(xp.asarray(h, dtype)[:, :n, :n])
    return upper + xp.swapaxes(upper, 1, 2) - xp.einsum("wi,ij->wij", xp.einsum("wii->wi", upper), xp.eye(n, dtype=dtype))


def run(program: Program, a, b, constants: Mapping[str, float], iterations: int,
        on_pass: Callable[[int, Any], None] | None = None, xp=None, dtype=None):
    """Run the program for every world, in float64 with numpy unless ``xp`` and ``dtype`` say otherwise (the JAX backend runs it
    in float32 with ``jax.numpy``). ``a`` is the full symmetric matrices (worlds, n, n), ``b`` is (worlds, n).

    Returns ``x`` (worlds, n). ``on_pass(k, x)`` is called with the solution after each pass of step.
    """
    import numpy as np
    xp, dtype = xp if xp is not None else np, dtype or np.float64
    a, b = xp.asarray(a, dtype), xp.asarray(b, dtype)
    worlds, n = b.shape
    scalar = lambda value: xp.asarray(float(value), dtype)
    env: dict[str, Any] = {"A": a, "rhs": b, "diag": xp.array(xp.einsum("wii->wi", a)), "n": scalar(n), "k": scalar(0.0),
                           **{c: scalar(constants[c]) for c in expr.CONSTANTS}}
    env["x"] = xp.zeros_like(b)

    def execute(block):
        for statement in block:
            value = xp.asarray(expr.evaluate(statement.law, env, dtype, xp), dtype)
            width = n if statement.law.axis else 1
            env[statement.target] = xp.array(xp.broadcast_to(value, (worlds, width)))

    execute(program.init)
    for k in range(int(iterations)):
        env["k"] = scalar(k)
        execute(program.step)
        if on_pass is not None:
            on_pass(k, env["x"])
    return env["x"]
