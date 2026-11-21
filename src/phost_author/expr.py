"""Typed equations for pointwise stages: parse, check, evaluate in float64, generate a Warp kernel.

A user types an expression such as ``-a*(q - b) - c*v + d*sin(e*time)``. This module accepts a small
grammar only, so an expression is data, not code:

* numbers, the names the stage provides (``q``, ``v``, ``time`` ...), the free constants ``a`` .. ``h``,
  and ``pi``;
* unary ``+ -``, binary ``+ - * / **``, comparisons ``< <= > >= == !=`` (only as the condition of ``where``);
* the functions in ``FUNCTIONS``.

Anything else (attribute access, subscripts, lambdas, other names, other operators, huge expressions) is
refused with a message naming the offending piece. The float64 reference evaluates the checked tree; the
Warp source is generated from the same tree from fixed tables, so nothing the user typed reaches the
generated code except numbers that were parsed and checked finite.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
import os
import sys
import tempfile
import atexit
import shutil
from dataclasses import dataclass
from typing import Any, Mapping

from phost import ContractError

CONSTANTS = tuple("abcdefgh")                 # free constants, set per implementation as parameters
MAX_CHARS, MAX_NODES, MAX_DEPTH = 500, 200, 30


@dataclass(frozen=True)
class Function:
    arity: int
    numpy: str                                 # attribute of numpy
    warp: str                                  # attribute of warp


FUNCTIONS: Mapping[str, Function] = {
    "sin": Function(1, "sin", "sin"), "cos": Function(1, "cos", "cos"), "tan": Function(1, "tan", "tan"),
    "tanh": Function(1, "tanh", "tanh"), "atan": Function(1, "arctan", "atan"), "exp": Function(1, "exp", "exp"),
    "log": Function(1, "log", "log"), "sqrt": Function(1, "sqrt", "sqrt"), "abs": Function(1, "abs", "abs"),
    "min": Function(2, "minimum", "min"), "max": Function(2, "maximum", "max"),
    "clamp": Function(3, "clip", "clamp"), "where": Function(3, "where", "where"),
}
_COMPARE = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}
_BINARY = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}


class ExpressionError(ContractError):
    """The text is not an acceptable expression."""


@dataclass(frozen=True)
class Expression:
    """A checked expression: the tree, the names it uses, and a stable key."""

    text: str
    tree: ast.Expression
    names: frozenset

    @property
    def key(self) -> str:
        return hashlib.sha256(ast.dump(self.tree).encode()).hexdigest()[:16]


def parse(text: str, variables: tuple[str, ...]) -> Expression:
    """Check ``text`` against the grammar. ``variables`` are the names the stage provides."""
    if not isinstance(text, str) or not text.strip():
        raise ExpressionError("the expression is empty")
    if len(text) > MAX_CHARS:
        raise ExpressionError(f"the expression has {len(text)} characters; the limit is {MAX_CHARS}")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as error:
        raise ExpressionError(f"syntax error: {error.msg} at column {error.offset}") from None
    allowed = set(variables) | set(CONSTANTS) | {"pi"}
    names: set[str] = set()
    count = [0]

    def visit(node, depth, in_where_condition=False):
        count[0] += 1
        if count[0] > MAX_NODES:
            raise ExpressionError(f"the expression has more than {MAX_NODES} elements")
        if depth > MAX_DEPTH:
            raise ExpressionError(f"the expression nests deeper than {MAX_DEPTH}")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ExpressionError(f"only numbers are allowed, found {node.value!r}")
            if not math.isfinite(float(node.value)):
                raise ExpressionError("numbers must be finite")
        elif isinstance(node, ast.Name):
            if node.id not in allowed:
                raise ExpressionError(f"unknown name {node.id!r}; this stage provides {sorted(allowed)}")
            names.add(node.id)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            visit(node.operand, depth + 1)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (*_BINARY, ast.Pow)):
            visit(node.left, depth + 1)
            visit(node.right, depth + 1)
        elif isinstance(node, ast.Compare):
            if not in_where_condition:
                raise ExpressionError("a comparison is only allowed as the first argument of where(...)")
            if len(node.ops) != 1 or type(node.ops[0]) not in _COMPARE:
                raise ExpressionError("use one comparison at a time (<, <=, >, >=, ==, !=)")
            visit(node.left, depth + 1)
            visit(node.comparators[0], depth + 1)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS or node.keywords:
                raise ExpressionError(f"unknown function; allowed: {sorted(FUNCTIONS)}")
            spec = FUNCTIONS[node.func.id]
            if len(node.args) != spec.arity:
                raise ExpressionError(f"{node.func.id}() takes {spec.arity} arguments, got {len(node.args)}")
            for index, argument in enumerate(node.args):
                visit(argument, depth + 1, in_where_condition=node.func.id == "where" and index == 0)
        else:
            raise ExpressionError(f"{type(node).__name__} is not allowed in an expression")

    visit(tree.body, 0)
    return Expression(text.strip(), tree, frozenset(names))


# -- float64 reference ------------------------------------------------------------------------------
def evaluate(expression: Expression, values: Mapping[str, Any], dtype=None):
    """Evaluate in numpy (float64 unless ``dtype`` says otherwise). ``values`` maps names to arrays or numbers."""
    import numpy as np
    dtype = dtype or np.float64
    env = {name: np.asarray(value, dtype) for name, value in values.items()}
    env["pi"] = np.asarray(math.pi, dtype)

    def go(node):
        if isinstance(node, ast.Constant):
            return np.asarray(node.value, dtype)
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.UnaryOp):
            value = go(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = go(node.left), go(node.right)
            if isinstance(node.op, ast.Pow):
                return np.power(left, right)
            return {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide}[type(node.op)](left, right)
        if isinstance(node, ast.Compare):
            return {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal, "==": np.equal,
                    "!=": np.not_equal}[_COMPARE[type(node.ops[0])]](go(node.left), go(node.comparators[0]))
        function = getattr(np, FUNCTIONS[node.func.id].numpy)
        return function(*[go(argument) for argument in node.args])

    with np.errstate(all="ignore"):
        return go(expression.tree.body)


# -- Warp source ------------------------------------------------------------------------------------
def to_warp(expression: Expression, emitted: Mapping[str, str]) -> str:
    """Warp source text of the expression. ``emitted`` maps each allowed name to the identifier used in the kernel.

    Only fixed tables and parsed numbers appear in the output.
    """
    def go(node) -> str:
        if isinstance(node, ast.Constant):
            return repr(float(node.value))
        if isinstance(node, ast.Name):
            return repr(math.pi) if node.id == "pi" else emitted[node.id]
        if isinstance(node, ast.UnaryOp):
            return f"({'-' if isinstance(node.op, ast.USub) else '+'}{go(node.operand)})"
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                return f"wp.pow({go(node.left)}, {go(node.right)})"
            return f"({go(node.left)} {_BINARY[type(node.op)]} {go(node.right)})"
        if isinstance(node, ast.Compare):
            return f"({go(node.left)} {_COMPARE[type(node.ops[0])]} {go(node.comparators[0])})"
        return f"wp.{FUNCTIONS[node.func.id].warp}({', '.join(go(a) for a in node.args)})"

    return go(expression.tree.body)


_DIR: list[str] = []


def _source_dir() -> str:
    if not _DIR:
        path = tempfile.mkdtemp(prefix="phost_expr_")
        _DIR.append(path)
        atexit.register(shutil.rmtree, path, ignore_errors=True)
    return _DIR[0]


def build_module(name: str, source: str):
    """Import generated source as a module (Warp reads kernel source from a file)."""
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(_source_dir(), f"{name}.py")
    with open(path, "w") as stream:
        stream.write(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
