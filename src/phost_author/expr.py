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
import math
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


# -- derivative -------------------------------------------------------------------------------------
def _num(value):
    return ast.Constant(value=float(value))


def _is(node, value) -> bool:
    return isinstance(node, ast.Constant) and float(node.value) == value


def _neg(a):
    return _num(-a.value) if isinstance(a, ast.Constant) else ast.UnaryOp(op=ast.USub(), operand=a)


def _add(a, b):
    return b if _is(a, 0) else a if _is(b, 0) else ast.BinOp(left=a, op=ast.Add(), right=b)


def _sub(a, b):
    return a if _is(b, 0) else _neg(b) if _is(a, 0) else ast.BinOp(left=a, op=ast.Sub(), right=b)


def _mul(a, b):
    if _is(a, 0) or _is(b, 0):
        return _num(0)
    return b if _is(a, 1) else a if _is(b, 1) else ast.BinOp(left=a, op=ast.Mult(), right=b)


def _div(a, b):
    return _num(0) if _is(a, 0) else a if _is(b, 1) else ast.BinOp(left=a, op=ast.Div(), right=b)


def _call(name, *arguments):
    return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=list(arguments), keywords=[])


def _where(condition, yes, no):
    return _num(0) if _is(yes, 0) and _is(no, 0) else _call("where", condition, yes, no)


def _compare(op, left, right):
    return ast.Compare(left=left, ops=[op()], comparators=[right])


def differentiate(expression: Expression, name: str) -> Expression:
    """The expression's derivative with respect to the variable ``name``, as an expression over the same names.

    Piecewise functions (abs, min, max, clamp, where) take the derivative of the branch they select, so at a kink the
    result is one of the one-sided derivatives. Built from the checked tree only, so it stays inside the grammar's
    functions and operators.
    """
    def d(node):
        if isinstance(node, ast.Constant):
            return _num(0)
        if isinstance(node, ast.Name):
            return _num(1 if node.id == name else 0)
        if isinstance(node, ast.UnaryOp):
            inner = d(node.operand)
            return _neg(inner) if isinstance(node.op, ast.USub) else inner
        if isinstance(node, ast.BinOp):
            left, right = node.left, node.right
            dl, dr = d(left), d(right)
            if isinstance(node.op, ast.Add):
                return _add(dl, dr)
            if isinstance(node.op, ast.Sub):
                return _sub(dl, dr)
            if isinstance(node.op, ast.Mult):
                return _add(_mul(dl, right), _mul(left, dr))
            if isinstance(node.op, ast.Div):
                return _sub(_div(dl, right), _div(_mul(left, dr), _mul(right, right)))
            if _is(dl, 0) and _is(dr, 0):                                  # x ** y, neither depends on the variable
                return _num(0)
            if _is(dr, 0):                                                 # y does not depend on it: y x^(y-1) x'
                return _mul(_mul(right, ast.BinOp(left=left, op=ast.Pow(), right=_sub(right, _num(1)))), dl)
            return _mul(node, _add(_mul(dr, _call("log", left)), _div(_mul(right, dl), left)))
        arguments = node.args                                              # a call
        function = node.func.id
        if function == "where":                                            # where(condition, yes, no): the condition is not differentiated
            return _where(arguments[0], d(arguments[1]), d(arguments[2]))
        derivatives = [d(a) for a in arguments]
        x, dx = arguments[0], derivatives[0]
        if function == "sin":
            return _mul(_call("cos", x), dx)
        if function == "cos":
            return _neg(_mul(_call("sin", x), dx))
        if function == "tan":
            return _div(dx, _mul(_call("cos", x), _call("cos", x)))
        if function == "tanh":
            return _mul(_sub(_num(1), _mul(_call("tanh", x), _call("tanh", x))), dx)
        if function == "atan":
            return _div(dx, _add(_num(1), _mul(x, x)))
        if function == "exp":
            return _mul(_call("exp", x), dx)
        if function == "log":
            return _div(dx, x)
        if function == "sqrt":
            return _div(dx, _mul(_num(2), _call("sqrt", x)))
        if function == "abs":
            return _where(_compare(ast.GtE, x, _num(0)), dx, _neg(dx))
        if function == "min":
            return _where(_compare(ast.LtE, x, arguments[1]), dx, derivatives[1])
        if function == "max":
            return _where(_compare(ast.GtE, x, arguments[1]), dx, derivatives[1])
        if function == "clamp":
            low, high = arguments[1], arguments[2]
            return _where(_compare(ast.Lt, x, low), derivatives[1], _where(_compare(ast.Gt, x, high), derivatives[2], dx))
        raise ExpressionError(f"cannot differentiate {function}()")        # unreachable: the grammar's functions are all handled

    body = d(expression.tree.body)
    tree = ast.fix_missing_locations(ast.Expression(body=body))
    names = frozenset(node.id for node in ast.walk(tree) if isinstance(node, ast.Name))
    return Expression(ast.unparse(body), tree, names)


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
