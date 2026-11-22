"""The text a person types for a stage, read against the form the stage declares (``phost.equation``).

* ``Pointwise``: one expression, or one ``name = expression`` line per output the stage writes. An output that is not
  assigned is 0. A stage with a single output also takes a bare expression.
* ``Channels``: one line per column of the output, ``label = expression`` (the label names the column) or a bare expression
  (labelled ``c0``, ``c1``, ...). Names are arrays; reduce them with ``sum``, ``norm``, ... or pick an entry with ``q[3]``.

Lines are separated by newlines or ``;``. Nothing here executes the text: it is parsed and checked into trees
(``phost_author.expr``), and the trees are what the backends turn into code.
"""

from __future__ import annotations

import ast
from typing import Mapping

from phost import Channels, ContractError, Pointwise, StageContract

from . import expr

MAX_LINES = 16


def _statements(text: str) -> list[ast.stmt]:
    if not isinstance(text, str) or not text.strip():
        raise expr.ExpressionError("the equation is empty")
    if len(text) > expr.MAX_CHARS * 4:
        raise expr.ExpressionError(f"the text has {len(text)} characters; the limit is {expr.MAX_CHARS * 4}")
    try:
        body = ast.parse(text.strip().replace("\r", ""), mode="exec").body
    except SyntaxError as error:
        raise expr.ExpressionError(f"syntax error: {error.msg} at line {error.lineno}, column {error.offset}") from None
    if len(body) > MAX_LINES:
        raise expr.ExpressionError(f"{len(body)} lines; the limit is {MAX_LINES}")
    return body


def _assignment(stmt: ast.stmt) -> tuple[str | None, ast.expr]:
    """(target name or None for a bare expression, value tree)."""
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            raise expr.ExpressionError("write one name on the left of '=', like  fx = -c*vx")
        return stmt.targets[0].id, stmt.value
    if isinstance(stmt, ast.Expr):
        return None, stmt.value
    raise expr.ExpressionError(f"{type(stmt).__name__} is not allowed here; write  name = expression")


def expression_of(tree: ast.expr, scope: expr.Scope) -> expr.Expression:
    """An ``Expression`` for a subtree that has been checked against ``scope``."""
    names, axis = expr.check(tree, scope)
    wrapped = ast.fix_missing_locations(ast.Expression(body=tree))
    return expr.Expression(ast.unparse(tree), wrapped, names, axis)


# ---- pointwise ----------------------------------------------------------------------------------------

def pointwise_scope(form: Pointwise) -> expr.Scope:
    return expr.Scope(scalars=tuple(n.name for n in form.inputs))


def parse_pointwise(text: str, form: Pointwise) -> dict[str, expr.Expression]:
    """Output name -> expression, in the order written. Outputs not mentioned are absent (they are 0)."""
    scope, outputs = pointwise_scope(form), [n.name for n in form.outputs]
    if len(outputs) == 1:
        try:
            ast.parse(text.strip(), mode="eval")
        except (SyntaxError, AttributeError):
            pass
        else:
            return {outputs[0]: expr.parse_in(text, scope)}
    try:
        statements = _statements(text)
    except expr.ExpressionError:
        if len(outputs) == 1 and isinstance(text, str) and text.strip() and "=" not in text:
            return {outputs[0]: expr.parse_in(text, scope)}      # the message of a plain bad expression
        raise
    laws: dict[str, expr.Expression] = {}
    for stmt in statements:
        target, value = _assignment(stmt)
        if target is None:
            if len(outputs) != 1:
                raise expr.ExpressionError(f"this stage writes {', '.join(outputs)}: write one line per output, "
                                           f"like  {outputs[0]} = ...")
            target = outputs[0]
        if target not in outputs:
            raise expr.ExpressionError(f"{target!r} is not an output of this stage; it writes {', '.join(outputs)}")
        if target in laws:
            raise expr.ExpressionError(f"{target!r} is assigned twice")
        laws[target] = expression_of(value, scope)
    if not laws:
        raise expr.ExpressionError("the equation is empty")
    return laws


# ---- channels -----------------------------------------------------------------------------------------

def channels_scope(form: Channels, contract: StageContract) -> expr.Scope:
    """Arrays by axis name (the dimension their port's second axis is), scalars for per-world ports."""
    scalars, vectors = [], {}
    for n in form.inputs:
        shape = contract.port(n.port).shape
        if len(shape) == 1:
            scalars.append(n.name)
        else:
            vectors[n.name] = str(shape[1]).rstrip("+")
    return expr.Scope(scalars=tuple(scalars), vectors=vectors, reductions=True, subscripts=True)


def parse_channels(text: str, form: Channels, contract: StageContract) -> list[tuple[str, expr.Expression]]:
    scope = channels_scope(form, contract)
    channels: list[tuple[str, expr.Expression]] = []
    for index, stmt in enumerate(_statements(text)):
        label, value = _assignment(stmt)
        label = label or f"c{index}"
        if label in expr.RESERVED or label in scope.vectors or label in scope.scalars or label.startswith("_"):
            raise expr.ExpressionError(f"{label!r} is a name the equation already uses; pick another label")
        if any(label == other for other, _ in channels):
            raise expr.ExpressionError(f"two channels are labelled {label!r}")
        law = expression_of(value, scope)
        if law.axis is not None:
            raise expr.ExpressionError(f"{label} is an array along {law.axis!r}, and a channel is one number: "
                                       "reduce it with sum(...), norm(...), amax(...) or pick an entry with name[0]")
        channels.append((label, law))
    return channels


def check_indices(channels, sizes: Mapping[str, int], contract: StageContract, form: Channels) -> None:
    """Raise unless every ``name[i]`` is inside the array. ``sizes`` maps an axis name to its length on this host."""
    scope = channels_scope(form, contract)
    for label, law in channels:
        for node in ast.walk(law.tree):
            if isinstance(node, ast.Subscript):
                name, index = node.value.id, node.slice.value
                size = sizes.get(scope.vectors[name])
                if size is not None and index >= size:
                    raise ContractError(f"channel {label}: {name}[{index}] is out of range, {name} has {size} entries on this model")
