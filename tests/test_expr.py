import numpy as np
import pytest

from phost_author import ExpressionError, evaluate, parse
from phost_author import expr

VARS = ("q", "v", "time")


def ev(text, **values):
    values = {"q": 0.0, "v": 0.0, "time": 0.0, **values}
    return float(evaluate(parse(text, VARS), values))


def test_arithmetic_and_precedence():
    assert ev("1 + 2*3 - 4/2") == 5.0
    assert ev("-q**2", q=3.0) == -9.0                    # unary minus binds looser than **, as in Python
    assert ev("(1+2)**2") == 9.0


def test_variables_constants_and_pi():
    assert ev("a*q + b*v + c", q=2.0, v=3.0, a=1.0, b=10.0, c=0.5) == 32.5
    assert ev("sin(pi/2)") == pytest.approx(1.0)


def test_functions_and_where():
    assert ev("max(q, v)", q=1.0, v=4.0) == 4.0
    assert ev("clamp(q, -1, 1)", q=5.0) == 1.0
    assert ev("where(q > 0, q, 0)", q=-2.0) == 0.0
    assert ev("where(q > 0, q, 0)", q=2.0) == 2.0
    assert ev("tanh(q)", q=0.5) == pytest.approx(np.tanh(0.5))


def test_arrays_broadcast_over_worlds_and_dofs():
    q = np.arange(6.0).reshape(2, 3)
    out = evaluate(parse("q*time", VARS), dict(q=q, v=0, time=np.array([[1.0], [10.0]])))
    assert out.shape == (2, 3) and out[1, 2] == 50.0


@pytest.mark.parametrize("text", [
    "__import__('os').system('true')", "q.real", "q[0]", "(lambda: 1)()", "[1, 2]", "open('x')", "q if v else 1",
    "q and v", "not q", "q % 2", "q // 2", "q << 1", "q < v", "1 < q < 2", "'text'", "True", "q; v", "eval('1')",
    "sin(q, v)", "sin()", "unknown(q)", "sin(x=q)", "x", "1e999", "", "   ", "q +", "where(q, 1, 2, 3)",
])
def test_bad_expressions_are_refused(text):
    with pytest.raises(ExpressionError):
        parse(text, VARS)


def test_names_must_be_provided_by_the_stage():
    parse("q + v", ("q", "v"))
    with pytest.raises(ExpressionError, match="unknown name 'time'"):
        parse("time", ("q", "v"))


def balanced(depth):
    return "q" if depth == 0 else f"({balanced(depth - 1)}+{balanced(depth - 1)})"


def test_size_limits(monkeypatch):
    with pytest.raises(ExpressionError, match="characters"):
        parse("1+" * 400 + "1", VARS)
    with pytest.raises(ExpressionError, match="nests deeper"):
        parse("+".join(["q"] * 150), VARS)               # a long chain nests deeply
    parse(balanced(5), VARS)                             # 63 elements, depth 5: fine
    monkeypatch.setattr(expr, "MAX_NODES", 20)
    with pytest.raises(ExpressionError, match="more than 20 elements"):
        parse(balanced(5), VARS)                         # wide and shallow: too many elements


def test_key_is_stable_under_spacing_and_distinguishes_equations():
    assert parse("a*q+b", VARS).key == parse("a * q + b", VARS).key
    assert parse("a*q+b", VARS).key != parse("a*q-b", VARS).key


def test_generated_source_contains_only_checked_pieces():
    e = parse("where(q > 0, -a*q**2, b*sin(time)) + 1e-3", VARS)
    source = expr.to_warp(e, {"q": "q[w, i]", "v": "v[w, i]", "time": "time[w]", **{c: c for c in expr.CONSTANTS}})
    assert source == "(wp.where((q[w, i] > 0.0), ((-a) * wp.pow(q[w, i], 2.0)), (b * wp.sin(time[w]))) + 0.001)"
    compile(source, "<generated>", "eval")           # valid Python text
