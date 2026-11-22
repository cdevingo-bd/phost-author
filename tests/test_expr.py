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


# -- derivative ---------------------------------------------------------------------------------------

DERIVATIVES = [
    "-a*q - c*v", "-c*v*abs(v)", "v**3 + 2*v", "a*sin(b*v)", "tanh(v) - exp(-v)", "sqrt(v*v + 1) + log(v*v + 2)",
    "atan(c*v)", "tan(0.3*v)", "cos(v)*q", "(q + v)/(1 + v*v)", "a**v", "q**2 + v**2", "min(a*v, v*v)", "max(v, -2*v)",
    "clamp(2*v, -1, 1)", "where(v > 0, a*v, -b*v*v)", "time*v + q", "5.0", "q - time",
]


@pytest.mark.parametrize("text", DERIVATIVES)
def test_the_derivative_matches_central_differences_of_the_equation(text):
    expression = parse(text, VARS)
    slope = expr.differentiate(expression, "v")
    rng = np.random.default_rng(0)
    constants = dict(a=1.3, b=0.7, c=0.4)
    v = rng.uniform(0.2, 1.5, 40) * rng.choice([-1.0, 1.0], 40)           # away from 0, where abs and where have kinks
    values = {"q": rng.uniform(-1, 1, 40), "time": rng.uniform(0, 3, 40), **constants}
    h = 1e-6
    want = (evaluate(expression, {**values, "v": v + h}) - evaluate(expression, {**values, "v": v - h})) / (2 * h)
    got = evaluate(slope, {**values, "v": v})
    assert np.allclose(got, want, rtol=1e-5, atol=1e-7), np.abs(np.asarray(got) - want).max()


def test_the_derivative_of_a_law_that_ignores_the_variable_is_zero_and_says_so():
    slope = expr.differentiate(parse("a*q + sin(time)", VARS), "v")
    assert slope.text == "0.0" and slope.names == frozenset()


def test_a_simple_damper_differentiates_to_its_coefficient_without_clutter():
    slope = expr.differentiate(parse("-c*v", VARS), "v")
    assert slope.names == {"c"} and float(evaluate(slope, {"c": 0.8, "v": 3.0})) == pytest.approx(-0.8)


def test_the_derivative_uses_only_what_the_grammar_allows():
    """Generated Warp source is built from the same tables, so a derivative must stay inside them."""
    import ast
    for text in DERIVATIVES:
        tree = expr.differentiate(parse(text, VARS), "v").tree
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                assert node.func.id in expr.FUNCTIONS
            assert not isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda))
