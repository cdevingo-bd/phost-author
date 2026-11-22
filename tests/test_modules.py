"""Loading generated source as a module, which Warp needs because it reads kernel source from a file."""

import inspect
import sys

import pytest

from phost_author.modules import load_source


def test_equal_source_is_one_module_and_different_source_is_not():
    a = load_source("VALUE = 1\n", "phost_test")
    assert load_source("VALUE = 1\n", "phost_test") is a
    b = load_source("VALUE = 2\n", "phost_test")
    assert b is not a and (a.VALUE, b.VALUE) == (1, 2)


def test_the_module_is_importable_by_name_and_the_prefix_is_in_it():
    module = load_source("X = 5\n", "phost_named")
    assert module.__name__.startswith("phost_named_") and sys.modules[module.__name__] is module


def test_functions_can_be_read_back_as_source():
    """The reason this exists: Warp calls inspect.getsource on a kernel, which needs a file behind the module."""
    module = load_source("def f(x):\n    return x + 1\n", "phost_readback")
    assert "return x + 1" in inspect.getsource(module.f)


def test_source_that_fails_to_run_leaves_nothing_registered():
    source = "raise RuntimeError('boom')\n"
    with pytest.raises(RuntimeError, match="boom"):
        load_source(source, "phost_broken")
    assert not [name for name in sys.modules if name.startswith("phost_broken_")]
    with pytest.raises(RuntimeError, match="boom"):                      # and a retry runs it again, not a stale module
        load_source(source, "phost_broken")


def test_a_syntax_error_is_reported_not_swallowed():
    with pytest.raises(SyntaxError):
        load_source("def (:\n", "phost_syntax")
