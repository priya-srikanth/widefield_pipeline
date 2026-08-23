"""The chunked G8f figure must actually render.

2026-08-23: a trailing comma closed the suptitle argument, so the next line's `+` became a UNARY
plus on a string. Both spatial_reorganisation invocations died with

    TypeError: bad operand type for unary +: 'str'

and nightly_figs recorded a failed step, which would have made the deck refuse to publish. The
figure code is not unit-testable without a full record tree, but the shape of the bug is: an
expression that must CONCATENATE and silently became unary.
"""
import ast
import inspect

from wfield_local import spatial_reorganisation as sr


def test_no_unary_plus_on_a_string_in_the_chunk_plotter():
    """A `+` at the start of a continuation line is concatenation only if the previous line does
    NOT end the expression. Walk the AST and assert no UnaryAdd survives."""
    src = inspect.getsource(sr)
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_plot_chunk")
    unary = [n for n in ast.walk(fn) if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.UAdd)]
    assert not unary, (
        f"{len(unary)} unary + in _plot_chunk -- a continuation-line concatenation lost its left "
        f"operand (line(s) {[n.lineno + fn.lineno - 1 for n in unary]})")


def test_the_part_suffix_concatenates():
    part, n_parts = 2, 3
    got = "BASE." + (f"   [part {part} of {n_parts}]" if n_parts > 1 else "")
    assert got == "BASE.   [part 2 of 3]"
    assert "BASE." + ("" if 1 > 1 else "") == "BASE."
