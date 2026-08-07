"""Tests for parsing utilities"""

from pathlib import Path

import pytest

from revrt.exceptions import revrtConfigurationError, revrtTypeError
from revrt.utilities.parsing import (
    normalize_str_list_input,
    parse_comparison_values,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("layer", ["layer"]),
        (["layer_a", "layer_b"], ["layer_a", "layer_b"]),
        (("layer_a", "layer_b"), ["layer_a", "layer_b"]),
    ],
)
def test_normalize_str_list_input(value, expected):
    """Test normalizing None, strings, and iterable string inputs"""

    assert normalize_str_list_input(value, "layers") == expected


def test_normalize_str_list_input_generator():
    """Test normalizing an iterator materializes its values"""

    value = (layer for layer in ("layer_a", "layer_b"))

    assert normalize_str_list_input(value, "layers") == ["layer_a", "layer_b"]


@pytest.mark.parametrize("value", [1, object()])
def test_normalize_str_list_input_invalid(value):
    """Test scalar inputs raise a type error with the input name"""

    with pytest.raises(
        revrtTypeError,
        match="layers must be a string or an iterable of strings",
    ):
        normalize_str_list_input(value, "layers")


@pytest.mark.parametrize(
    ("comparison_values", "expected"),
    [
        ("> 10", ("gt", 10.0)),
        (">= -5.25", ("ge", -5.25)),
        ("<.5", ("lt", 0.5)),
        ("<= 1e3", ("le", 1000.0)),
        ("== 0", ("eq", 0.0)),
        ("!= -2E-2", ("ne", -0.02)),
    ],
)
def test_parse_comparison_values(comparison_values, expected):
    """Test parsing supported comparison operators and numeric values"""

    assert parse_comparison_values(comparison_values) == expected


@pytest.mark.parametrize("comparison_values", ["10", "~= 5", "> foo"])
def test_parse_comparison_values_invalid(comparison_values):
    """Test invalid comparison strings raise configuration error"""

    with pytest.raises(
        revrtConfigurationError,
        match="Barrier values must use one of the supported comparison",
    ):
        parse_comparison_values(comparison_values)


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
