"""Tests for parsing utilities"""

from pathlib import Path

import pytest

from revrt.exceptions import revrtConfigurationError
from revrt.utilities.parsing import parse_comparison_values


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
