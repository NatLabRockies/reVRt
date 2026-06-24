"""reVrt parsing utilities"""

import re

from revrt.exceptions import revrtConfigurationError


_COMPARISON_VALUE_PATTERN = re.compile(
    r"^\s*(!=|>=|<=|==|>|<)\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)
_OPERATOR_MAP = {
    "!=": "ne",
    ">": "gt",
    ">=": "ge",
    "<": "lt",
    "<=": "le",
    "==": "eq",
}


def parse_comparison_values(comparison_values):
    """Parse comparison text into an operator and threshold

    Parameters
    ----------
    comparison_values : str
        Comparison definition describing barrier cells.

    Returns
    -------
    str, float
        Tuple of operator and threshold value. The operator is one of
        the following: 'ne', 'gt', 'ge', 'lt', 'le', 'eq'. The threshold
        is a float value.

    Raises
    ------
    revrtConfigurationError
        If the comparison_values string does not match the expected
        pattern of a comparison operator followed by a number.
    """
    match = _COMPARISON_VALUE_PATTERN.fullmatch(comparison_values)
    if match is None:
        msg = (
            "Barrier values must use one of the supported comparison "
            "operators ('==', '!=', '>', '>=', '<', '<=') followed by a "
            f"number. Got: {comparison_values!r}"
        )
        raise revrtConfigurationError(msg)

    operator, threshold = match.groups()
    return _OPERATOR_MAP[operator], float(threshold)
