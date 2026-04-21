"""reVRt warning tests"""

from pathlib import Path
from warnings import warn

import pytest

from revrt.warn import revrtWarning, revrtDeprecationWarning


BASIC_WARNING_MESSAGE = "A warning message"


@pytest.mark.parametrize(
    "warning_class", [revrtWarning, revrtDeprecationWarning]
)
def test_warnings_log_message(warning_class, assert_message_was_logged):
    """Test that a raised warning logs message, if any"""

    with pytest.warns(warning_class):
        warn(BASIC_WARNING_MESSAGE, warning_class)
    assert_message_was_logged(warning_class.__name__, "WARNING")
    assert_message_was_logged(BASIC_WARNING_MESSAGE, "WARNING")


def test_warning_empty_init_does_not_log_message(caplog):
    """Test that initializing an empty warning does not log a message"""

    revrtWarning()
    assert not caplog.records


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
