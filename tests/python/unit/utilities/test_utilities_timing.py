"""Tests for timing utilities"""

from pathlib import Path
import logging

import pytest

from revrt.utilities import timing


@pytest.mark.parametrize(
    ("hours", "end_time", "expected"),
    [
        (False, 130.0, "routing completed in 2.00 minute(s)"),
        (True, 7210.0, "routing completed in 2.00 hour(s)"),
    ],
)
def test_log_time_logs_elapsed_time(
    caplog, monkeypatch, hours, end_time, expected
):
    """Test that log_time logs elapsed time in the requested units"""

    times = iter([10.0, end_time])
    monkeypatch.setattr(timing.time, "monotonic", lambda: next(times))

    with (
        caplog.at_level(logging.INFO, logger=timing.logger.name),
        timing.log_time("routing", hours=hours),
    ):
        pass

    assert caplog.messages == [expected]


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
