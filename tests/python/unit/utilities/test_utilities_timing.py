"""Tests for timing utilities"""

from pathlib import Path
import logging

import pytest

from revrt.utilities import timing, elapsed_time_as_str


@pytest.mark.parametrize(
    ("end_time", "expected"),
    [
        (11.0, "routing executed in 0:00:01"),
        (130.0, "routing executed in 0:02:00"),
        (7210.0, "routing executed in 2:00:00"),
        (86482.0, "routing executed in 1 day, 0:01:12"),
    ],
)
def test_log_time_logs_elapsed_time(caplog, monkeypatch, end_time, expected):
    """Test that log_time logs elapsed time with the formatted duration"""

    times = iter([10.0, end_time])
    monkeypatch.setattr(timing.time, "monotonic", lambda: next(times))

    with (
        caplog.at_level(logging.INFO, logger=timing.logger.name),
        timing.log_time("routing"),
    ):
        pass

    assert caplog.messages == [expected]


def test_log_time_uses_requested_log_level(caplog, monkeypatch):
    """Test that log_time emits at the requested logging level"""

    times = iter([10.0, 11.0])
    monkeypatch.setattr(timing.time, "monotonic", lambda: next(times))

    with (
        caplog.at_level(logging.DEBUG, logger=timing.logger.name),
        timing.log_time("routing", log_level=logging.DEBUG),
    ):
        pass

    assert [(record.levelno, record.message) for record in caplog.records] == [
        (logging.DEBUG, "routing executed in 0:00:01"),
    ]


def test_log_time_logs_when_block_raises(caplog, monkeypatch):
    """Test that log_time still logs elapsed time if the block raises"""

    times = iter([10.0, 11.0])
    monkeypatch.setattr(timing.time, "monotonic", lambda: next(times))

    msg = "A test error"
    with (
        caplog.at_level(logging.INFO, logger=timing.logger.name),
        pytest.raises(RuntimeError, match=msg),
        timing.log_time("routing"),
    ):
        raise RuntimeError(msg)

    assert caplog.messages == ["routing executed in 0:00:01"]


def test_elapsed_time_as_str():
    """Test elapsed_time_as_str utility function"""

    assert elapsed_time_as_str(1) == "0:00:01"
    assert elapsed_time_as_str(46) == "0:00:46"
    assert elapsed_time_as_str(60) == "0:01:00"
    assert elapsed_time_as_str(62) == "0:01:02"
    assert elapsed_time_as_str(1 * 60 * 60) == "1:00:00"
    assert elapsed_time_as_str(1 * 60 * 60 + 42) == "1:00:42"
    assert elapsed_time_as_str(1 * 60 * 60 + 63) == "1:01:03"
    assert elapsed_time_as_str(2 * 60 * 60 + 63) == "2:01:03"
    assert elapsed_time_as_str(13 * 60 * 60 + 63) == "13:01:03"
    assert elapsed_time_as_str(24 * 60 * 60) == "1 day, 0:00:00"
    assert elapsed_time_as_str(24 * 60 * 60 + 72) == "1 day, 0:01:12"
    assert elapsed_time_as_str(50 * 60 * 60 + 72) == "2 days, 2:01:12"


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
