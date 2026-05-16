"""Tests for monitoring utilities"""

import logging
from pathlib import Path

import pytest

from revrt.utilities import monitoring, elapsed_time_as_str


@pytest.mark.parametrize(
    ("end_time", "expected"),
    [
        (11.0, "routing took 0:00:01 to run"),
        (130.0, "routing took 0:02:00 to run"),
        (7210.0, "routing took 2:00:00 to run"),
        (86482.0, "routing took 1 day, 0:01:12 to run"),
    ],
)
def test_log_runtime_logs_elapsed_time(
    caplog, monkeypatch, end_time, expected
):
    """Test that log_runtime logs elapsed time with the formatted duration"""

    times = iter([10.0, end_time])
    monkeypatch.setattr(monitoring.time, "monotonic", lambda: next(times))

    with (
        caplog.at_level(logging.INFO, logger=monitoring.logger.name),
        monitoring.log_runtime("routing"),
    ):
        pass

    assert caplog.messages == [expected]


def test_log_runtime_uses_requested_log_level(caplog, monkeypatch):
    """Test that log_runtime emits at the requested logging level"""

    times = iter([10.0, 11.0])
    monkeypatch.setattr(monitoring.time, "monotonic", lambda: next(times))

    with (
        caplog.at_level(logging.DEBUG, logger=monitoring.logger.name),
        monitoring.log_runtime("routing", log_level=logging.DEBUG),
    ):
        pass

    assert [(record.levelno, record.message) for record in caplog.records] == [
        (logging.DEBUG, "routing took 0:00:01 to run"),
    ]


def test_log_runtime_logs_when_block_raises(caplog, monkeypatch):
    """Test that log_runtime still logs elapsed time if the block raises"""

    times = iter([10.0, 11.0])
    monkeypatch.setattr(monitoring.time, "monotonic", lambda: next(times))

    msg = "A test error"
    with (
        caplog.at_level(logging.INFO, logger=monitoring.logger.name),
        pytest.raises(RuntimeError, match=msg),
        monitoring.log_runtime("routing"),
    ):
        raise RuntimeError(msg)

    assert caplog.messages == ["routing took 0:00:01 to run"]


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


def test_dask_performance_report_uses_unique_filename(monkeypatch, tmp_path):
    """Test that dask_performance_report appends a UUID to filenames"""

    called = {}

    class FakeUuid:
        hex = "12345678123456781234567812345678"

    def fake_performance_report(*, filename):
        called["filename"] = filename
        return "report-context"

    monkeypatch.setattr(monitoring, "uuid4", FakeUuid)
    monkeypatch.setattr(
        monitoring,
        "performance_report",
        fake_performance_report,
    )

    context = monitoring.dask_performance_report("run", out_dir=tmp_path)

    assert context == "report-context"
    assert called["filename"] == (
        tmp_path / "dask-report_run_12345678123456781234567812345678.html"
    )


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
