"""Tests for core logging helper."""

from line_tracker.core.logging import get_logger


def test_get_logger_does_not_duplicate_handlers():
    logger1 = get_logger("line_tracker.tests.logging", run_id="r1", source="s1")
    count1 = len(logger1.logger.handlers)
    logger2 = get_logger("line_tracker.tests.logging", run_id="r2", source="s2")
    count2 = len(logger2.logger.handlers)

    assert count1 == count2
    assert count2 == 1


def test_get_logger_includes_context_fields():
    logger = get_logger("line_tracker.tests.logging_ctx", run_id="run-123", source="svc")
    assert logger.extra["run_id"] == "run-123"
    assert logger.extra["source"] == "svc"
