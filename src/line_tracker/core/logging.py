"""Project logging helpers."""

from __future__ import annotations

import logging


class _ContextAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("run_id", self.extra.get("run_id", "-"))
        extra.setdefault("source", self.extra.get("source", "-"))
        return msg, kwargs


def get_logger(name: str, *, run_id: str | None = None, source: str | None = None):
    """Return configured logger adapter with run/source context.

    Prevents duplicate handlers for repeated calls.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not any(getattr(h, "_line_tracker_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler._line_tracker_handler = True  # type: ignore[attr-defined]
        fmt = (
            "%(asctime)s %(levelname)s [%(name)s]"
            " [run_id=%(run_id)s source=%(source)s]"
            " %(message)s"
        )
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return _ContextAdapter(logger, {"run_id": run_id or "-", "source": source or "-"})
