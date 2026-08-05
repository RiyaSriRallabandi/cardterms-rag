"""Structured JSON logging with run-scoped context."""

import logging
import sys
import uuid

import structlog


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    processors = [
        structlog.contextvars.merge_contextvars,  # auto-attach bound context
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level)
        ),
        cache_logger_on_first_use=True,
    )


def new_run_id() -> str:
    return str(uuid.uuid4())


def bind_run(run_id: str, **kwargs) -> None:
    """Attach run_id (and anything else) to every subsequent log line."""
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(run_id=run_id, **kwargs)


log = structlog.get_logger()
