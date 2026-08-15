"""Database connection helpers."""

from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from cardterms.config import Settings

_settings = Settings()


@contextmanager
def get_conn():
    """Yield a connection; commit on success, roll back on error."""
    with psycopg.connect(_settings.database_url, row_factory=dict_row) as conn:
        yield conn


def connect():
    """Open a connection the caller owns and must close.

    The context manager above suits scripts, which run once and exit. A service
    holds one connection for its lifetime; entering the context manager and
    keeping only the connection would leave the manager unreferenced, and
    closing it on garbage collection would close the connection underneath the
    service. Autocommit because the service only reads.
    """
    return psycopg.connect(
        _settings.database_url, row_factory=dict_row, autocommit=True
    )


def healthcheck() -> bool:
    with get_conn() as conn:
        row = conn.execute("SELECT 1 AS ok").fetchone()
        return row["ok"] == 1
