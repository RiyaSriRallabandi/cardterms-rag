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


def healthcheck() -> bool:
    with get_conn() as conn:
        row = conn.execute("SELECT 1 AS ok").fetchone()
        return row["ok"] == 1
