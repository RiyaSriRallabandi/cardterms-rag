"""Apply SQL migrations in order."""

from pathlib import Path

from cardterms.db import get_conn
from cardterms.logging import configure_logging, log

MIGRATIONS = Path(__file__).parent.parent / "db"


def main() -> None:
    configure_logging(json_output=False)
    files = sorted(MIGRATIONS.glob("*.sql"))
    with get_conn() as conn:
        for f in files:
            log.info("applying_migration", file=f.name)
            conn.execute(f.read_text())
        conn.commit()
    log.info("migrations_complete", count=len(files))


if __name__ == "__main__":
    main()
