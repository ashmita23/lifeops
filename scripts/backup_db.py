"""Dumps the SQLite database to a timestamped, restorable SQL file.

Run manually (`python scripts/backup_db.py`) or on a schedule (e.g. a
Railway cron job service running this against the same volume) - see the
"Persistence on Railway" section in README.md.
"""

import datetime
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings

DEFAULT_BACKUP_DIR = Path(__file__).resolve().parent.parent / "backups"


def backup_db(db_path: str | None = None, backup_dir: Path = DEFAULT_BACKUP_DIR) -> Path:
    db_path = db_path or settings.database_path
    if not Path(db_path).exists():
        raise FileNotFoundError(f"No database file at {db_path}")

    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dest = backup_dir / f"lifeops-backup-{timestamp}.sql"

    connection = sqlite3.connect(db_path)
    try:
        with dest.open("w") as f:
            for line in connection.iterdump():
                f.write(f"{line}\n")
    finally:
        connection.close()

    return dest


if __name__ == "__main__":
    path = backup_db()
    print(f"Backed up {settings.database_path} -> {path}")
