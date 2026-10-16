"""SQLite warehouse builder.

The pipeline keeps its transformations in ``sql/*.sql`` and executes them in
filename order. That gives two things a notebook-only project cannot offer: a
reviewable transformation history, and a warehouse that can be rebuilt from
scratch in one command.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from . import config as cfg


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(cfg.DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def execute_sql_file(connection: sqlite3.Connection, path: Path) -> None:
    """Execute a .sql file statement by statement.

    ``sqlite3.executescript`` is avoided deliberately: it wraps the whole file
    in an implicit transaction and swallows partial failures, which makes
    debugging a broken transformation much harder.
    """
    script = path.read_text(encoding="utf-8")
    statements = [s.strip() for s in script.split(";") if s.strip()]
    for statement in statements:
        cleaned = "\n".join(line for line in statement.splitlines() if not line.strip().startswith("--")).strip()
        if not cleaned:
            continue
        try:
            connection.execute(cleaned)
        except sqlite3.Error as error:  # pragma: no cover - surfaced to the caller
            raise RuntimeError(f"{path.name}: {error}\n\n{cleaned[:500]}") from error


def build_warehouse(loans: pd.DataFrame, macro: pd.DataFrame, verbose: bool = True) -> None:
    """Load the raw ingestion output, then run every SQL script in order."""
    if cfg.DB_PATH.exists():
        cfg.DB_PATH.unlink()

    connection = get_connection()
    try:
        loans.to_sql("raw_loans", connection, if_exists="replace", index=False)
        macro.to_sql("raw_macro", connection, if_exists="replace", index=False)

        sql_files = sorted(cfg.SQL_DIR.glob("*.sql"))
        if not sql_files:
            raise RuntimeError(f"No SQL scripts found in {cfg.SQL_DIR}")
        for path in sql_files:
            execute_sql_file(connection, path)
            if verbose:
                print(f"  [sql] applied {path.name}")
        connection.commit()
    finally:
        connection.close()


def read_table(name: str) -> pd.DataFrame:
    """Read a warehouse table into a DataFrame."""
    with get_connection() as connection:
        return pd.read_sql_query(f"SELECT * FROM {name}", connection)


def list_tables() -> list[str]:
    query = "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"
    with get_connection() as connection:
        return [row[0] for row in connection.execute(query).fetchall()]


def table_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    with get_connection() as connection:
        for name in list_tables():
            if name.startswith("sqlite_"):
                continue
            counts[name] = connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    return counts
