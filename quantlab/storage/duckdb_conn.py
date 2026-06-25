"""Acceso a DuckDB — chokepoint del Data Storage Layer.

Contrato de almacenamiento ABSTRACTO por diseño (Blueprint §5/§19): este módulo
es el único punto de acceso, de modo que el motor físico (DuckDB hoy) pueda
cambiarse sin tocar a los consumidores. P1.0 solo establece la conexión.
"""

from __future__ import annotations

from pathlib import Path

import duckdb


def connect(db_path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """Abre una conexión DuckDB. Sin ruta -> base en memoria (tests)."""
    if db_path is None:
        return duckdb.connect(":memory:")
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def ping(conn: duckdb.DuckDBPyConnection) -> bool:
    """Health-check mínimo: SELECT 1."""
    return conn.execute("SELECT 1").fetchone()[0] == 1
