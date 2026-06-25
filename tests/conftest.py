from pathlib import Path

import pytest

from quantlab.config import load_frozen_params
from quantlab.storage.duckdb_conn import connect

FROZEN = Path(__file__).resolve().parents[1] / "config" / "frozen_params.toml"


@pytest.fixture
def frozen_params():
    return load_frozen_params(FROZEN)


@pytest.fixture
def mem_db():
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()