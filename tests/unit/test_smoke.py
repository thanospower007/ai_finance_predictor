import quantlab
from quantlab.storage.duckdb_conn import ping


def test_package_imports_with_version():
    assert quantlab.__version__


def test_duckdb_connects(mem_db):
    assert ping(mem_db)
