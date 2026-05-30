import duckdb
from config.settings import DB_PATH

_conn: duckdb.DuckDBPyConnection | None = None


def get_conn() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = duckdb.connect(str(DB_PATH))
    return _conn


def query(sql: str, params: list = None):
    """Execute SQL and return a DataFrame."""
    conn = get_conn()
    if params:
        return conn.execute(sql, params).df()
    return conn.execute(sql).df()


def execute(sql: str, params: list = None):
    """Execute SQL with no return value."""
    conn = get_conn()
    if params:
        conn.execute(sql, params)
    else:
        conn.execute(sql)


def close():
    global _conn
    if _conn:
        _conn.close()
        _conn = None
