from data.db.client import get_conn


def init():
    conn = get_conn()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            ts       TIMESTAMP NOT NULL,
            symbol   VARCHAR   NOT NULL,
            interval VARCHAR   NOT NULL,
            open     DOUBLE    NOT NULL,
            high     DOUBLE    NOT NULL,
            low      DOUBLE    NOT NULL,
            close    DOUBLE    NOT NULL,
            volume   DOUBLE    NOT NULL,
            PRIMARY KEY (ts, symbol, interval)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS symbols (
            symbol      VARCHAR PRIMARY KEY,
            name        VARCHAR,
            asset_class VARCHAR,
            currency    VARCHAR,
            active      BOOLEAN   DEFAULT TRUE,
            added_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Tracks what date ranges have been fetched per symbol+interval
    # so incremental ingestion only pulls what's missing
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            symbol        VARCHAR   NOT NULL,
            interval      VARCHAR   NOT NULL,
            fetched_from  TIMESTAMP NOT NULL,
            fetched_to    TIMESTAMP NOT NULL,
            rows_inserted INTEGER,
            fetched_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    print("DB schema initialised.")
