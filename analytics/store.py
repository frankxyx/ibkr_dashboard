"""SQLite persistence for daily bars and IBKR's volatility series.

On disk rather than in memory because IBKR rate-limits historical data to roughly
60 requests per 10 minutes. A 3-year backfill across the watchlist plus five
benchmark ETFs is most of that budget, so losing it on every restart would make
the module unusable in practice.
"""

import logging
import os
import sqlite3
import threading
from datetime import date

logger = logging.getLogger(__name__)

DB_PATH = os.getenv(
    "ANALYTICS_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "market_data.db"),
)

# One connection guarded by a lock. FastAPI runs these handlers on one event loop,
# but backfill runs as a background task and could otherwise interleave a write
# with a read mid-transaction.
_lock = threading.Lock()
_conn = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol TEXT NOT NULL,
    day    TEXT NOT NULL,          -- ISO YYYY-MM-DD
    open   REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (symbol, day)
);
CREATE INDEX IF NOT EXISTS idx_bars_symbol_day ON bars(symbol, day);

-- IBKR's own implied and historical volatility series, pulled from
-- reqHistoricalData with whatToShow OPTION_IMPLIED_VOLATILITY / HISTORICAL_VOLATILITY
-- on the UNDERLYING. Cheap source for the variance risk premium: no option chain,
-- no OPRA entitlement, same code path as price bars.
CREATE TABLE IF NOT EXISTS vol_series (
    symbol TEXT NOT NULL,
    day    TEXT NOT NULL,
    kind   TEXT NOT NULL,          -- 'IV' | 'HV'
    value  REAL,
    PRIMARY KEY (symbol, day, kind)
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def connect():
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(SCHEMA)
        _conn.commit()
        logger.info(f"Analytics store ready at {DB_PATH}")
    return _conn


def save_bars(symbol, bars):
    """Upserts daily bars. `bars` are ib_async BarData (or anything with those attrs)."""
    if not bars:
        return 0
    rows = [
        (symbol, str(b.date), float(b.open), float(b.high),
         float(b.low), float(b.close), float(b.volume or 0))
        for b in bars
    ]
    with _lock:
        conn = connect()
        conn.executemany(
            "INSERT INTO bars (symbol, day, open, high, low, close, volume) "
            "VALUES (?,?,?,?,?,?,?) ON CONFLICT(symbol, day) DO UPDATE SET "
            "open=excluded.open, high=excluded.high, low=excluded.low, "
            "close=excluded.close, volume=excluded.volume",
            rows,
        )
        conn.commit()
    return len(rows)


def save_vol(symbol, bars, kind):
    """Stores an IV or HV series. IBKR returns these as decimals (0.35 = 35%)."""
    if not bars:
        return 0
    rows = [(symbol, str(b.date), kind, float(b.close)) for b in bars]
    with _lock:
        conn = connect()
        conn.executemany(
            "INSERT INTO vol_series (symbol, day, kind, value) VALUES (?,?,?,?) "
            "ON CONFLICT(symbol, day, kind) DO UPDATE SET value=excluded.value",
            rows,
        )
        conn.commit()
    return len(rows)


def load_bars(symbol, limit=None):
    """Oldest-first bars for one symbol."""
    with _lock:
        conn = connect()
        if limit:
            # Newest N, then flip — ORDER BY day DESC LIMIT is the only way to
            # take the TAIL of the series without reading all of it.
            rows = conn.execute(
                "SELECT * FROM (SELECT * FROM bars WHERE symbol=? ORDER BY day DESC LIMIT ?) "
                "ORDER BY day ASC", (symbol, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM bars WHERE symbol=? ORDER BY day ASC", (symbol,)
            ).fetchall()
    return [dict(r) for r in rows]


def load_closes(symbol, limit=None):
    """(days, closes) — the input to almost every metric."""
    rows = load_bars(symbol, limit)
    return [r["day"] for r in rows], [r["close"] for r in rows]


def load_vol(symbol, kind, limit=None):
    with _lock:
        conn = connect()
        if limit:
            rows = conn.execute(
                "SELECT * FROM (SELECT day, value FROM vol_series WHERE symbol=? AND kind=? "
                "ORDER BY day DESC LIMIT ?) ORDER BY day ASC", (symbol, kind, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT day, value FROM vol_series WHERE symbol=? AND kind=? ORDER BY day ASC",
                (symbol, kind),
            ).fetchall()
    return [r["day"] for r in rows], [r["value"] for r in rows]


def stored_symbols():
    with _lock:
        conn = connect()
        rows = conn.execute("SELECT DISTINCT symbol FROM bars ORDER BY symbol").fetchall()
    return [r["symbol"] for r in rows]


def bar_count(symbol):
    with _lock:
        conn = connect()
        row = conn.execute("SELECT COUNT(*) AS n FROM bars WHERE symbol=?", (symbol,)).fetchone()
    return row["n"] if row else 0


def last_bar_day(symbol):
    """ISO date of the newest stored bar, or None."""
    with _lock:
        conn = connect()
        row = conn.execute("SELECT MAX(day) AS d FROM bars WHERE symbol=?", (symbol,)).fetchone()
    return row["d"] if row else None


def last_vol_day(symbol, kind):
    """ISO date of the newest stored IV/HV point, or None."""
    with _lock:
        conn = connect()
        row = conn.execute(
            "SELECT MAX(day) AS d FROM vol_series WHERE symbol=? AND kind=?", (symbol, kind)
        ).fetchone()
    return row["d"] if row else None


def coverage():
    """Per-symbol bar count and date range — what the UI needs to say 'no history yet'."""
    with _lock:
        conn = connect()
        rows = conn.execute(
            "SELECT symbol, COUNT(*) AS bars, MIN(day) AS first_day, MAX(day) AS last_day "
            "FROM bars GROUP BY symbol ORDER BY symbol"
        ).fetchall()
    return {r["symbol"]: dict(r) for r in rows}


def set_meta(key, value):
    with _lock:
        conn = connect()
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)),
        )
        conn.commit()


def get_meta(key, default=None):
    with _lock:
        conn = connect()
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default
