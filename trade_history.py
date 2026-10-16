"""SQLite persistence for executed fills, and the order history derived from them.

On disk because IBKR's execution API only serves the *current* trading day:
reqExecutions returns nothing from yesterday, so an in-memory history would reset
to empty every session and the realized PnL of anything older would be gone for
good. Every fill IBKR reports is written here once, keyed by its execution id, and
the history endpoint reads from this table rather than from the broker.

Realized PnL is IBKR's own number, taken from the commission report that follows
each execution. It is reported only on closing fills and is already net of
commissions — we never recompute it.
"""

import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

DB_PATH = os.getenv(
    "TRADE_HISTORY_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_history.db"),
)

# Same single-connection-plus-lock arrangement as analytics/store.py: the API
# handlers share an event loop, but the fill sync runs as a background task.
_lock = threading.Lock()
_conn = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS fills (
    exec_id        TEXT PRIMARY KEY,   -- IBKR's execution id: unique, stable, so a
                                       -- replayed execDetails is an idempotent write
    perm_id        INTEGER,            -- broker-side order id, survives reconnects
    order_id       INTEGER,            -- client-side id, recycled across sessions
    account        TEXT,
    symbol         TEXT NOT NULL,
    currency       TEXT,               -- contract currency
    side           TEXT,               -- IBKR's BOT / SLD
    shares         REAL,
    price          REAL,
    time           TEXT,               -- ISO 8601, UTC
    commission     REAL,               -- NULL until the commission report arrives
    commission_ccy TEXT,
    realized_pnl   REAL,               -- NULL on opening fills; IBKR reports none
    sec_type       TEXT                -- STK, CASH (an FX conversion), ...
);
CREATE INDEX IF NOT EXISTS idx_fills_time ON fills(time DESC);
CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills(symbol);
"""

# IBKR sends unset doubles as DBL_MAX. ib_async normalises realizedPNL but not
# commission, so anything this large is "not reported", not a real number.
UNSET_THRESHOLD = 1e300


# Columns added after the first release. CREATE TABLE IF NOT EXISTS will not add a
# column to a table that already exists, so each one is applied on open instead.
MIGRATIONS = [
    ("sec_type", "ALTER TABLE fills ADD COLUMN sec_type TEXT"),
]


def connect():
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(SCHEMA)
        have = {r["name"] for r in _conn.execute("PRAGMA table_info(fills)")}
        for column, ddl in MIGRATIONS:
            if column not in have:
                _conn.execute(ddl)
                logger.info(f"Trade history: added column {column}")
        _conn.commit()
        logger.info(f"Trade history store ready at {DB_PATH}")
    return _conn


def _num(v):
    """A float, or None when IBKR did not actually report a value."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (f != f or abs(f) > UNSET_THRESHOLD) else f


def fill_row(fill):
    """An ib_async Fill flattened into the columns above."""
    ex, con, rep = fill.execution, fill.contract, fill.commissionReport

    # rep.execId is set only once a real commission report has merged into the Fill.
    # Without it the money fields are absent, not zero — 0.00 would read as break-even.
    reported = bool(rep and rep.execId)
    # Magnitude, not sign: the live API reports commission positive and Flex reports
    # the same charge negative, so both are stored as a positive cost.
    commission = abs(_num(rep.commission)) if reported and _num(rep.commission) is not None else None
    realized = _num(rep.realizedPNL) if reported else None

    # ib_async rewrites an unset realizedPNL to 0.0. A close is net of commission, so
    # an exact 0.00 is effectively impossible — treat it as an opening fill instead.
    if realized == 0.0:
        realized = None

    when = ex.time or fill.time
    return {
        "exec_id": ex.execId,
        "perm_id": ex.permId or 0,
        "order_id": ex.orderId or 0,
        "account": ex.acctNumber or "",
        "symbol": con.symbol,
        "currency": con.currency or "USD",
        "side": ex.side or "",
        "shares": _num(ex.shares) or 0.0,
        "price": _num(ex.price) or 0.0,
        "time": when.isoformat() if hasattr(when, "isoformat") else str(when),
        "commission": commission,
        "commission_ccy": (rep.currency or None) if reported else None,
        "realized_pnl": realized,
        "sec_type": con.secType or "",
    }


def record_fills(fills):
    """Upserts ib_async Fill objects. Returns the number of rows written."""
    rows = []
    for f in fills:
        try:
            r = fill_row(f)
        except Exception as e:
            logger.warning(f"Skipping unparseable fill: {e}")
            continue
        if r["exec_id"]:
            rows.append(r)
    return record_rows(rows)


def record_rows(rows):
    """Upserts already-flattened rows — the shared path for live fills and Flex.

    The commission report arrives on a separate callback some time after the
    execution, so a fill is normally written twice: once bare, once complete.
    The money columns therefore only overwrite when the incoming row actually
    has a value — otherwise a re-sync of an old fill would erase its realized
    PnL, and a Flex import would erase what the live feed already recorded.
    """
    rows = [r for r in rows if r and r.get("exec_id")]
    if not rows:
        return 0

    with _lock:
        conn = connect()
        conn.executemany(
            """
            INSERT INTO fills (exec_id, perm_id, order_id, account, symbol, currency,
                               side, shares, price, time, commission, commission_ccy,
                               realized_pnl, sec_type)
            VALUES (:exec_id, :perm_id, :order_id, :account, :symbol, :currency,
                    :side, :shares, :price, :time, :commission, :commission_ccy,
                    :realized_pnl, :sec_type)
            ON CONFLICT(exec_id) DO UPDATE SET
                perm_id        = excluded.perm_id,
                order_id       = excluded.order_id,
                account        = excluded.account,
                sec_type       = COALESCE(NULLIF(excluded.sec_type, ''), fills.sec_type),
                shares         = excluded.shares,
                price          = excluded.price,
                commission     = COALESCE(excluded.commission, fills.commission),
                commission_ccy = COALESCE(excluded.commission_ccy, fills.commission_ccy),
                realized_pnl   = COALESCE(excluded.realized_pnl, fills.realized_pnl)
            """,
            rows,
        )
        conn.commit()
    return len(rows)


# permId is the broker's stable order id; orderId is a client-side counter that
# restarts at 1 and collides across sessions. Group on permId wherever one exists.
_GROUP_KEY = (
    "CASE WHEN COALESCE(perm_id, 0) != 0 THEN 'p' || perm_id ELSE 'o' || COALESCE(order_id, 0) END"
)


def normalize_bound(value, end_of_day=False):
    """A date/datetime bound rendered exactly the way `time` values are stored.

    Times go into the table as datetime.isoformat() of a UTC datetime, so a bound
    put through the same pipeline compares correctly with a plain string `>=`.
    Comparing raw input instead would quietly misbehave: '...T13:33:03Z' and
    '...T13:33:03+00:00' are the same instant but not the same string.

    Returns None for empty input, and raises ValueError on something unparseable
    so the caller can reject it rather than silently returning the whole history.
    """
    if value in (None, ""):
        return None
    if isinstance(value, str):
        text = value.strip()
        # A bare date means the whole of that day, not midnight at its start.
        if len(text) == 10:
            when = datetime.fromisoformat(text)
            when = when.replace(hour=23, minute=59, second=59, microsecond=999999) if end_of_day else when
        else:
            when = datetime.fromisoformat(text)
    else:
        when = value

    # A bound with no zone is read as UTC, matching how fills are stored.
    when = when.replace(tzinfo=timezone.utc) if when.tzinfo is None else when.astimezone(timezone.utc)
    return when.isoformat()


def _filters(account=None, symbol=None, start=None, end=None):
    where, params = [], []
    if account:
        where.append("account = ?")
        params.append(account)
    if symbol:
        where.append("symbol = ?")
        params.append(symbol.upper())
    # Inclusive both ends: the bounds are normalised above, and `end` from a bare
    # date already carries that day's last microsecond.
    if start:
        where.append("time >= ?")
        params.append(start)
    if end:
        where.append("time <= ?")
        params.append(end)
    return ("WHERE " + " AND ".join(where)) if where else "", params


def order_history(account=None, symbol=None, start=None, end=None, limit=200):
    """Executed orders, newest first — one row per order, its fills rolled up."""
    clause, params = _filters(account, symbol, start, end)
    with _lock:
        conn = connect()
        orders = conn.execute(
            f"""
            SELECT {_GROUP_KEY}                         AS group_key,
                   MAX(perm_id)                         AS permId,
                   MAX(order_id)                        AS orderId,
                   symbol, side, account,
                   MAX(currency)                        AS currency,
                   MAX(commission_ccy)                  AS commissionCurrency,
                   MAX(sec_type)                        AS secType,
                   SUM(shares)                          AS shares,
                   SUM(shares * price) / NULLIF(SUM(shares), 0) AS avgPrice,
                   MIN(time)                            AS firstFill,
                   MAX(time)                            AS lastFill,
                   COUNT(*)                             AS fillCount,
                   SUM(commission)                      AS commission,
                   SUM(realized_pnl)                    AS realizedPnL
            FROM fills
            {clause}
            GROUP BY group_key, symbol, side
            ORDER BY lastFill DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()

        # Per-fill detail for the rows just returned, so a partially filled order
        # can be expanded without a second round trip.
        keys = [r["group_key"] for r in orders]
        detail = {}
        if keys:
            placeholders = ",".join("?" * len(keys))
            for r in conn.execute(
                f"SELECT {_GROUP_KEY} AS group_key, exec_id, shares, price, time, "
                f"commission, realized_pnl FROM fills "
                f"WHERE {_GROUP_KEY} IN ({placeholders}) ORDER BY time ASC",
                keys,
            ).fetchall():
                detail.setdefault(r["group_key"], []).append({
                    "execId": r["exec_id"],
                    "shares": r["shares"],
                    "price": r["price"],
                    "time": r["time"],
                    "commission": r["commission"],
                    "realizedPnL": r["realized_pnl"],
                })

    res = []
    for r in orders:
        row = dict(r)
        row["fills"] = detail.get(row.pop("group_key"), [])
        # BOT/SLD is IBKR's wire vocabulary; the rest of the dashboard says BUY/SELL.
        row["action"] = {"BOT": "BUY", "SLD": "SELL"}.get(row["side"], row["side"])
        res.append(row)
    return res


def totals(account=None, symbol=None, start=None, end=None):
    """Realized PnL and commissions across the whole filtered history.

    Deliberately not computed from the capped order list: the summary has to cover
    every fill in the selected period, not just the page being shown.
    """
    clause, params = _filters(account, symbol, start, end)
    with _lock:
        conn = connect()
        # Per order, not per fill: one exit split into three partials is one closed
        # trade, and counting it as three would distort the hit rate.
        # Split by currency as well: IBKR reports realized PnL in the currency the
        # trade settled in, so adding a GBP close to a USD one produces a number
        # that is not money in any currency. No FX rate is available here to
        # convert with, so the split is reported rather than papered over.
        rows = conn.execute(
            f"""
            SELECT ccy                                   AS currency,
                   SUM(n)                                AS fills,
                   COUNT(*)                              AS orders,
                   SUM(pnl)                              AS realizedPnL,
                   SUM(comm)                             AS commission,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) AS losses
            FROM (
                SELECT COUNT(*)          AS n,
                       SUM(realized_pnl) AS pnl,
                       SUM(commission)   AS comm,
                       -- Not every Flex query includes a commission-currency column;
                       -- the contract currency is the right fallback.
                       COALESCE(MAX(commission_ccy), MAX(currency)) AS ccy
                FROM fills
                {clause}
                GROUP BY {_GROUP_KEY}, symbol, side
            )
            GROUP BY ccy
            ORDER BY ABS(COALESCE(SUM(pnl), 0)) DESC
            """,
            params,
        ).fetchall()

    per_currency = [dict(r) for r in rows if r["currency"] or r["fills"]]
    if not per_currency:
        return {"fills": 0, "orders": 0, "realizedPnL": None, "commission": None,
                "wins": 0, "losses": 0, "currency": None, "byCurrency": []}

    # Counts are currency-independent, so they still roll up.
    out = {
        "fills": sum(r["fills"] or 0 for r in per_currency),
        "orders": sum(r["orders"] or 0 for r in per_currency),
        "wins": sum(r["wins"] or 0 for r in per_currency),
        "losses": sum(r["losses"] or 0 for r in per_currency),
        "byCurrency": per_currency,
        "mixedCurrency": len(per_currency) > 1,
    }
    # A single-currency total is a real number; a mixed one is deliberately null so
    # nothing downstream can render a meaningless sum.
    if len(per_currency) == 1:
        only = per_currency[0]
        out.update(currency=only["currency"], realizedPnL=only["realizedPnL"],
                   commission=only["commission"])
    else:
        out.update(currency=None, realizedPnL=None, commission=None)
    return out


def coverage(account=None, symbol=None):
    """The span the store actually holds, ignoring any selected date range.

    Lets the UI distinguish "you executed nothing last month" from "this dashboard
    was not recording last month" — the two look identical in an empty table, and
    only the second one is fixable (by importing a Flex report).
    """
    clause, params = _filters(account, symbol)
    with _lock:
        conn = connect()
        row = conn.execute(
            f"SELECT MIN(time) AS firstFill, MAX(time) AS lastFill, COUNT(*) AS fills "
            f"FROM fills {clause}",
            params,
        ).fetchone()
    return dict(row) if row else {}


# How far apart two records of the same trade can sit and still be the same trade.
# Flex reports to the second but a Flex timestamp is exchange-local while a live
# fill is stamped in UTC, so the two can disagree by more than rounding.
DUPLICATE_WINDOW = timedelta(minutes=5)


def find_duplicate(symbol, side, shares, when, account=None):
    """exec_id of an existing fill that looks like the same trade, or None.

    Only for rows whose source gave no execution id — with one, matching is exact
    and this guesswork is neither needed nor wanted.
    """
    try:
        at = datetime.fromisoformat(when)
    except (TypeError, ValueError):
        return None

    params = [symbol, side, float(shares),
              (at - DUPLICATE_WINDOW).isoformat(), (at + DUPLICATE_WINDOW).isoformat()]
    clause = ""
    if account:
        clause = " AND account = ?"
        params.append(account)

    with _lock:
        conn = connect()
        row = conn.execute(
            "SELECT exec_id FROM fills WHERE symbol = ? AND side = ? AND shares = ? "
            "AND time BETWEEN ? AND ?" + clause + " LIMIT 1",
            params,
        ).fetchone()
    return row["exec_id"] if row else None
