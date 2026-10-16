"""Backfills trade history from an IBKR Flex Report.

The live API (reqExecutions) only serves the CURRENT trading day, so everything
executed before the dashboard started recording is unreachable through it. The
Flex Web Service is the only source IBKR offers for older executions, and it
carries the realized PnL per trade (fifoPnlRealized) that this dashboard wants.

Rows land in the same store as live fills and are deduped on IBKR's own execution
id, so importing a report that overlaps today cannot double-count a trade.

To get a token: IBKR web portal -> Settings -> Configure Flex Web Service ->
generate token. Then create a Trade Confirmation or Activity flex query that
includes the Trades/TradeConfirm section, and note its query id.
"""

import hashlib
import logging
from datetime import datetime, timedelta, timezone

import trade_history

logger = logging.getLogger(__name__)

# TradeConfirm comes from a Trade Confirmation query, Trade from an Activity one.
# Accept either, rather than forcing the user to build one specific query.
TRADE_TOPICS = ("TradeConfirm", "Trade")

# The Trades section has an Options row — Symbol Summary, Asset Class, Order,
# Execution, Closed Lots — and each enabled level emits its own <Trade> rows for the
# SAME trades. Ingesting a mix would count every trade once per level, so exactly one
# level is used. Execution first: it is the only one carrying an IB Execution ID.
LEVEL_PRIORITY = ("EXECUTION", "ORDER")

# Flex renders timestamps differently depending on how the query is configured,
# and the date/time separator is a query option the user controls.
_TIME_FORMATS = (
    "%Y%m%d;%H%M%S", "%Y%m%d;%H:%M:%S", "%Y-%m-%d;%H:%M:%S",
    "%Y%m%d %H%M%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d, %H:%M:%S",
    "%Y%m%d%H%M%S",  "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M",
    "%d-%b-%y;%H:%M:%S", "%m/%d/%Y;%H:%M:%S", "%m/%d/%y;%H:%M:%S",
    "%m/%d/%Y %H:%M:%S", "%d-%b-%y %H:%M:%S",
    "%Y%m%d", "%Y-%m-%d", "%d-%b-%y", "%m/%d/%Y", "%m/%d/%y",
)

# Sanity bound for the digits-only fallback below: enough to reject a date that
# parsed under the wrong field order (a day-first 10042026 becomes year 1004).
_MIN_YEAR, _MAX_YEAR = 1990, 2100

# Slash/dash dates are ordered by a per-query Flex setting, so the convention is
# detected per report (see detect_date_order) rather than assumed.
_DMY_FORMATS = ("%d/%m/%Y;%H:%M:%S", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y;%H%M%S",
                "%d/%m/%Y", "%d/%m/%y;%H:%M:%S", "%d/%m/%y")
_MDY_FORMATS = ("%m/%d/%Y;%H:%M:%S", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y;%H%M%S",
                "%m/%d/%Y", "%m/%d/%y;%H:%M:%S", "%m/%d/%y")

# Flex stamps exchange-local time and names the zone. Mapping it beats assuming UTC:
# a 20:00 EDT fill is the NEXT day in UTC, and would land in the wrong day's filter.
_ZONE_OFFSETS = {
    "UTC": 0, "GMT": 0, "Z": 0,
    "EST": -5, "EDT": -4, "CST": -6, "CDT": -5,
    "MST": -7, "MDT": -6, "PST": -8, "PDT": -7,
    "AKST": -9, "AKDT": -8, "HST": -10,
    "BST": 1, "WET": 0, "WEST": 1, "CET": 1, "CEST": 2, "EET": 2, "EEST": 3,
    "JST": 9, "KST": 9, "HKT": 8, "SGT": 8,
    "AEST": 10, "AEDT": 11, "NZST": 12, "NZDT": 13,
}
_unknown_zones = set()


def detect_date_order(items):
    """'DMY', 'MDY', or None — inferred from the report's own unambiguous dates.

    A slash date is only self-describing when one component exceeds 12. Reading the
    whole report first means a run containing 17/09 settles the convention for the
    ambiguous 07/08 rows too, instead of each row being guessed independently.
    """
    dmy = mdy = 0
    for it in items:
        text = raw_timestamp(it)
        if not text:
            continue
        head = text.split(";")[0].split(" ")[0]
        parts = head.replace("-", "/").split("/")
        if len(parts) != 3 or not all(p.isdigit() for p in parts[:2]):
            continue
        a, b = int(parts[0]), int(parts[1])
        if a > 12 >= b:
            dmy += 1
        elif b > 12 >= a:
            mdy += 1
    if dmy and mdy:
        logger.warning(f"Flex dates are inconsistent ({dmy} day-first, {mdy} month-first)")
        return None
    return "DMY" if dmy else ("MDY" if mdy else None)


def _split_zone(text):
    """(timestamp without its trailing zone name, tzinfo or None)."""
    parts = text.rsplit(" ", 1)
    if len(parts) == 2 and parts[1].isalpha():
        name = parts[1].upper()
        if name in _ZONE_OFFSETS:
            return parts[0].strip(), timezone(timedelta(hours=_ZONE_OFFSETS[name]))
        # Unknown zone: drop the label and fall back to UTC rather than reject the
        # row outright, but say so once so it can be added.
        if name not in _unknown_zones:
            _unknown_zones.add(name)
            logger.warning(f"Unknown Flex timezone {name!r} — treating those times as UTC")
        return parts[0].strip(), None
    return text, None


def _get(obj, *names):
    """First present, non-empty attribute among `names`."""
    for n in names:
        v = getattr(obj, n, None)
        if v not in (None, "", "-"):
            return v
    return None


def _as_text(v):
    """parseNumbers turns an all-digit timestamp into a number — undo that."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def raw_timestamp(row):
    """The timestamp text Flex gave us, before any parsing."""
    raw = _get(row, "dateTime", "tradeDate", "reportDate", "settleDate")
    if raw is None:
        return None
    text = _as_text(raw).strip()

    # A separate tradeTime column, when the query splits date and time.
    t = _get(row, "tradeTime")
    if t is not None and not any(c in text for c in ";, ") and len(text) <= 10:
        text = f"{text};{_as_text(t)}"
    return text


def _parse_time(row, date_order=None):
    """Flex timestamps, in whichever of its many configurable shapes.

    Date format, time format, separator and timezone label are all per-query Flex
    options, so the shape varies by account rather than being fixed.
    """
    text = raw_timestamp(row)
    if not text:
        return None

    text, zone = _split_zone(text)
    # An unlabelled time is read as UTC — the same assumption live fills are stored
    # under, so the two sort together.
    zone = zone or timezone.utc

    ordered = _DMY_FORMATS if date_order == "DMY" else _MDY_FORMATS if date_order == "MDY" else ()
    for fmt in ordered + _TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=zone).astimezone(timezone.utc)
        except ValueError:
            continue

    # Last resort: keep only the digits and read them positionally. This covers
    # separator combinations the explicit list above misses.
    digits = "".join(c for c in text if c.isdigit())
    candidates = [(14, "%Y%m%d%H%M%S"), (12, "%Y%m%d%H%M"), (8, "%Y%m%d")]
    if date_order == "DMY":
        candidates = [(14, "%d%m%Y%H%M%S"), (8, "%d%m%Y")] + candidates
    for length, fmt in candidates:
        if len(digits) != length:
            continue
        try:
            when = datetime.strptime(digits, fmt)
        except ValueError:
            continue
        if _MIN_YEAR <= when.year <= _MAX_YEAR:
            return when.replace(tzinfo=zone).astimezone(timezone.utc)
    return None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def synth_id(row):
    """A deterministic id for a row whose Flex query emitted none.

    Hashed from the fields that identify a trade, so the same report imported
    twice produces the same id and updates in place rather than duplicating.
    It cannot match the id the live API uses for the same execution, though —
    that gap is what find_duplicate() covers.
    """
    key = "|".join(str(row.get(k, "")) for k in
                   ("account", "symbol", "side", "shares", "price", "time", "realized_pnl"))
    return "flexsyn:" + hashlib.sha1(key.encode()).hexdigest()[:24]


class SkipRow(Exception):
    """Why a Flex row could not be used — surfaced to the user, not swallowed."""


def to_row(item, date_order=None):
    """One Flex trade -> a trade_history fills row. Raises SkipRow if unusable."""
    # ibExecID is the SAME id the live API reports, so a trade in both sources
    # collapses to one row. tradeID is the fallback, namespaced to avoid collisions.
    exec_id = _get(item, "ibExecID", "execID")
    synthetic = False
    if not exec_id:
        trade_id = _get(item, "tradeID", "transactionID")
        if trade_id:
            exec_id = f"flex:{int(trade_id) if isinstance(trade_id, float) else trade_id}"
        else:
            # No id column at all. Derive a stable one from the row's own content so
            # re-importing the same report stays idempotent — see synth_id().
            synthetic = True

    when = _parse_time(item, date_order)
    if when is None:
        raw = raw_timestamp(item)
        raise SkipRow(
            f"unreadable timestamp {raw!r}" if raw
            else "no date field (add Date/Time to the query)"
        )

    symbol = str(_get(item, "symbol", "underlyingSymbol") or "")
    if not symbol:
        raise SkipRow("no symbol")

    qty = _num(_get(item, "quantity", "shares")) or 0.0
    side = str(_get(item, "buySell", "side") or "").upper()
    # Flex signs the quantity; buySell is the authority on direction and the
    # quantity is stored unsigned to match the live-API rows.
    if side in ("BUY", "BOT"):
        side = "BOT"
    elif side in ("SELL", "SLD"):
        side = "SLD"
    else:
        side = "SLD" if qty < 0 else "BOT"

    realized = _num(_get(item, "fifoPnlRealized", "realizedPnl", "realizedPNL"))
    # Same rule as the live path: IBKR reports a flat 0 on an opening trade, and a
    # real close is never exactly 0.00 once commission is netted.
    if realized == 0.0:
        realized = None

    # Flex signs a commission negative (a charge); the live API reports the same
    # number positive. Stored as a positive cost so the two sources can be summed.
    commission = _num(_get(item, "ibCommission", "commission"))
    if commission is not None:
        commission = abs(commission)
    perm_id = _num(_get(item, "ibOrderID", "permId"))
    order_id = _num(_get(item, "orderID", "orderId"))

    row = {
        "exec_id": str(exec_id) if exec_id else "",
        "perm_id": int(perm_id) if perm_id else 0,
        "order_id": int(order_id) if order_id else 0,
        "account": str(_get(item, "accountId", "account") or ""),
        "symbol": symbol,
        "currency": str(_get(item, "currency") or "USD"),
        "side": side,
        "shares": abs(qty),
        "price": _num(_get(item, "tradePrice", "price", "tradeMoney")),
        "time": when.isoformat(),
        "commission": commission,
        "commission_ccy": str(_get(item, "ibCommissionCurrency") or "") or None,
        "realized_pnl": realized,
        "sec_type": str(_get(item, "assetCategory", "secType") or ""),
    }
    if synthetic:
        row["exec_id"] = synth_id(row)
        row["synthetic_id"] = True
    return row


def load_report(token=None, query_id=None, path=None):
    """Downloads or opens a Flex report. Blocking — call it off the event loop."""
    from ib_async import FlexReport

    if path:
        logger.info(f"Loading Flex report from {path}")
        return FlexReport(path=path)
    logger.info(f"Downloading Flex query {query_id} from IBKR")
    return FlexReport(token=token, queryId=query_id)


def pick_level(items):
    """One level of detail out of a report that may carry several.

    Returns (rows, note) — note is set only when something was actually dropped.
    """
    levels = {}
    for it in items:
        levels.setdefault(str(getattr(it, "levelOfDetail", "") or "").upper(), []).append(it)
    if len(levels) <= 1:
        return items, None

    for want in LEVEL_PRIORITY:
        if levels.get(want):
            dropped = len(items) - len(levels[want])
            others = ", ".join(sorted(k or "(none)" for k in levels if k != want))
            return levels[want], (
                f"used {want.lower()}-level rows and ignored {dropped} aggregate "
                f"row(s) ({others}) that would have double-counted"
            )

    # No level we recognise — the largest group is the least-bad guess.
    best = max(levels, key=lambda k: len(levels[k]))
    return levels[best], f"multiple levels of detail present; used {best or 'unlabelled'}"


def import_report(report):
    """Writes every trade in the report to the store. Returns a summary."""
    topics = report.topics()
    items, used = [], []
    for topic in TRADE_TOPICS:
        if topic in topics:
            found = report.extract(topic, parseNumbers=True)
            if found:
                items.extend(found)
                used.append(f"{topic} ({len(found)})")

    if not items:
        return {
            "status": "error",
            "message": (
                "The report contains no trade rows. Add a Trades (or Trade "
                f"Confirmations) section to the Flex query. Sections present: "
                f"{', '.join(sorted(topics)) or 'none'}"
            ),
        }

    items, level_note = pick_level(items)
    if level_note:
        logger.info(f"Flex import: {level_note}")

    # Reported once at the end rather than per row: a missing column is a property
    # of the query, so all 62 rows would otherwise say the same thing 62 times.
    first = items[0]
    no_ids = not any(hasattr(first, a) for a in ("ibExecID", "execID", "tradeID", "transactionID"))
    no_price = not any(hasattr(first, a) for a in ("tradePrice", "price"))

    date_order = detect_date_order(items)
    if date_order:
        logger.info(f"Flex dates read as {'day-first' if date_order == 'DMY' else 'month-first'}")

    rows, reasons = [], {}
    for it in items:
        try:
            rows.append(to_row(it, date_order))
        except SkipRow as e:
            reasons[str(e)] = reasons.get(str(e), 0) + 1
        except Exception as e:
            reasons[f"unexpected: {e}"] = reasons.get(f"unexpected: {e}", 0) + 1

    # Rows with a synthesised id cannot be matched against live fills by id, so a
    # trade already recorded from the API would otherwise be counted twice. Drop
    # those before writing — the live record is the better of the two.
    deduped = 0
    keep = []
    for r in rows:
        if r.pop("synthetic_id", False):
            existing = trade_history.find_duplicate(
                r["symbol"], r["side"], r["shares"], r["time"], r["account"] or None
            )
            if existing and not existing.startswith("flexsyn:"):
                deduped += 1
                continue
        keep.append(r)
    rows = keep

    skipped = sum(reasons.values())
    written = trade_history.record_rows(rows)
    logger.info(f"Flex import: {written} trade(s) written, {skipped} skipped")

    # The field names a Flex query emits are chosen by the person who built it, so
    # a total failure is usually a missing column. Dump one row's field list to the
    # log rather than leaving the user to guess at it.
    if skipped:
        for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            logger.warning(f"Flex import skipped {n} row(s): {reason}")
        if not written and items:
            logger.warning(
                f"Fields present on the first row: {sorted(vars(items[0]))}"
            )

    # Most common reason first — with the offending value, so the user can see
    # what the parser actually received.
    detail = "; ".join(
        f"{n}× {reason}" for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:3]
    )
    if written:
        message = f"Imported {written} trade(s) from {', '.join(used)}"
        if deduped:
            message += f", skipping {deduped} already recorded live"
        if skipped:
            message += f". Skipped {skipped}: {detail}"
        if no_ids:
            message += (
                ". This query has no Trade ID / IB Exec ID column, so ids were derived "
                "from each row's contents — add those columns for exact matching"
            )
        if no_price:
            message += ". No Trade Price column in this query, so fill prices are blank"
        if level_note:
            message += f". Note: {level_note}"
        if date_order:
            message += f". Dates read as {'day/month/year' if date_order == 'DMY' else 'month/day/year'}"
    else:
        message = f"Imported nothing — all {skipped} row(s) rejected: {detail}"
        # Which columns the query emits is the user's choice, so name what actually
        # arrived; it is almost always a missing column rather than a broken report.
        if items:
            message += f". Fields on the first row: {', '.join(sorted(vars(items[0])))}"

    return {
        "status": "success" if written else "error",
        "imported": written,
        "skipped": skipped,
        "reasons": reasons,
        "deduped": deduped,
        "sections": used,
        "fields": sorted(vars(items[0])) if items else [],
        "message": message,
    }
