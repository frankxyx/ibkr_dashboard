import asyncio
import hmac
import json
import logging
import math
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from ib_async import IB, LimitOrder, OrderStatus, Stock, util
from pydantic import BaseModel

import trade_history

try:
    from dotenv import load_dotenv
    load_dotenv() # looks for a file named .env in the current directory and coipies each KEY=value line into os.environ, as if you'd exported them in your shell. 
except ImportError:
    pass

logger = logging.getLogger(__name__)
# LOG_LEVEL=DEBUG dumps the raw IBKR wire protocol and a line per tick — useful for
# protocol-level tracing, far too noisy otherwise.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def env_flag(name):
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


# Debug endpoints expose account balances and positions with no auth — opt in explicitly.
DEBUG_API_ENABLED = env_flag("DASHBOARD_DEBUG")

# --- DEPLOYMENT CONFIG ---
# IBKR_PORT: 4001 = IB Gateway LIVE, 4002 = IB Gateway paper,
#            7496 = TWS live,        7497 = TWS paper.
IBKR_HOST = os.getenv("IBKR_HOST", "127.0.0.1")
IBKR_PORT = int(os.getenv("IBKR_PORT", "4001"))
IBKR_CLIENT_ID = int(os.getenv("IBKR_CLIENT_ID", "1"))

# Market data mode: 1 = live, 2 = frozen (last close), 3 = delayed (~15 min),
# 4 = delayed-frozen.
IBKR_MARKET_DATA_TYPE = int(os.getenv("IBKR_MARKET_DATA_TYPE", "3"))
MARKET_DATA_TYPE_NAMES = {1: "live", 2: "frozen", 3: "delayed", 4: "delayed-frozen"}

# Bind to loopback by default. The old 0.0.0.0 put an unauthenticated dashboard that can
# place live orders onto the whole local network — fine to opt into, wrong to ship.
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# --- WATCHLIST PERSISTENCE ---
# PORTFOLIO lives in memory, so a ticker added through the UI vanished on restart
# and the list silently reverted to the env default. A saved file, when present,
# wins over the env so the thing you curated survives a reboot.
WATCHLIST_FILE = os.getenv(
    "WATCHLIST_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "watchlist.json"),
)


def load_saved_watchlist():
    """Saved symbols, or None when nothing has been saved yet."""
    try:
        with open(WATCHLIST_FILE) as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning(f"Saved watchlist at {WATCHLIST_FILE} is unreadable ({e}) — using the default")
        return None
    syms, seen = [], set()
    for raw in (data.get("symbols") or []):
        sym = str(raw).strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            syms.append(sym)
    # A readable file is authoritative even when empty; None means "no file".
    return syms


def autosave_watchlist(reason):
    """Persists after a change. Never raises — a failed save must not fail the
    add or delete the user actually asked for, but it must be visible in the log."""
    ok, detail = save_watchlist(PORTFOLIO)
    if ok:
        logger.info(f"Watchlist auto-saved after {reason}: {len(PORTFOLIO)} symbols")
    else:
        logger.error(f"Watchlist auto-save after {reason} FAILED: {detail}")
    return ok


def save_watchlist(symbols):
    """Writes the watchlist atomically. Returns (ok, message)."""
    payload = {
        "symbols": list(symbols),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    tmp = f"{WATCHLIST_FILE}.tmp"
    try:
        # Write-then-rename: a crash mid-write must not leave a truncated file
        # that silently resets the watchlist on the next boot.
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, WATCHLIST_FILE)
        return True, WATCHLIST_FILE
    except Exception as e:
        logger.error(f"Could not save the watchlist: {e}")
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False, str(e)


PORTFOLIO = [
    t.strip().upper()
    for t in os.getenv("PORTFOLIO", "NBIS,CRWV,BE,FCEL,FSLR,MRVL,AVGO,SNDK,TSM,NVDA,AMD,INTC,MDB,SNOW,ESTC,AMZN,MSFT,GOOGL,PLTR,RKLB,PL").split(",")
    if t.strip()
]

# --- API AUTHENTICATION ---
# A shared secret, required on every /api/* request. This is a SEPARATE defence from CORS and solves a different problem: CORS is enforced by the browser and
# is invisible to anything that is not one, so it does nothing against a local process that can reach this port. Only a secret the caller must possess stops that.

# Set DASHBOARD_TOKEN in .env to keep it stable across restarts. 
# Left unset, a fresh one is generated per boot and printed to the terminal secure, but you will have to re-paste it into the UI after every restart.
DASHBOARD_TOKEN = os.getenv("DASHBOARD_TOKEN") or secrets.token_urlsafe(24)
TOKEN_FROM_ENV = bool(os.getenv("DASHBOARD_TOKEN"))


def log_auth_banner():
    if TOKEN_FROM_ENV:
        logger.info("API auth enabled using DASHBOARD_TOKEN from the environment")
        return
    logger.warning("=" * 72)
    logger.warning("  API TOKEN (generated for this run — changes on every restart):")
    logger.warning(f"      {DASHBOARD_TOKEN}")
    logger.warning("  Paste it into the dashboard when prompted.")
    logger.warning("  To keep it stable, add this line to .env and restart:")
    logger.warning(f"      DASHBOARD_TOKEN={DASHBOARD_TOKEN}")
    logger.warning("=" * 72)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # _startup and _shutdown are defined further down, beside the code they touch;
    # Python resolves them at call time, so the forward reference is fine.
    await _startup()
    try:
        yield
    finally:
        # finally, not a bare call after yield: a crash while serving must still
        # release the IBKR client id, or the next start dies with error 326.
        await _shutdown()


app = FastAPI(title="Trading Dashboard API", lifespan=lifespan)
ib = IB()

@app.middleware("http")
async def require_dashboard_token(request, call_next):
    # Only the API is gated. index.html must stay reachable so the UI can load
    # and prompt for the token; it contains no secrets of its own.
    if request.url.path.startswith("/api/"):
        supplied = request.headers.get("x-dashboard-token", "")
        # compare_digest rather than ==: a plain string comparison returns early
        # at the first differing byte, which leaks the token a character at a time to anyone who can measure response timing.
        if not hmac.compare_digest(supplied, DASHBOARD_TOKEN):
            client = request.client.host if request.client else "unknown"
            logger.warning(
                f"REJECTED unauthenticated {request.method} {request.url.path} from {client}"
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid X-Dashboard-Token header"},
            )
    return await call_next(request)

# The UI is served from this same origin, so the only origins that ever need to be allowed are this server's own. 
ALLOWED_ORIGINS = [
    f"http://127.0.0.1:{SERVER_PORT}",
    f"http://localhost:{SERVER_PORT}",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Dashboard-Token"],
)

def new_quote_entry(currency="USD"):
    # last_tick_at is time.monotonic() of the last real tick, or None if none yet.
    return {"price": 0.0, "bid": 0.0, "ask": 0.0, "currency": currency, "last_tick_at": None}


# global dictionary to store the latest price in memory
# onPendingTickers event handler and GET /api/prices work asynchronously
# "currency" starts as USD and is corrected by record_currency() once the contract is resolved: a non-US listing (e.g. VUAG in GBP) fixes itself as soon as it qualifies.
_saved_watchlist = load_saved_watchlist()
if _saved_watchlist is not None:
    PORTFOLIO[:] = _saved_watchlist
    logger.info(f"Watchlist restored from {WATCHLIST_FILE}: {len(PORTFOLIO)} symbols")
else:
    logger.info(f"No saved watchlist — using the {len(PORTFOLIO)}-symbol default")

latest_prices = {ticker: new_quote_entry() for ticker in PORTFOLIO}
active_contracts = {}
# symbol -> every currency IBKR lists it in, when there is more than one.
contract_alternatives = {}
# symbol -> the same fund's ticker in the account currency, when one exists.
base_twins = {}
# symbol -> IBKR's price magnifier. 100 means the feed quotes PENCE while
# reporting the currency as GBP, which is normal for many LSE lines. Returns
# are unaffected (the factor cancels), so only displayed prices need dividing.
price_magnifiers = {}


def in_currency(symbol, value):
    """A quoted price converted to whole currency units."""
    mag = price_magnifiers.get(symbol.upper()) or 1
    return value / mag if (value and mag != 1) else value

connection_settings = {
    "host": IBKR_HOST,
    "port": IBKR_PORT,
    "client_id": IBKR_CLIENT_ID
}

# Maps an IBKR request id to a timestamped list of the error messages it produced,
# so place_order() can look up why IBKR rejected an order.
order_errors = {}

ORDER_ERRORS_TTL = 900.0             # forget an order's errors 15 min after its last message
ORDER_ERRORS_MAX_IDS = 500           # hard cap on tracked ids, regardless of age
ORDER_ERRORS_MAX_PER_ID = 20         # one order can emit many messages; keep the recent ones
ORDER_ERRORS_SWEEP_INTERVAL = 300.0  # how often the sweep runs

def is_informational_error(code):
    return 2100 <= code < 2200 or code in {10349}


def prune_order_errors():
    cutoff = time.monotonic() - ORDER_ERRORS_TTL
    expired = [
        key for key, entries in order_errors.items()
        if not entries or max(e["ts"] for e in entries) < cutoff
    ]
    for key in expired:
        order_errors.pop(key, None)

    overflow = len(order_errors) - ORDER_ERRORS_MAX_IDS
    dropped_for_size = 0
    if overflow > 0:
        for key in sorted(order_errors)[:overflow]:
            order_errors.pop(key, None)
            dropped_for_size += 1

    if expired or dropped_for_size:
        logger.info(
            f"order_errors sweep: dropped {len(expired)} expired + {dropped_for_size} "
            f"over-cap, {len(order_errors)} ids retained"
        )


async def sweep_order_errors():
    while True:
        # yields control back to the loop. The coroutine effectively says "wake me in five minutes," and the loop goes off and serves everything else.
        # It consumes no thread and no CPU while waiting.
        await asyncio.sleep(ORDER_ERRORS_SWEEP_INTERVAL) 
        try:
            prune_order_errors()
        except Exception as e:
            logger.error(f"order_errors sweep failed: {e}", exc_info=e)


# --- MARKET DATA FEED HEALTH ---
US_EASTERN = ZoneInfo("America/New_York")

feed_state = {
    "last_quote_at": None,   # time.monotonic() of the last tick carrying a real value
    "farms": {},             # farm name -> "ok" | "down" | "connecting" | "inactive"
    "data_errors": [],       # recent entitlement/subscription failures
    "delayed": False,        # IBKR fell back to the delayed feed (code 10167)
}

# Silence that means trouble. Split by session: 21 liquid names quoting during RTH
# go quiet only if something broke, while sparse pre/after-hours ticking is normal.
FEED_STALE_RTH = 30.0
FEED_STALE_EXTENDED = 180.0
FEED_MAX_DATA_ERRORS = 10

# Feed-failure codes shown to the user verbatim; 10167 is excluded because it means delayed data IS flowing.
MARKET_DATA_ERROR_CODES = {354, 10089, 10168, 10197, 200}

# 162/165 are the HISTORICAL data service, not the streaming quote feed. Keeping 162
# in the set above made a failed analytics refresh light up the Live Watchlist with
# "No market data subscription" — about a different feed entirely. Refresh failures
# surface through /api/analytics/coverage and the Refresh button instead.
HISTORICAL_DATA_ERROR_CODES = {162, 165}
DELAYED_FEED_CODE = 10167

# Farm-status notices. The message tail after ':' is the farm name (e.g. "...OK:usfarm").
FARM_STATUS_CODES = {2103: "down", 2104: "ok", 2107: "inactive", 2108: "inactive", 2119: "connecting"}


# --- ERROR SEVERITY ---
# Good news and normal lifecycle events.
INFO_ERROR_CODES = {
    2104,   # Market data farm connection is OK
    2106,   # HMDS data farm connection is OK
    2158,   # Sec-def data farm connection is OK
    2107,   # HMDS farm inactive, available on demand
    2108,   # Market data farm inactive, available on demand
    2119,   # Market data farm is connecting
    1102,   # Connectivity restored, data maintained
    202,    # Order cancelled — this is the CONFIRMATION of a successful cancel
    10349,  # Order TIF defaulted to DAY by preset
    10167,  # Falling back to the delayed feed — data IS being served
}

# Real problems, but expected/recoverable — not code faults.
WARNING_ERROR_CODES = {
    2103,   # Market data farm connection is broken
    2105,   # HMDS data farm connection is broken
    2157,   # Sec-def data farm connection is broken
    1100,   # Connectivity to IBKR lost
    1101,   # Connectivity restored, subscriptions LOST (needs resubscribe)
    354, 10089, 10168, 10197,  # market data not subscribed / not entitled (10167 is INFO, see above)
    110, 165, 321, 329, 399, 404, 434,  # order/request validation warnings
}


def error_log_level(code):
    # Maps an IBKR code to a log level. Unknown codes stay ERROR — fail loud, not silent.
    if code in INFO_ERROR_CODES:
        return logging.INFO
    if code in WARNING_ERROR_CODES:
        return logging.WARNING
    if 2100 <= code < 2200:
        # Remaining 21xx are system status notices; warn rather than error.
        return logging.WARNING
    return logging.ERROR


def parse_farm_name(message):
    # Extracts the farm name from a 21xx status notice.
    # IBKR uses two separators for the same kind of message:
    if ":" in message:
        return message.rsplit(":", 1)[1].strip()
    marker = "upon demand."
    idx = message.find(marker)
    if idx != -1:
        return message[idx + len(marker):].strip()
    return ""

market_schedule = {
    "covered_days": set(),   # every date the cached string described, CLOSED ones included
    "liquid": [],            # RTH sessions (9:30–16:00 US Eastern)
    "trading": [],           # extended hours (4:00–9:30 and 16:00–20:00 US Eastern)
    "fetched_at": None,
    "source": None,          # "ibkr" when the cache is usable, else None
}

SCHEDULE_REFRESH_INTERVAL = 12 * 3600


def _covered_days(raw):
    days = set()
    for part in (raw or "").split(";"):
        key = part.split(":")[0].strip()
        if len(key) == 8 and key.isdigit():
            days.add(date(int(key[:4]), int(key[4:6]), int(key[6:8])))
    return days


# --- PER-EXCHANGE SCHEDULES ---
# The US calendar cannot speak for a London listing: VUAG trades 08:00-16:30 UK
# while NYSE is shut, and "market closed" was being decided globally off SPY.
# Keyed by primary exchange, so one request covers every symbol listed there.
exchange_schedules = {}


def _schedule_from_details(d):
    return {
        "covered_days": _covered_days(d.tradingHours) | _covered_days(d.liquidHours),
        "liquid": d.liquidSessions(),
        "trading": d.tradingSessions(),
        "tz": getattr(d, "timeZoneId", "") or "",
    }


def _session_from(sched, now_utc):
    """Session for one exchange's schedule, or None when it cannot say.

    Compared in UTC: IBKR returns tz-aware session boundaries in the exchange's
    own zone, so naive local comparison would silently judge London by New York.
    """
    if not sched:
        return None
    liquid = sched.get("liquid") or []
    trading = sched.get("trading") or []
    if not liquid and not trading:
        return None

    def to_utc(x):
        return x.astimezone(timezone.utc) if x.tzinfo else x.replace(tzinfo=timezone.utc)

    today = [x for x in liquid if to_utc(x.start).date() == now_utc.date()
             or to_utc(x.end).date() == now_utc.date()]
    if any(to_utc(x.start) <= now_utc <= to_utc(x.end) for x in liquid):
        return "open"
    if any(to_utc(x.start) <= now_utc <= to_utc(x.end) for x in trading):
        return "extended"
    covered = sched.get("covered_days") or set()
    if now_utc.date() not in covered:
        return None
    return "weekend" if now_utc.weekday() >= 5 else ("holiday" if not today else "closed")


def exchange_of(symbol):
    c = active_contracts.get(symbol.upper())
    if not c:
        return None
    return (c.primaryExchange or c.exchange or "").upper() or None


def symbol_session(symbol, now_utc=None):
    """(session, last close date) for one symbol's own exchange."""
    ex = exchange_of(symbol)
    sched = exchange_schedules.get(ex) if ex else None
    if not sched:
        return None, None
    now_utc = now_utc or datetime.now(timezone.utc)
    sess = _session_from(sched, now_utc)

    def to_utc(x):
        return x.astimezone(timezone.utc) if x.tzinfo else x.replace(tzinfo=timezone.utc)

    ended = [to_utc(x.end) for x in (sched.get("liquid") or []) if to_utc(x.end) <= now_utc]
    last_close = max(ended).date().isoformat() if ended else None
    return sess, last_close


async def record_magnifier(contract):
    """Caches the contract's price magnifier so prices can be shown in currency."""
    try:
        det = await asyncio.wait_for(ib.reqContractDetailsAsync(contract), timeout=10.0)
        match = next((d for d in (det or [])
                      if d.contract.conId == contract.conId), None) or (det[0] if det else None)
        mag = int(getattr(match, "priceMagnifier", 1) or 1) if match else 1
    except Exception as e:
        logger.debug(f"priceMagnifier lookup failed for {contract.symbol}: {e}")
        mag = 1
    price_magnifiers[contract.symbol.upper()] = mag
    if mag != 1:
        logger.info(f"{contract.symbol} is quoted in minor units "
                    f"(priceMagnifier={mag}) — dividing displayed prices by {mag}")
    return mag


async def refresh_base_twins():
    """For each non-base-currency holding, find the same fund in the account's
    currency. Runs once after subscriptions; the ISIN results are cached."""
    base = base_currency()
    for sym, c in list(active_contracts.items()):
        if (c.currency or "").upper() == base:
            continue
        try:
            twin = await base_currency_twin(sym, c.currency)
            if twin:
                base_twins[sym] = twin
                logger.info(f"{sym} ({c.currency}) has a {base} twin: "
                            f"{twin['symbol']} on {twin['exchange']}")
        except Exception as e:
            logger.debug(f"twin lookup failed for {sym}: {e}")
        await asyncio.sleep(0.3)


async def refresh_exchange_schedules():
    """One contract-details request per distinct exchange on the watchlist."""
    if not ib.isConnected():
        return False
    by_exchange = {}
    for sym, c in list(active_contracts.items()):
        ex = (c.primaryExchange or c.exchange or "").upper()
        if ex and ex not in by_exchange:
            by_exchange[ex] = c
    for ex, contract in by_exchange.items():
        try:
            details = await asyncio.wait_for(
                ib.reqContractDetailsAsync(contract), timeout=15.0
            )
            if details:
                exchange_schedules[ex] = _schedule_from_details(details[0])
                logger.info(f"Schedule cached for {ex} ({exchange_schedules[ex]['tz'] or 'tz unknown'})")
        except Exception as e:
            logger.warning(f"Could not fetch the {ex} schedule: {e}")
        await asyncio.sleep(0.3)
    return bool(exchange_schedules)


async def refresh_market_schedule():
    # Caches IBKR's trading calendar. True when the cache was updated.
    if not ib.isConnected():
        return False
    try:
        details = await asyncio.wait_for(
            ib.reqContractDetailsAsync(Stock("SPY", "SMART", "USD")), timeout=15.0
        )
    except Exception as e:
        logger.warning(f"Market schedule fetch failed, falling back to weekday hours: {e}")
        return False
    if not details:
        return False

    d = details[0]
    market_schedule.update(
        covered_days=_covered_days(d.tradingHours) | _covered_days(d.liquidHours),
        liquid=d.liquidSessions(),
        trading=d.tradingSessions(),
        fetched_at=time.monotonic(),
        source="ibkr",
    )
    days = market_schedule["covered_days"]
    logger.info(
        f"Market schedule from IBKR: {len(days)} days ({min(days)} to {max(days)})"
        if days else "Market schedule came back empty"
    )
    return True


async def schedule_refresher():
    # Keeps the cached calendar current for the life of the process.
    while True:
        await asyncio.sleep(SCHEDULE_REFRESH_INTERVAL)
        try:
            await refresh_market_schedule()
        except Exception as e:
            logger.error(f"Schedule refresh failed: {e}", exc_info=e)


def _session_from_schedule(now):
    # Session per IBKR, or None when the cache does not cover this moment.
    if market_schedule["source"] != "ibkr":
        return None
    if now.date() not in market_schedule["covered_days"]:
        return None

    today_trading = [x for x in market_schedule["trading"] if x.start.date() == now.date()]
    today_liquid = [x for x in market_schedule["liquid"] if x.start.date() == now.date()]

    # Covered by the schedule but carrying no session: a holiday, or a weekend.
    if not today_trading and not today_liquid:
        return "weekend" if now.weekday() >= 5 else "holiday"
    if any(x.start <= now <= x.end for x in today_liquid):
        return "open"
    if any(x.start <= now <= x.end for x in today_trading):
        first_rth = min((x.start for x in today_liquid), default=None)
        return "premarket" if (first_rth and now < first_rth) else "afterhours"
    return "closed"


def last_completed_session(now=None):
    """Date of the most recent finished regular session, or None if unknown.

    What the price on a closed-market card actually represents. Without it the UI
    can only say "last close" and leave the user to work out which day that was.
    """
    if market_schedule["source"] != "ibkr":
        return None
    now = now or datetime.now(US_EASTERN)
    ended = [x.end for x in market_schedule["liquid"] if x.end <= now]
    return max(ended).date().isoformat() if ended else None


def us_market_session(now=None):
    # Current session per IBKR's calendar, or None when IBKR has not told us.
    return _session_from_schedule(now or datetime.now(US_EASTERN))


# --- GLOBAL EVENT HANDLERS ---
def onError(reqId, errorCode, errorString, _contract):
    '''farm status tracking, order rejection capture, and feed health logging.'''
    # reqId is included because it is the only link back to the order or data request that failed; without it a rejection message cannot be attributed to anything.
    logger.log(
        error_log_level(errorCode),
        f"IBKR {errorCode} (reqId {reqId}): {errorString}"
    )

    if errorCode in FARM_STATUS_CODES:
        farm = parse_farm_name(errorString)
        if farm:
            feed_state["farms"][farm] = FARM_STATUS_CODES[errorCode]

    # Not a fault: IBKR is telling us the ~ 15min delayed feed is being served in place of real time data.
    # Worth surfacing because order entry seeds its limit price from these quotes, so the user should know the number they are looking at lags the market.
    if errorCode == DELAYED_FEED_CODE:
        feed_state["delayed"] = True

    # 1101 = "connectivity restored, subscriptions LOST". The socket is fine, so the
    # supervisor sees nothing wrong — but every reqMktData is dead until re-issued.
    if errorCode == 1101:
        logger.warning("IBKR 1101: subscriptions lost on reconnect — re-subscribing")
        # resubscribe_all() rather than clearing: forgetting a contract without
        # cancelMktData leaks the data line it was using.
        spawn_background(resubscribe_all(), "resubscribe-after-1101")

    if errorCode in MARKET_DATA_ERROR_CODES:
        feed_state["data_errors"].append({
            "code": errorCode, "message": errorString, "ts": time.monotonic()
        })
        del feed_state["data_errors"][:-FEED_MAX_DATA_ERRORS]

    # For order-related errors IBKR sets reqId to the orderId; -1 means unrelated to an order.
    if reqId and reqId > 0:
        entries = order_errors.setdefault(reqId, [])
        entries.append({"code": errorCode, "message": errorString, "ts": time.monotonic()})
        # Keep only the most recent messages for this id (no-op below the cap).
        del entries[:-ORDER_ERRORS_MAX_PER_ID]

ib.errorEvent += onError


def record_currency(contract):
    entry = latest_prices.setdefault(contract.symbol, new_quote_entry())
    entry["currency"] = contract.currency or "USD"

# except catches both TypeError fron None and ValueError from a non-numeric string. 
def safe_float(v):
    try:
        f = float(v)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0

def resolve_quote(t):
    # t.marketPrice() is typically the bid/ask midpoint.
    price = safe_float(t.last) or safe_float(t.marketPrice()) or safe_float(t.close)
    return price, safe_float(t.bid), safe_float(t.ask)


def apply_quote(symbol, price, bid, ask):
    entry = latest_prices.get(symbol)
    if entry is None:
        return
    if price > 0.0:
        entry["price"] = price
    if bid > 0.0:
        entry["bid"] = bid
    if ask > 0.0:
        entry["ask"] = ask


def onPendingTickers(tickers):
    for t in tickers:
        # Update the global state so the API can serve the freshest data
        symbol = t.contract.symbol
        if symbol not in latest_prices:
            continue

        price, bid, ask = resolve_quote(t)
        apply_quote(symbol, price, bid, ask)

        if price > 0.0 or bid > 0.0 or ask > 0.0:
            now = time.monotonic()
            feed_state["last_quote_at"] = now
            latest_prices[symbol]["last_tick_at"] = now

        logger.debug(f"[TICK] {symbol} price={price} bid={bid} ask={ask}")

ib.pendingTickersEvent += onPendingTickers


# --- EXECUTED FILL CAPTURE ---
# Two paths into the store, because neither alone is sufficient:
#   * the events below catch fills the moment they happen, while the dashboard is up;
#   * the sweep re-reads ib.fills() because a commission report for an order placed
#     before this connection started updates the Fill in place WITHOUT firing
#     commissionReportEvent (ib_async only emits it when it can match a live Trade).
# Realized PnL rides on the commission report, so missing that second path would
# leave every pre-existing fill showing no PnL forever.
FILL_SYNC_INTERVAL = 15.0


def save_fills(fills, context):
    try:
        n = trade_history.record_fills(fills)
        if n:
            logger.debug(f"Recorded {n} fill(s) from {context}")
    except Exception as e:
        # A history write must never take down order handling.
        logger.error(f"Failed to record fills from {context}: {e}")


def onExecDetails(trade, fill):
    logger.info(
        f"Fill: {fill.execution.side} {fill.execution.shares} "
        f"{fill.contract.symbol} @ {fill.execution.price}"
    )
    save_fills([fill], "execDetails")


def onCommissionReport(trade, fill, report):
    save_fills([fill], "commissionReport")


ib.execDetailsEvent += onExecDetails
ib.commissionReportEvent += onCommissionReport


async def sync_fills():
    """Sweeps ib_async's fill cache into the store on a timer."""
    while True:
        await asyncio.sleep(FILL_SYNC_INTERVAL)
        if not ib.isConnected():
            continue
        try:
            save_fills(ib.fills(), "periodic sync")
        except Exception as e:
            logger.error(f"Fill sync failed: {e}", exc_info=e)


async def backfill_fills():
    """Pulls the current session's executions after a (re)connect.

    IBKR only serves the current trading day here — everything older is already in
    the store from the session it happened in, which is the whole reason the store
    exists.
    """
    if not ib.isConnected():
        return
    try:
        fills = await asyncio.wait_for(ib.reqExecutionsAsync(), timeout=15.0)
        save_fills(fills, "connect backfill")
        logger.info(f"Fill backfill: {len(fills)} execution(s) from IBKR today")
    except Exception as e:
        logger.warning(f"Could not backfill executions from IBKR: {e}")


_background_tasks = set() # a registry of every long-running task the app spawns, so they can be cancelled on shutdown. Each task removes itself from this set when it completes or fails.
SHUTDOWN_GRACE = 5.0


def _force_exit():
    # last resort if clean shutdown stalls.
    logger.error(
        f"Shutdown stalled beyond {SHUTDOWN_GRACE}s — forcing exit so the IBKR "
        f"client id is released"
    )
    os._exit(1)


def spawn_background(coro, name):
    task = asyncio.create_task(coro, name=name) #1. schedule it
    _background_tasks.add(task) #2. keep a strong reference so it doesn't get garbage collected and cancelled

    def _on_done(t):
        _background_tasks.discard(t) # 3a. clean up the registry when the task completes, fails or is cancelled
        if t.cancelled():
            logger.warning(f"Background task '{name}' was cancelled")
            return
        exc = t.exception() #3b. surface the failure 
        if exc:
            logger.error(f"Background task '{name}' failed: {exc!r}", exc_info=exc)

    task.add_done_callback(_on_done) # 4. hand it back
    return task

# IBKR secType codes, for error messages only. Anything not listed is reported verbatim.
SEC_TYPE_NAMES = {
    "IND": "index",
    "FUT": "future",
    "OPT": "option",
    "FOP": "future option",
    "CASH": "forex pair",
    "CFD": "CFD",
    "BOND": "bond",
    "FUND": "mutual fund",
    "CRYPTO": "cryptocurrency",
    "WAR": "warrant",
}

# a safe wrapper around IBKR's contract lookup. It turns a call with three different failure modes into one that returns either a usable contract or None.
async def try_qualify(contract, timeout=10.0):
    try:
        result = await asyncio.wait_for(ib.qualifyContractsAsync(contract), timeout=timeout)
    except Exception as e:
        logger.debug(f"qualify failed for {contract.symbol} ({contract.exchange}/{contract.currency}): {e}")
        return None
    qualified = result[0] if result else None
    return qualified if (qualified and qualified.conId) else None


# Which currency line to prefer when a symbol lists in several. An LSE ETF
# commonly has GBP and USD lines for the same fund, and hardcoding USD handed a
# GBP-based account the wrong one. Set PREFERRED_CURRENCY to override.
PREFERRED_CURRENCY = os.getenv("PREFERRED_CURRENCY", "").strip().upper() or None


def account_base_currency():
    """The account's own reporting currency, or None if IBKR has not said yet."""
    if PREFERRED_CURRENCY:
        return PREFERRED_CURRENCY
    if not ib.isConnected():
        return None
    best, ccy = 0.0, None
    for v in ib.accountValues():
        if v.tag.lower() in ("netliquidation", "netliquidationbycurrency", "totalcashvalue"):
            c = (v.currency or "").upper()
            if c and c != "BASE":
                val = safe_float(v.value)
                if val > best:
                    best, ccy = val, c
    return ccy


def currency_preference():
    """Currency order to try when resolving a ticker.

    USD first unless explicitly overridden. Preferring the ACCOUNT's currency
    seemed right for a GBP user but resolved AMZN and MSFT to their London
    cross-listings — a worse answer than the one it replaced. A currency
    alternative is surfaced by the ISIN twin lookup instead, which suggests
    without hijacking what the user actually typed.
    """
    order = []
    if PREFERRED_CURRENCY:
        order.append(PREFERRED_CURRENCY)
    if "USD" not in order:
        order.append("USD")
    return order


async def resolve_contract(symbol):
    symbol = symbol.upper()
    prefer = currency_preference()

    # Fast path, but in the ACCOUNT's currency first. Trying USD first meant a
    # London fund with both lines always came back as the USD one.
    for ccy in prefer:
        qualified = await try_qualify(Stock(symbol, 'SMART', ccy))
        if qualified:
            return qualified, None

    # Ask IBKR what this symbol actually is.
    try:
        matches = await asyncio.wait_for(ib.reqMatchingSymbolsAsync(symbol), timeout=10.0)
    except Exception as e:
        logger.error(f"Symbol search failed for {symbol}: {e}")
        return None, f"Symbol lookup failed for '{symbol}'"

    # reqMatchingSymbols is a fuzzy search, it returns anything CONTAINING the string,
    # so an exact match is required or searching 'BE' would resolve to something else.
    exact = [m.contract for m in (matches or []) if m.contract.symbol.upper() == symbol]
    candidates = [c for c in exact if c.secType == "STK"]

    if not candidates:
        # the symbol is valid on IBKR, but not an instrument this dashboard handles.
        if exact:
            kinds = ", ".join(sorted({SEC_TYPE_NAMES.get(c.secType, c.secType) for c in exact}))
            return None, (
                f"'{symbol}' exists on IBKR but not as a stock or ETF (listed as: {kinds}). "
                f"This dashboard supports stocks and ETFs only."
            )
        # no exact match but fuzzy hits exist, this is usually a typo, so show the closest mataches to help the user correct it.
        if matches:
            near = ", ".join(
                f"{m.contract.symbol} ({SEC_TYPE_NAMES.get(m.contract.secType, m.contract.secType)})"
                for m in matches[:4]
            )
            return None, f"Unknown symbol '{symbol}'. Closest IBKR matches: {near}"
        return None, f"Unknown symbol '{symbol}' — no match on IBKR"
    
    # Rank by the same preference, so the search path agrees with the fast path.
    def rank(c):
        try:
            return prefer.index((c.currency or "").upper())
        except ValueError:
            return len(prefer)
    candidates.sort(key=rank)

    alts = sorted({(c.currency or "?").upper() for c in candidates})
    if len(alts) > 1:
        logger.info(f"{symbol} lists in {', '.join(alts)} — preferring {prefer[0]}")

    for cand in candidates:
        # retries SMART routing with the correct currency
        qualified = await try_qualify(Stock(symbol, 'SMART', cand.currency))
        if qualified:
            logger.info(
                f"Resolved {symbol} via symbol search: {qualified.currency} "
                f"on {qualified.primaryExchange} (conId={qualified.conId})"
            )
            contract_alternatives[symbol] = alts
            return qualified, None
        # some instruments are not SMART-routable; fall back to the primary exchange.
        if cand.primaryExchange:
            qualified = await try_qualify(Stock(symbol, cand.primaryExchange, cand.currency))
            if qualified:
                logger.info(
                    f"Resolved {symbol} on {qualified.primaryExchange} "
                    f"({qualified.currency}, conId={qualified.conId})"
                )
                contract_alternatives[symbol] = alts
                return qualified, None

    listed = ", ".join(sorted({f"{c.currency}/{c.primaryExchange or 'SMART'}" for c in candidates}))
    return None, f"Could not qualify '{symbol}' (IBKR lists it as {listed})"


# --- SUBSCRIPTION UTILITY ---
async def start_portfolio_subscriptions():
    logger.info(f">>> start_portfolio_subscriptions called. Connected={ib.isConnected()}, PORTFOLIO={PORTFOLIO}")
    if not ib.isConnected():
        logger.warning("Cannot start subscriptions - not connected")
        return
    
    # Held names you are not watching still need a quote. Without this a position
    # outside PORTFOLIO never subscribes, so its price, market value and unrealised
    # PnL stay blank forever — which is exactly how VUAG went unpriced.
    wanted = list(PORTFOLIO)
    try:
        for pos in ib.positions():
            sym = (pos.contract.symbol or "").upper()
            if sym and sym not in wanted:
                wanted.append(sym)
                logger.info(f"{sym} is held but not on the watchlist — subscribing anyway")
    except Exception as e:
        logger.warning(f"Could not add held positions to the subscription list: {e}")

    for ticker in wanted:
        if ticker in active_contracts:
            continue
        try:
            # Resolves currency/exchange per symbol, so a PORTFOLIO holding non-US listings subscribes correctly at startup.
            logger.info(f"Qualifying {ticker}...")
            qualified, error = await resolve_contract(ticker)
            if qualified is None:
                logger.error(f"Failed to qualify contract for {ticker}: {error}")
                continue

            record_currency(qualified)
            # apply_quote() ignores symbols with no entry here, so this must exist
            # before the first tick arrives or the subscription is wasted.
            latest_prices.setdefault(qualified.symbol, new_quote_entry(qualified.currency))
            active_contracts[qualified.symbol] = qualified
            await record_magnifier(qualified)
            ib.reqMktData(qualified, '', snapshot=False, regulatorySnapshot=False)
            logger.info(f"Subscribed to {qualified.symbol} (conId={qualified.conId}, exchange={qualified.primaryExchange})")

        # the except catches any network errors or unexpected crashes during the qualify/subscribe process
        except Exception as e:
            logger.error(f"Exception qualifying/subscribing to {ticker}: {e}")

    logger.info(f"<<< start_portfolio_subscriptions finished. active_contracts={list(active_contracts.keys())}")


def apply_market_data_type():
    ib.reqMarketDataType(IBKR_MARKET_DATA_TYPE) # set the mode for all subsequent subscriptions.
    feed_state["delayed"] = False # clear stale session state
    feed_state["data_errors"].clear()
    name = MARKET_DATA_TYPE_NAMES.get(IBKR_MARKET_DATA_TYPE, str(IBKR_MARKET_DATA_TYPE))
    '''
    1 - real time market data (requires a live subscription)
    2 - frozen market data (last close)
    3 - delayed market data (~15 min)
    4 - delayed frozen market data
    '''
    logger.info(f"Requested {name} market data (type {IBKR_MARKET_DATA_TYPE})")
    if IBKR_MARKET_DATA_TYPE == 1:
        logger.info(
            "Type 1 needs a live subscription; without one IBKR errors (10089) and "
            "sends nothing. Set IBKR_MARKET_DATA_TYPE=3 for delayed quotes instead."
        )


# --- ACCOUNT-SUMMARY LEAK GUARD ---
# ib_async re-subscribes to the account summary on every 1102 ("connectivity
# restored") with a fresh reqId and never cancels the previous one — see
# IB._onError. A flapping gateway therefore stacks subscriptions until IBKR
# refuses with error 322, after which further requests are ignored. Measured: 76
# flaps in 22h produced 147 of them. Cancelling the predecessor keeps exactly one
# alive without disabling ib_async's behaviour.
_account_summary_req = {"id": None, "cancelled": 0}


def _install_account_summary_guard():
    original = ib.client.reqAccountSummary

    def guarded(reqId, groupName, tags):
        previous = _account_summary_req["id"]
        if previous is not None and previous != reqId:
            try:
                ib.client.cancelAccountSummary(previous)
                _account_summary_req["cancelled"] += 1
            except Exception as e:
                logger.debug(f"cancelAccountSummary({previous}) failed: {e}")
        _account_summary_req["id"] = reqId
        return original(reqId, groupName, tags)

    ib.client.reqAccountSummary = guarded


_install_account_summary_guard()


# --- FEED WATCHDOG ---
# The connection supervisor watches the SOCKET. This watches the DATA, because
# the two fail independently: IBKR answers a reconnect with 1102 "data
# maintained" even when every market-data line is dead, and 1101 (the code that
# would tell us to resubscribe) is never sent. Measured on 2026-09-15: 21
# subscriptions accepted, zero ticks for 22 hours, socket healthy throughout.
# So this trusts the observable — are quotes arriving? — over IBKR's claim.
FEED_WATCHDOG_POLL = 15.0
FEED_RECOVERY_COOLDOWN = 180.0

feed_recovery = {"last_attempt": None, "count": 0}


async def resubscribe_all():
    """Cancels every market-data line and requests them again.

    The cancel is not optional: reqMktData without a matching cancelMktData leaks
    a data line each time, and IBKR allows roughly 100.
    """
    for symbol, contract in list(active_contracts.items()):
        try:
            if contract is not None and getattr(contract, "conId", 0):
                ib.cancelMktData(contract)
        except Exception as e:
            logger.debug(f"cancelMktData({symbol}) failed, continuing: {e}")
    active_contracts.clear()
    await asyncio.sleep(1.0)  # give IBKR a moment to release the lines
    await start_portfolio_subscriptions()


async def feed_watchdog():
    """Rebuilds subscriptions when quotes stop during regular hours."""
    while True:
        await asyncio.sleep(FEED_WATCHDOG_POLL)
        try:
            if not ib.isConnected() or not connection_state["wanted"]:
                continue
            if us_market_session() != "open" or not active_contracts:
                continue

            last = feed_state["last_quote_at"]
            # None means nothing has EVER arrived this session — also a stall,
            # and the case that actually bit us.
            age = (time.monotonic() - last) if last else None
            if age is not None and age <= FEED_STALE_RTH:
                continue

            now = time.monotonic()
            since = feed_recovery["last_attempt"]
            if since is not None and now - since < FEED_RECOVERY_COOLDOWN:
                continue

            feed_recovery["last_attempt"] = now
            feed_recovery["count"] += 1
            logger.warning(
                f"Feed watchdog: no quotes for "
                f"{'ever' if age is None else f'{age:.0f}s'} during RTH — "
                f"re-subscribing {len(active_contracts)} symbols "
                f"(recovery #{feed_recovery['count']})"
            )
            await resubscribe_all()
        except Exception as e:
            logger.error(f"Feed watchdog failed: {e}", exc_info=e)


# --- CONNECTION SUPERVISOR ---
# "wanted" is intent, not status: it stays True across a dropped socket so the
# supervisor reconnects, and goes False only when YOU press Disconnect — otherwise
# the button would be undone a few seconds later.
connection_state = {"wanted": True, "reconnects": 0, "last_reconnect": None, "last_error": None}

RECONNECT_POLL = 5.0        # how often to notice a drop
RECONNECT_MIN_DELAY = 5.0
RECONNECT_MAX_DELAY = 120.0  # IB Gateway's nightly restart can take a while


async def attempt_reconnect():
    """One reconnect try. Rebuilds subscriptions on success. Never raises."""
    try:
        await asyncio.wait_for(
            ib.connectAsync(
                connection_settings["host"], connection_settings["port"],
                clientId=connection_settings["client_id"],
            ),
            timeout=10.0,
        )
    except Exception as e:
        connection_state["last_error"] = str(e)
        logger.warning(f"Reconnect failed: {e}")
        return False

    connection_state["reconnects"] += 1
    connection_state["last_reconnect"] = time.strftime("%Y-%m-%d %H:%M:%S")
    connection_state["last_error"] = None
    logger.warning(
        f"Reconnected to IBKR (attempt #{connection_state['reconnects']}) — "
        f"re-subscribing {len(PORTFOLIO)} symbols"
    )

    apply_market_data_type()
    # Market data lines do not survive the drop. Clearing this is what makes
    # deferred_subscriptions re-request them instead of skipping every symbol.
    active_contracts.clear()
    spawn_background(deferred_subscriptions(), "reconnect-subscriptions")
    spawn_background(backfill_fills(), "reconnect-fill-backfill")
    spawn_background(refresh_market_schedule(), "reconnect-schedule")
    warn_if_live_account()
    return True


async def connection_supervisor():
    """Reconnects after a drop, backing off so a dead gateway is not hammered."""
    delay = RECONNECT_MIN_DELAY
    while True:
        await asyncio.sleep(RECONNECT_POLL)

        if ib.isConnected():
            delay = RECONNECT_MIN_DELAY
            continue
        if not connection_state["wanted"]:
            continue  # you disconnected on purpose

        logger.warning(f"IBKR connection is down — retrying in {delay:.0f}s")
        await asyncio.sleep(delay)
        if ib.isConnected() or not connection_state["wanted"]:
            continue

        delay = RECONNECT_MIN_DELAY if await attempt_reconnect() else min(delay * 2, RECONNECT_MAX_DELAY)


# --- STARTUP / SHUTDOWN EVENTS ---
async def _startup():
    log_auth_banner()

    # spawn_background() is used rather than asyncio.create_task() so the task is registered in _background_tasks and can be cancelled on shutdown.
    spawn_background(sweep_order_errors(), "order-errors-sweep")
    spawn_background(sync_fills(), "fill-sync")
    spawn_background(refresh_fx_rates(), "fx-refresh")
    # Outside the try as well: if the Gateway is down at boot, this is what brings
    # the dashboard up on its own once you start it.
    spawn_background(connection_supervisor(), "connection-supervisor")
    spawn_background(feed_watchdog(), "feed-watchdog")

    # Attempt to connect with defaults on startup, but fail gracefully if offline
    logger.info(f"Connecting to IBKR Gateway at {connection_settings['host']}:{connection_settings['port']}...")
    try:
        # Give connection 5 seconds to respond
        await asyncio.wait_for(
            ib.connectAsync(
                connection_settings["host"],
                connection_settings["port"],
                clientId=connection_settings["client_id"]
            ),
            timeout=5.0
        )
        apply_market_data_type()

        spawn_background(deferred_subscriptions(), "startup-subscriptions")
        spawn_background(backfill_fills(), "startup-fill-backfill")
        spawn_background(refresh_market_schedule(), "market-schedule")
        spawn_background(schedule_refresher(), "market-schedule-refresher")
        logger.info("IBKR connected — subscriptions will start in background")
        warn_if_live_account()
    except Exception as e:
        logger.warning(f"Could not connect to IBKR Gateway on startup (Gateway likely offline): {e}")


def warn_if_live_account():
    """
    The shipped default port is 4001 (LIVE) real money account.
    """
    accounts = ib.managedAccounts() or []
    # Paper accounts are prefixed DU/DF; anything else is real money.
    live = [a for a in accounts if not (a.startswith("DU") or a.startswith("DF"))]
    if live:
        logger.warning("=" * 72)
        logger.warning(f"  LIVE ACCOUNT(S) CONNECTED: {', '.join(live)}")
        logger.warning(f"  Port {connection_settings['port']} — orders placed here USE REAL MONEY.")
        logger.warning("  For simulated trading set IBKR_PORT=4002 (IB Gateway paper) and restart.")
        logger.warning("=" * 72)
    elif accounts:
        logger.info(f"Paper account(s) connected: {', '.join(accounts)} — simulated trading")
'''
a retry wrapper around the actual subscription work, because a silent failure here leaves the price feed dead for the whole session: 
every symbol falls back to the stale portfolio snapshot in /api/prices, which has no bid/ask, and nothing ever tries again.
'''
async def deferred_subscriptions(delay: float = 2.0, attempts: int = 3):
    for attempt in range(1, attempts + 1):
        await asyncio.sleep(delay)

        if not ib.isConnected():
            logger.warning(f"Subscription attempt {attempt}/{attempts}: not connected")
            continue

        await start_portfolio_subscriptions()

        missing = [s for s in PORTFOLIO if s not in active_contracts]
        if not missing:
            logger.info(f"Subscriptions complete: {list(active_contracts.keys())}")
            # Only now are the contracts known, and the schedule lookup is keyed
            # off their exchanges — spawning this earlier found nothing.
            await refresh_exchange_schedules()
            await refresh_base_twins()
            return

        logger.warning(f"Subscription attempt {attempt}/{attempts} incomplete, missing {missing}")

    logger.error(
        f"Subscriptions still incomplete after {attempts} attempts: "
        f"missing {[s for s in PORTFOLIO if s not in active_contracts]}"
    )
    # Whatever did subscribe still deserves a correct session label.
    await refresh_exchange_schedules()

async def _shutdown():
    logger.info("Shutting down...")

    # Armed FIRST, before anything that can block. It used to start after the task
    # gather below — so a gather that hung meant the watchdog never armed at all, and
    # the process survived `kill` still holding clientId 1. That is exactly the orphan
    # that then blocks the next start with error 326.
    watchdog = threading.Timer(SHUTDOWN_GRACE, _force_exit)
    watchdog.daemon = True
    watchdog.start()
    try:
        for task in list(_background_tasks):
            task.cancel()
        if _background_tasks:
            # Bounded as well as watchdogged: a cancel that ib_async does not honour
            # should cost a couple of seconds, not the full grace period.
            try:
                await asyncio.wait_for(
                    asyncio.gather(*list(_background_tasks), return_exceptions=True),
                    timeout=SHUTDOWN_GRACE / 2,
                )
            except asyncio.TimeoutError:
                logger.warning("Background tasks did not finish cancelling — continuing")

        close_analytics_data_connection()

        if ib.isConnected():
            ib.disconnect()
            logger.info("IBKR disconnected")
    except Exception as e:
        logger.warning(f"Shutdown did not complete cleanly: {e}")
    finally:
        watchdog.cancel()


# --- CONNECTION API ENDPOINTS ---
class ConnectionConfig(BaseModel):
    host: str = IBKR_HOST
    port: int = IBKR_PORT
    client_id: int = IBKR_CLIENT_ID

@app.get("/api/connection/status")
async def get_connection_status():
    # Returns the current IBKR gateway connection status.
    return {
        "connected": ib.isConnected(),
        "host": connection_settings["host"],
        "port": connection_settings["port"],
        "client_id": connection_settings["client_id"],
        "auto_reconnect": connection_state["wanted"],
        "feed_recoveries": feed_recovery["count"],
        "stale_account_subs_cancelled": _account_summary_req["cancelled"],
        "reconnects": connection_state["reconnects"],
        "last_reconnect": connection_state["last_reconnect"],
    }

@app.post("/api/connection/connect")
async def connect_gateway(config: ConnectionConfig):
    # Dynamically connects to a user-specified IBKR gateway.
    if ib.isConnected():
        try:
            ib.disconnect()
            active_contracts.clear()
        except Exception as e:
            logger.error(f"Error disconnecting previous session: {e}")

    connection_settings["host"] = config.host
    connection_settings["port"] = config.port
    connection_settings["client_id"] = config.client_id
    connection_state["wanted"] = True

    try:
        logger.info(f"Connecting dynamically to {config.host}:{config.port} (client {config.client_id})...")
        await asyncio.wait_for(
            ib.connectAsync(config.host, config.port, clientId=config.client_id),
            timeout=5.0
        )
        apply_market_data_type()

        # Give IBKR a moment to stream initial account + position data
        await asyncio.sleep(2)

        # Clear old tracking
        active_contracts.clear()
        for symbol in PORTFOLIO:
            known = latest_prices.get(symbol, {}).get("currency", "USD")
            latest_prices[symbol] = new_quote_entry(known)

        # Fire subscriptions in background so the browser gets a response immediately
        spawn_background(deferred_subscriptions(), "connect-subscriptions")
        spawn_background(backfill_fills(), "connect-fill-backfill")
        spawn_background(refresh_market_schedule(), "market-schedule")
        return {"status": "success", "message": "Connected successfully — subscriptions starting"}
    except Exception as e:
        logger.error(f"connect_gateway failed: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/connection/disconnect")
async def disconnect_gateway():
    # Cleanly disconnects the IBKR gateway session.
    # Clear intent first, or the supervisor reconnects within seconds.
    connection_state["wanted"] = False
    try:
        if ib.isConnected():
            ib.disconnect()
        active_contracts.clear()
        # Reset prices to 0.0, keeping each symbol's currency (see connect_gateway).
        for symbol in PORTFOLIO:
            known = latest_prices.get(symbol, {}).get("currency", "USD")
            latest_prices[symbol] = new_quote_entry(known)
        return {"status": "success", "message": "Disconnected successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- FX CONVERSION ---
# Positions in different currencies cannot be weighted against each other without
# a rate. Refreshed on a timer so the sync weight builder can read it cheaply.
# The currency everything is expressed in. Defaults to the ACCOUNT's own
# reporting currency once IBKR says what it is — hardcoding USD converted a
# GBP account's holdings into dollars, the opposite of what it needed.
BASE_CURRENCY_ENV = (os.getenv("BASE_CURRENCY", "") or "").strip().upper() or None
_base_ccy_cache = {"value": None}


def base_currency():
    if BASE_CURRENCY_ENV:
        return BASE_CURRENCY_ENV
    if _base_ccy_cache["value"]:
        return _base_ccy_cache["value"]
    detected = account_base_currency()
    if detected:
        _base_ccy_cache["value"] = detected
        logger.info(f"Base currency detected from the account: {detected}")
        return detected
    return "USD"   # only until the gateway reports otherwise


FX_REFRESH_INTERVAL = 600.0
FX_RETRY_INTERVAL = 15.0
fx_rates = {}


async def _fx_from_market(ccy):
    """(rate, source) for ccy -> base off IBKR market data, or (None, None).

    Tries the conventional quote direction first (GBPUSD = dollars per pound),
    then the inverse (USDJPY = yen per dollar) and reciprocates. Uses a daily
    MIDPOINT bar rather than a live quote so it still resolves at the weekend.
    """
    from ib_async import Forex
    for pair, invert in ((f"{ccy}{base_currency()}", False), (f"{base_currency()}{ccy}", True)):
        try:
            contract = Forex(pair)
            details = await asyncio.wait_for(ib.qualifyContractsAsync(contract), timeout=10.0)
            if not details:
                continue
            bars = await asyncio.wait_for(
                ib.reqHistoricalDataAsync(
                    contract, endDateTime="", durationStr="5 D", barSizeSetting="1 day",
                    whatToShow="MIDPOINT", useRTH=False, formatDate=1,
                ),
                timeout=20.0,
            )
            if not bars:
                continue
            px = safe_float(bars[-1].close)
            if px > 0:
                return (1.0 / px if invert else px), f"{pair} close {bars[-1].date}"
        except Exception as e:
            logger.debug(f"FX {pair} lookup failed: {e}")
    return None, None


async def refresh_fx_rates():
    """Keeps a rate for every currency actually held. Never raises."""
    connected_once = False
    while True:
        try:
            if ib.isConnected():
                needed = {
                    (item.contract.currency or base_currency()).upper()
                    for item in ib.portfolio()
                } | {
                    (p.contract.currency or base_currency()).upper() for p in ib.positions()
                }
                # IBKR's own account rate first: free, already computed for the
                # account's base currency, and available while markets are shut.
                account_rates = {}
                for v in ib.accountValues():
                    if v.tag == "ExchangeRate" and v.currency:
                        r = safe_float(v.value)
                        if r > 0:
                            account_rates[v.currency.upper()] = r

                for ccy in needed - {base_currency()}:
                    if ccy in account_rates:
                        fx_rates[ccy] = {"rate": account_rates[ccy],
                                         "source": "IBKR account ExchangeRate",
                                         "at": time.strftime("%Y-%m-%d %H:%M")}
                        continue
                    rate, src = await _fx_from_market(ccy)
                    if rate:
                        fx_rates[ccy] = {"rate": rate, "source": src,
                                         "at": time.strftime("%Y-%m-%d %H:%M")}
                        logger.info(f"FX {ccy}/{base_currency()} = {rate:.5f} ({src})")
                    elif ccy not in fx_rates:
                        logger.warning(
                            f"No {ccy}/{base_currency()} rate available — positions in "
                            f"{ccy} cannot be weighted against {base_currency()} ones"
                        )
                connected_once = True
        except Exception as e:
            logger.error(f"FX refresh failed: {e}", exc_info=e)
        # This task starts before the gateway is up. Retry quickly until there is
        # a connection, or the first run lands on a disconnected client and then
        # sleeps for ten minutes with no rate at all.
        await asyncio.sleep(FX_REFRESH_INTERVAL if connected_once else FX_RETRY_INTERVAL)


def to_base(value, ccy):
    """(converted value, rate info) or (None, None) when no rate is known."""
    ccy = (ccy or base_currency()).upper()
    # A value already in the base currency needs no conversion. Returning None
    # here made every caller report "no FX rate" for the one currency that can
    # never need one — fx_rates holds no identity entry because the base is
    # detected at runtime, and the refresher correctly never fetches it.
    if ccy == base_currency():
        return value, {"rate": 1.0, "source": "base currency", "at": None}
    hit = fx_rates.get(ccy)
    if not hit:
        return None, None
    return value * hit["rate"], hit


def portfolio_weights():
    """Open positions as analytics wants them, off the same resolver as the API.

    Market value rather than share count: 5 shares of a $1,800 stock is not the
    same position as 5 shares of an $18 one.
    """
    out = []
    for p in resolve_positions():
        mv = safe_float(p["marketValue"])
        # A stale or missing quote must not make a real position disappear from
        # the weighting — fall back to cost basis, flagged as an estimate.
        estimated = mv == 0.0
        if estimated:
            mv = abs(safe_float(p["position"]) * safe_float(p["avgCost"]))

        ccy = (p["currency"] or base_currency()).upper()
        converted, rate = to_base(mv, ccy)
        out.append({
            "symbol": p["symbol"],
            # Everything in one currency, so weights are shares of a real total.
            "market_value": converted if converted is not None else mv,
            "currency": ccy,
            "base_currency": base_currency(),
            "converted": bool(rate and ccy != base_currency()),
            "fx_rate": rate["rate"] if (rate and ccy != base_currency()) else None,
            "fx_source": rate["source"] if (rate and ccy != base_currency()) else None,
            "no_fx": converted is None,
            "estimated": estimated,
        })
    return out


# --- ANALYTICS (optional module) ---
'''
Mounted defensively: analytics needs numpy and its own database, and neither should be
able to stop the trading server from starting. If the import or setup fails, the dashboard runs exactly as before, minus /api/analytics/*.
'''
ANALYTICS_ENABLED = False
analytics_backfill = None


# No-op default so _shutdown can call this unconditionally even when the analytics
# import below fails — the whole point of mounting analytics defensively.
def close_analytics_data_connection():
    pass


try:
    from analytics.routes import (
        attach as attach_analytics,
    )
    from analytics.routes import (
        backfill_new_symbol as analytics_backfill,
    )
    from analytics.routes import (
        close_data_connection as close_analytics_data_connection,
    )
    from analytics.routes import (
        router as analytics_router,
    )

    attach_analytics(ib, lambda: PORTFOLIO, resolve_contract, portfolio_weights,
                     lambda sym: price_magnifiers.get(str(sym).upper(), 1),
                     lambda sym: (latest_prices.get(str(sym).upper()) or {}).get("currency"))
    app.include_router(analytics_router)
    ANALYTICS_ENABLED = True
    logger.info("Analytics enabled at /api/analytics/*")
except Exception as e:
    logger.warning(f"Analytics module unavailable ({type(e).__name__}: {e}) — trading unaffected")


# --- DEBUG ENDPOINTS ---
debug_router = APIRouter(prefix="/api/debug", tags=["debug"])

@debug_router.get("/account-values")
async def debug_account_values():
    if not ib.isConnected():
        return {"connected": False, "values": []}

    values = ib.accountValues()
    managed = ib.managedAccounts()
    raw = [{"tag": v.tag, "value": v.value, "currency": v.currency, "account": v.account} for v in values]
    return {
        "connected": True,
        "managed_accounts": managed,
        "total_values": len(values),
        "values": raw
    }

@debug_router.get("/subscriptions")
async def debug_subscriptions():
    # Diagnostic: shows active contracts, ib.tickers() state, and latest_prices to trace data flow.
    if not ib.isConnected():
        return {"connected": False, "message": "Not connected to IBKR"}

    # What contracts we think we've subscribed to
    tracked = {sym: {"conId": c.conId, "exchange": c.primaryExchange} for sym, c in active_contracts.items()}

    # What ib_async actually has internally (the ground truth)
    ib_ticker_data = {}
    for t in ib.tickers():
        sym = t.contract.symbol
        ib_ticker_data[sym] = {
            "conId": t.contract.conId,
            "last": safe_float(t.last),
            "marketPrice": safe_float(t.marketPrice()),
            "close": safe_float(t.close),
            "bid": safe_float(t.bid),
            "ask": safe_float(t.ask),
            "delayedLast": safe_float(getattr(t, 'delayedLast', 0.0)),
        }

    return {
        "connected": True,
        "portfolio_list": PORTFOLIO,
        "active_contracts": tracked,
        "ib_tickers_count": len(ib.tickers()),
        "ib_tickers": ib_ticker_data,
        "latest_prices": latest_prices,
    }

@app.get("/api/account")
async def get_account_summary(account: str = None):
    """Returns NetLiquidation, BuyingPower, CashBalance, AccountCode and live account flag."""
    if not ib.isConnected():
        return {
            "connected": False,
            "net_liquidation": 0.0,
            "buying_power": 0.0,
            "cash_balance": 0.0,
            "account_code": "N/A",
            "is_live": False,
            "accounts": []
        }
    try:
        managed_accounts = ib.managedAccounts()
        if not account and managed_accounts:
            account = managed_accounts[0]

        values = ib.accountValues()
        if not account:
            for v in values:
                if v.account:
                    account = v.account
                    break

            # Fallback to managed accounts
            if not account:
                if managed_accounts:
                    account = managed_accounts[0]

        active_account = account or "N/A"

        account_values = [v for v in values if v.account == active_account]

        summary = {
            "connected": True,
            "net_liquidation": 0.0,
            "buying_power": 0.0,
            "cash_balance": 0.0,
            "account_code": active_account,
            "is_live": False,
            "accounts": managed_accounts,
            "currency": "USD"  # placeholder; the real value is set below
        }

        # Account codes starting with DU or DF are paper accounts, others are live.
        summary["is_live"] = not (
            active_account.startswith("DU") or active_account.startswith("DF")
        ) and active_account != "N/A"

        net_ccy = cash_ccy = bp_ccy = None

        for v in account_values:
            tag_lower = v.tag.lower()
            ccy = v.currency if (v.currency and v.currency != "BASE") else None

            # Largest non-zero value wins per group, ignoring the 'currency' suffix variations.
            if tag_lower in ["netliquidation", "netliquidationvalue", "equitywithloanvalue"]:
                val = safe_float(v.value)
                if val > summary["net_liquidation"]:
                    summary["net_liquidation"], net_ccy = val, ccy
            elif tag_lower in ["buyingpower"]:
                val = safe_float(v.value)
                if val > summary["buying_power"]:
                    summary["buying_power"], bp_ccy = val, ccy
            elif tag_lower in ["cashbalance", "totalcashvalue", "totalcashbalance", "fullcashvalue"]:
                val = safe_float(v.value)
                if val > summary["cash_balance"]:
                    summary["cash_balance"], cash_ccy = val, ccy

        summary["currency"] = net_ccy or cash_ccy or bp_ccy or "USD"

        # Post-process fallbacks if some tags aren't returned (e.g. on standard cash accounts)
        if summary["net_liquidation"] == 0.0 and summary["cash_balance"] > 0.0:
            summary["net_liquidation"] = summary["cash_balance"]
        if summary["buying_power"] == 0.0 and summary["cash_balance"] > 0.0:
            summary["buying_power"] = summary["cash_balance"]

        return summary
    except Exception as e:
        return {"status": "error", "message": str(e)}

def cost_per_share(symbol, raw_avg_cost, qty, market_value=None, unrealized=None):
    """Average cost per share in whole currency units.

    IBKR's averageCost is inconsistent across instrument types — for some it is
    per share in currency, for others it carries the price magnifier. Rather than
    pick a convention and hope, derive the answer from IBKR's own arithmetic:
        unrealized = (market value) - (cost basis)
    so cost basis = market value - unrealized, and per-share follows. The raw
    value is only used when that cross-check is unavailable.
    """
    raw = safe_float(raw_avg_cost)
    if qty and market_value is not None and unrealized is not None and market_value:
        derived = (market_value - unrealized) / qty
        if derived > 0:
            # Flag a real disagreement; a cent of rounding is not one.
            if raw and abs(derived - raw) > max(0.02 * abs(derived), 0.01):
                logger.info(
                    f"{symbol}: IBKR averageCost {raw:,.4f} but cost basis implies "
                    f"{derived:,.4f} per share (ratio {raw / derived:.4g}) — using the "
                    f"derived figure"
                )
            return derived
    # No cross-check possible: fall back to the magnifier assumption.
    return in_currency(symbol, raw)


def resolve_positions(account=None):
    """Open positions with live prices and market value. Never raises.

    Shared by /api/positions and the analytics holdings weights, so both see the
    same book — including the ib.portfolio() -> ib.positions() fallback, which an
    independent reimplementation silently missed.
    """
    if not ib.isConnected():
        return []
    try:
        # ib.portfolio() is populated by reqAccountUpdates and includes market value + PnL
        portfolio_items = ib.portfolio()

        # Fallback: if portfolio is empty, try ib.positions()
        if not portfolio_items:
            positions = ib.positions()
            logger.debug(f"portfolio() empty, positions() returned {len(positions)} items")
            res = []
            for p in positions:
                if account and p.account != account:
                    continue
                symbol = p.contract.symbol
                price_data = latest_prices.get(symbol, {"price": 0.0})
                # latest_prices holds QUOTED units (pence on many LSE lines).
                # Multiplying that by share count gave a market value 100x too big.
                current_price = in_currency(symbol, safe_float(price_data.get("price", 0.0)))
                avg_cost = cost_per_share(symbol, p.avgCost, p.position)

                market_value = p.position * current_price
                cost_basis = p.position * avg_cost
                unrealized_pnl = market_value - cost_basis if current_price > 0 else 0.0

                res.append({
                    "symbol": symbol,
                    "position": p.position,
                    "avgCost": avg_cost,
                    "currentPrice": current_price,
                    "marketValue": market_value,
                    "unrealizedPnL": unrealized_pnl,
                    "currency": p.contract.currency or "USD"
                })
            return res

        # Primary path: use portfolio() data which has built-in market values
        res = []
        for item in portfolio_items:
            if account and item.account != account:
                continue
            symbol = item.contract.symbol
            price_data = latest_prices.get(symbol, {"price": 0.0})
            # Both sources quote in the contract's units; marketValue does not.
            quoted = safe_float(price_data.get("price", 0.0)) or safe_float(item.marketPrice)
            current_price = in_currency(symbol, quoted)
            qty = safe_float(item.position)
            mv = safe_float(item.marketValue)

            # IBKR's own marketValue is the ground truth for units. If price x qty
            # disagrees with it, the magnifier assumption is wrong — say so rather
            # than silently publishing a figure that is out by a factor of 100.
            if qty and current_price and mv:
                implied = current_price * qty
                if abs(implied - mv) > max(0.02 * abs(mv), 0.01):
                    logger.warning(
                        f"{symbol}: price x qty = {implied:,.2f} but IBKR reports "
                        f"marketValue {mv:,.2f} (ratio {implied / mv:.4g}, "
                        f"magnifier {price_magnifiers.get(symbol, 1)}) — using IBKR's"
                    )
                    current_price = mv / qty

            res.append({
                "symbol": symbol,
                "position": qty,
                "avgCost": cost_per_share(symbol, safe_float(item.averageCost), qty,
                                          market_value=mv,
                                          unrealized=safe_float(item.unrealizedPNL)),
                "currentPrice": current_price,
                "marketValue": mv,
                "unrealizedPnL": safe_float(item.unrealizedPNL),
                "currency": item.contract.currency or "USD"
            })
        return res
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        return []


def with_base_values(rows):
    """Adds base-currency twins so positions in different currencies can be summed.

    The native figures stay untouched: a GBP holding's avg cost IS in pounds, and
    rewriting it would disagree with the broker statement.
    """
    out = []
    for p in rows:
        ccy = (p.get("currency") or base_currency()).upper()
        mv_base, rate = to_base(safe_float(p.get("marketValue")), ccy)
        pnl_base, _ = to_base(safe_float(p.get("unrealizedPnL")), ccy)
        out.append({
            **p,
            "baseCurrency": base_currency(),
            "marketValueBase": mv_base,
            "unrealizedPnLBase": pnl_base,
            "fxRate": rate["rate"] if (rate and ccy != base_currency()) else None,
            "fxSource": rate["source"] if (rate and ccy != base_currency()) else None,
        })
    return out


@app.get("/api/positions")
async def get_positions(account: str = None):
    return with_base_values(resolve_positions(account))


@debug_router.get("/positions")
async def debug_positions():
    """Debug endpoint: dumps raw positions and portfolio data."""
    if not ib.isConnected():
        return {"connected": False}

    positions = ib.positions()
    portfolio = ib.portfolio()

    return {
        "connected": True,
        "positions_count": len(positions),
        "positions": [{"symbol": p.contract.symbol, "pos": p.position, "avgCost": p.avgCost, "account": p.account} for p in positions],
        "portfolio_count": len(portfolio),
        "portfolio": [{"symbol": i.contract.symbol, "pos": i.position, "avgCost": i.averageCost, "mktValue": i.marketValue, "pnl": i.unrealizedPNL, "account": i.account} for i in portfolio]
    }


# Mount last: include_router() copies the routes registered so far, so this has to run after every @debug_router route above has been defined.
if DEBUG_API_ENABLED:
    app.include_router(debug_router)
    logger.warning(
        "DASHBOARD_DEBUG is set — /api/debug/* is exposed WITHOUT authentication. "
        "Unset it when you are done diagnosing."
    )
else:
    logger.info("Debug API disabled (set DASHBOARD_DEBUG=1 to enable /api/debug/*)")


# --- TRADING / ORDERS ENDPOINTS ---
def account_funds(account=None):
    """(available, currency) for the pre-trade check, or (None, None) if unknown.

    Prefers AvailableFunds — what can be spent right now — over BuyingPower, which on
    a margin account is a multiple of equity and would wave through an order the cash
    cannot settle.
    """
    if not ib.isConnected():
        return None, None
    managed = ib.managedAccounts() or []
    target = account or (managed[0] if managed else None)
    if not target:
        return None, None

    rows = [v for v in ib.accountValues() if v.account == target]
    for tag in ("availablefunds", "buyingpower", "totalcashvalue", "cashbalance"):
        best, ccy = None, None
        for v in rows:
            if v.tag.lower() != tag:
                continue
            c = v.currency if (v.currency and v.currency != "BASE") else None
            val = safe_float(v.value)
            if best is None or val > best:
                best, ccy = val, c
        if best is not None:
            return best, ccy
    return None, None


# How long to wait for IBKR to accept or reject a newly placed order.
ORDER_ACK_TIMEOUT = 4.0
ORDER_POLL_INTERVAL = 0.25

class OrderRequest(BaseModel):
    symbol: str
    action: str        # "BUY" or "SELL"
    order_type: str    # "LMT" only — market orders are not accepted
    quantity: int
    limit_price: float | None = None
    account: str | None = None  # target account code

@app.post("/api/orders")
async def place_order(req: OrderRequest):
    """Submits a stock order to the IBKR Gateway."""
    if not ib.isConnected():
        return {"status": "error", "message": "IBKR Gateway is not connected"}

    symbol = req.symbol.upper()
    action = req.action.upper()
    order_type = req.order_type.upper()

    if action not in ["BUY", "SELL"]:
        return {"status": "error", "message": "Action must be BUY or SELL"}

    # Rejected rather than silently converted to a limit order: a stale browser tab
    # still sending MKT must fail loudly, not have its intent quietly reinterpreted.
    if order_type != "LMT":
        return {
            "status": "error",
            "message": (
                f"Order type '{order_type}' is not accepted — this dashboard places "
                f"limit orders only. Market orders are disabled because quotes here "
                f"can be delayed, so the fill price is unknowable at submit time."
            ),
        }

    if req.quantity <= 0:
        return {"status": "error", "message": "Quantity must be positive"}

    try:
        # Resolve contract from PORTFOLIO or create a qualified one
        contract = active_contracts.get(symbol)
        if not contract:
            # Reject unknown symbols before registering them
            contract, error = await resolve_contract(symbol)
            if contract is None:
                return {"status": "error", "message": error}

            # Start tracking its market data as well
            PORTFOLIO.append(symbol)
            latest_prices[symbol] = new_quote_entry(contract.currency)
            active_contracts[symbol] = contract
            await record_magnifier(contract)
            ib.reqMktData(contract, '', snapshot=False, regulatorySnapshot=False)
            # Trading a symbol adds it to the watchlist, so it persists like any
            # other addition rather than vanishing on the next restart.
            autosave_watchlist(f"trading {symbol}")

        if req.limit_price is None or req.limit_price <= 0:
            return {"status": "error", "message": "A limit price above 0 is required"}

        # --- pre-trade funds check (BUY only: a SELL reduces exposure, and IBKR
        # rejects a short on a cash account anyway) ---
        if action == "BUY":
            notional = req.quantity * req.limit_price
            available, funds_ccy = account_funds(req.account)
            order_ccy = contract.currency or "USD"

            if available is not None and funds_ccy == order_ccy:
                # Same currency, so the comparison is plain arithmetic — refuse.
                if notional > available:
                    return {
                        "status": "error",
                        "message": (
                            f"Order needs {notional:,.2f} {order_ccy} but only "
                            f"{available:,.2f} {funds_ccy} is available. Reduce the "
                            f"quantity or the limit price."
                        ),
                    }
            elif available is not None and funds_ccy:
                # Different currencies, and IBKR exposes no exchange rate here, so any
                # comparison would be invented. Record both and let IBKR rule on it.
                logger.warning(
                    f"Pre-trade funds check skipped: order is {notional:,.2f} {order_ccy} "
                    f"but available funds are {available:,.2f} {funds_ccy} — no FX rate"
                )

        # The UI shows and sends whole currency units, but IBKR expects the price
        # in the contract's QUOTED units. For a pence-quoted LSE line that is a
        # factor of 100: sending 27.25 for a £27.25 stock would be read as 27.25p
        # — a buy that never fills, or a sell that fills instantly at 1% of value.
        if symbol not in price_magnifiers:
            await record_magnifier(contract)
        mag = price_magnifiers.get(symbol, 1) or 1
        wire_price = round(req.limit_price * mag, 6)
        if mag != 1:
            logger.info(
                f"{symbol} is quoted in minor units (x{mag}): limit "
                f"{req.limit_price} {contract.currency} -> {wire_price} on the wire"
            )

        order = LimitOrder(action, req.quantity, wire_price)

        if req.account:
            order.account = req.account

        trade = ib.placeOrder(contract, order)
        order_id = trade.order.orderId
        order_errors.pop(order_id, None)  # drop anything stale from a recycled id

        loop = asyncio.get_running_loop()
        deadline = loop.time() + ORDER_ACK_TIMEOUT
        dead_polls = 0

        while loop.time() < deadline:
            await asyncio.sleep(ORDER_POLL_INTERVAL)
            status = trade.orderStatus.status

            if status in (OrderStatus.Submitted, OrderStatus.PreSubmitted, OrderStatus.Filled):
                logger.info(f"Order {order_id} accepted by IBKR (status={status})")
                return {
                    "status": "success",
                    "orderId": order_id,
                    "statusText": status,
                    "message": f"{action} {req.quantity} {symbol} {order_type} accepted by IBKR ({status})"
                }

            if status in (OrderStatus.Cancelled, OrderStatus.ApiCancelled, OrderStatus.Inactive):
                dead_polls += 1
                if dead_polls >= 2:
                    errors = order_errors.get(order_id, [])
                    reason = next(
                        (e for e in errors if e["code"] == 201),
                        next((e for e in reversed(errors) if not is_informational_error(e["code"])), None)
                    )
                    detail = f": {reason['message']}" if reason else " (no reason reported by IBKR)"
                    logger.error(f"Order {order_id} rejected by IBKR{detail}")
                    return {
                        "status": "error",
                        "orderId": order_id,
                        "statusText": status,
                        "message": f"Order rejected by IBKR{detail}"
                    }
            else:
                dead_polls = 0

        # Still unresolved — do NOT claim success; the order may or may not be working.
        logger.warning(f"Order {order_id} unresolved after {ORDER_ACK_TIMEOUT}s (status={trade.orderStatus.status})")
        return {
            "status": "pending",
            "orderId": order_id,
            "statusText": trade.orderStatus.status,
            "message": (
                f"Order {order_id} sent but IBKR has not confirmed it yet "
                f"(status={trade.orderStatus.status}). Check your working orders."
            )
        }
    except Exception as e:
        logger.error(f"place_order failed: {e}")
        return {"status": "error", "message": str(e)}

# How long to wait for IBKR to confirm a cancellation before declaring it failed.
CANCEL_CONFIRM_TIMEOUT = 6.0
CANCEL_POLL_INTERVAL = 0.4


async def broker_open_trades():
    """Open orders straight from IBKR — the only trustworthy source.

    ib.openTrades() reads a local cache that lies in one specific, dangerous way:
    ib_async's error handler flips a trade to Cancelled on ANY order error ("something is
    wrong with the order, cancel it"), and openOrder() never restores orderStatus for a
    trade it already tracks. So a *rejected* cancel permanently hides an order that is
    still working at the broker. Anything IBKR reports here is open by definition, so
    stale done-statuses are repaired on the way through.
    """
    trades = await ib.reqAllOpenOrdersAsync()
    for t in trades:
        if t.orderStatus.status in OrderStatus.DoneStates:
            logger.warning(
                f"Order {t.order.orderId} is OPEN at IBKR but cached locally as "
                f"{t.orderStatus.status!r} — repairing stale status"
            )
            t.orderStatus.status = OrderStatus.Submitted
    return trades


@app.get("/api/orders")
async def get_active_orders(account: str = None):
    """Retrieves list of active working trades, as reported by the broker."""
    if not ib.isConnected():
        return []
    try:
        try:
            trades = await broker_open_trades()
        except Exception as e:
            # Never silently fall back to a source that can hide live orders.
            logger.error(f"reqAllOpenOrders failed, using local cache: {e}")
            trades = ib.openTrades()
        res = []
        for t in trades:
            if account and t.order.account != account:
                continue
            res.append({
                "orderId": t.order.orderId,
                "symbol": t.contract.symbol,
                "action": t.order.action,
                "quantity": t.order.totalQuantity,
                "orderType": t.order.orderType,
                "lmtPrice": t.order.lmtPrice or 0.0,
                "status": t.orderStatus.status
            })
        return res
    except Exception as e:
        logger.error(f"Error fetching active orders: {e}")
        return []


# Cap on rows returned by the history endpoint, so a long-lived store cannot
# hand the browser an unbounded table.
HISTORY_MAX_LIMIT = 1000
HISTORY_DEFAULT_LIMIT = 200


@app.get("/api/orders/history")
async def get_order_history(
    account: str = None,
    symbol: str = None,
    start: str = None,      # ISO date or datetime, inclusive
    end: str = None,        # ISO date or datetime, inclusive
    limit: int = HISTORY_DEFAULT_LIMIT,
):
    """Executed orders and their realized PnL, read from the local store.

    Not gated on ib.isConnected(): this is recorded history, and it stays
    answerable — and correct — while the Gateway is down.
    """
    limit = max(1, min(limit, HISTORY_MAX_LIMIT))
    try:
        # Rejected rather than ignored: silently dropping a bad bound would answer
        # "last month" with the entire history, which reads as real data.
        start_at = trade_history.normalize_bound(start)
        end_at = trade_history.normalize_bound(end, end_of_day=True)
    except ValueError:
        return {
            "orders": [], "totals": {}, "coverage": {}, "limit": limit,
            "error": "start and end must be ISO dates (YYYY-MM-DD) or datetimes",
        }

    try:
        return {
            "orders": trade_history.order_history(
                account=account, symbol=symbol, start=start_at, end=end_at, limit=limit
            ),
            "totals": trade_history.totals(
                account=account, symbol=symbol, start=start_at, end=end_at
            ),
            # Unbounded by date on purpose — it is what the store holds overall,
            # so the UI can say where the recorded history actually begins.
            "coverage": trade_history.coverage(account=account, symbol=symbol),
            "limit": limit,
        }
    except Exception as e:
        logger.error(f"Error reading order history: {e}")
        return {"orders": [], "totals": {}, "coverage": {}, "limit": limit, "error": str(e)}


class FlexImportRequest(BaseModel):
    token: str | None = None     # Flex Web Service token, from the IBKR portal
    query_id: str | None = None  # id of a Flex query that includes a Trades section
    path: str | None = None      # or a report XML already downloaded by hand


# Generous: IBKR queues the report, then ib_async polls once a second until it is
# ready. A large multi-year query really can take a couple of minutes.
FLEX_IMPORT_TIMEOUT = 180.0


@app.post("/api/orders/history/import")
async def import_flex_history(req: FlexImportRequest):
    """Backfills executions that predate this dashboard, from an IBKR Flex report.

    Needed because reqExecutions only ever returns the current trading day, so
    nothing older can be recovered from the live connection at all.
    """
    if not req.path and not (req.token and req.query_id):
        return {
            "status": "error",
            "message": "Provide either a Flex token and query_id, or a path to a report XML.",
        }
    try:
        import flex_import
    except Exception as e:
        return {"status": "error", "message": f"Flex import unavailable: {e}"}

    try:
        # Downloading blocks on urlopen and sleeps between polls, so it runs off
        # the event loop — otherwise every quote and order stalls behind it.
        report = await asyncio.wait_for(
            asyncio.to_thread(
                flex_import.load_report,
                token=req.token, query_id=req.query_id, path=req.path,
            ),
            timeout=FLEX_IMPORT_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return {
            "status": "error",
            "message": (
                f"IBKR did not return the report within {FLEX_IMPORT_TIMEOUT:.0f}s. "
                f"Large queries can take longer — try a shorter date range."
            ),
        }
    except Exception as e:
        logger.error(f"Flex report fetch failed: {e}")
        return {"status": "error", "message": f"Could not fetch the Flex report: {e}"}

    try:
        result = await asyncio.to_thread(flex_import.import_report, report)
        if result.get("status") == "success":
            logger.info(result["message"])
        return result
    except Exception as e:
        logger.error(f"Flex import failed: {e}")
        return {"status": "error", "message": f"Import failed: {e}"}


@app.delete("/api/orders/{order_id}")
async def cancel_order(order_id: int):
    """Cancels a working order and CONFIRMS with the broker before reporting success.

    ib.cancelOrder() only writes the request to the socket — it never waits for IBKR to
    accept it, and it optimistically marks the trade PendingCancel regardless. Reporting
    success off that call alone told the user an order was dead while it was still live
    and fillable. Only the broker's own open-order list settles it.
    """
    if not ib.isConnected():
        return {"status": "error", "message": "IBKR Gateway is not connected"}
    try:
        trade = next((t for t in ib.openTrades() if t.order.orderId == order_id), None)
        if trade is None:
            # May be hidden behind a stale local status — ask the broker before giving up.
            trade = next(
                (t for t in await broker_open_trades() if t.order.orderId == order_id),
                None
            )
        if trade is None:
            return {"status": "error", "message": f"Order {order_id} not found among active orders"}

        perm_id = trade.order.permId
        ib.cancelOrder(trade.order)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + CANCEL_CONFIRM_TIMEOUT
        while True:
            await asyncio.sleep(CANCEL_POLL_INTERVAL)

            if trade.orderStatus.status == OrderStatus.Filled:
                logger.warning(f"Order {order_id} filled before the cancel took effect")
                return {
                    "status": "error",
                    "message": f"Order {order_id} FILLED before it could be cancelled — check your positions"
                }

            still_open = any(
                t.order.orderId == order_id or (perm_id and t.order.permId == perm_id)
                for t in await broker_open_trades()
            )
            if not still_open:
                logger.info(f"Order {order_id} cancellation confirmed by IBKR")
                return {"status": "success", "message": f"Order {order_id} cancelled (confirmed by IBKR)"}

            if loop.time() >= deadline:
                break

        logger.error(f"Order {order_id} STILL WORKING at IBKR after cancel request")
        return {
            "status": "error",
            "message": (
                f"Order {order_id} is STILL WORKING at IBKR — the cancel was not accepted. "
                f"Cancel it directly in TWS/Gateway."
            )
        }
    except Exception as e:
        logger.error(f"cancel_order({order_id}) failed: {e}")
        return {"status": "error", "message": str(e)}


_isin_cache = {}


async def isin_family(symbol, exchange=None, currency=None):
    """(ISIN, every contract carrying it) for a symbol.

    The only reliable way to find the same fund in another currency. LSE gives
    each currency line its own ticker — CMOD is the USD line and CMOP the GBP
    line of one fund — so searching by ticker finds neither the sibling nor
    proof that none exists. The ISIN does.
    """
    key = symbol.upper()
    if key in _isin_cache:
        return _isin_cache[key]
    isin, out = None, []
    try:
        c = active_contracts.get(key)
        probe = c if c else (Stock(key, exchange or "SMART", currency or "USD") if currency else None)
        if probe is None:
            return None, []
        det = await asyncio.wait_for(ib.reqContractDetailsAsync(probe), timeout=10.0)
        if det:
            isin = next((sid.value for sid in (det[0].secIdList or [])
                         if sid.tag.upper() == "ISIN"), None)
        if isin:
            byisin = Stock()
            byisin.secIdType, byisin.secId, byisin.exchange = "ISIN", isin, ""
            alld = await asyncio.wait_for(ib.reqContractDetailsAsync(byisin), timeout=20.0)
            seen = set()
            for d in (alld or []):
                k = d.contract
                sig = (k.symbol, k.currency, k.primaryExchange or k.exchange)
                if sig in seen:
                    continue
                seen.add(sig)
                out.append({"symbol": k.symbol, "currency": k.currency,
                            "exchange": k.primaryExchange or k.exchange,
                            "conId": k.conId, "name": d.longName or ""})
    except Exception as e:
        logger.info(f"ISIN lookup for {symbol} failed: {e}")
    _isin_cache[key] = (isin, out)
    return isin, out


async def base_currency_twin(symbol, currency):
    """The same fund's line in the account currency, if one exists."""
    base = base_currency()
    if (currency or "").upper() == base:
        return None
    isin, fam = await isin_family(symbol)
    if not fam:
        return None
    match = next((f for f in fam
                  if (f["currency"] or "").upper() == base and f["symbol"].upper() != symbol.upper()), None)
    if not match:
        return None
    return {**match, "isin": isin}


@app.get("/api/tickers/{symbol}/listings")
async def get_listings(symbol: str):
    """Every currency/exchange line IBKR knows for a symbol.

    Answers "why is my London ETF priced in dollars" directly: many LSE funds are
    USD-denominated, and the exchange a thing trades on says nothing about the
    currency it is denominated in.
    """
    if not ib.isConnected():
        return {"status": "error", "message": "IBKR Gateway is not connected"}
    symbol = symbol.upper()
    try:
        matches = await asyncio.wait_for(ib.reqMatchingSymbolsAsync(symbol), timeout=10.0)
    except Exception as e:
        return {"status": "error", "message": f"Symbol lookup failed: {e}"}

    exact = [m.contract for m in (matches or []) if m.contract.symbol.upper() == symbol]
    # Sibling tickers matter: LSE usually gives each currency line its OWN ticker
    # (CSPX in USD vs CSP1 in GBP), so an exact-symbol search alone cannot prove
    # a GBP version does not exist.
    siblings = [
        {"symbol": m.contract.symbol, "currency": m.contract.currency,
         "exchange": m.contract.primaryExchange or "", "desc": getattr(m, "description", "") or ""}
        for m in (matches or []) if m.contract.symbol.upper() != symbol
    ]
    listings = []
    for c in exact:
        # Confirm each line really exists rather than trusting the fuzzy search.
        q = await try_qualify(Stock(symbol, c.primaryExchange or "SMART", c.currency))
        # longName is what actually settles whether two tickers are the same fund;
        # a shared ticker prefix is a hint, not evidence.
        name = ""
        if q:
            try:
                det = await asyncio.wait_for(ib.reqContractDetailsAsync(q), timeout=10.0)
                name = (det[0].longName or "") if det else ""
            except Exception:
                pass
        mag = None
        if q:
            try:
                dd = await asyncio.wait_for(ib.reqContractDetailsAsync(q), timeout=10.0)
                mag = getattr(dd[0], "priceMagnifier", None) if dd else None
            except Exception:
                pass
        listings.append({
            "currency": c.currency, "exchange": c.primaryExchange or "SMART",
            "secType": c.secType, "conId": q.conId if q else None,
            "qualifies": bool(q), "name": name,
            # 100 means the feed quotes pence while the currency says GBP.
            "price_magnifier": mag,
        })
    seed = next((l for l in listings if l["qualifies"]), None)
    isin, by_isin = await isin_family(
        symbol, seed["exchange"] if seed else None, seed["currency"] if seed else None)

    active = active_contracts.get(symbol)
    return {
        "status": "ok", "symbol": symbol, "isin": isin, "same_isin": by_isin,
        "in_use": {"currency": active.currency, "exchange": active.primaryExchange,
                   "conId": active.conId} if active else None,
        "preference": currency_preference(),
        "listings": listings,
        "siblings": siblings,
    }


# --- WATCHLIST API ---
@app.get("/api/watchlist")
async def get_watchlist():
    """Current list, what is on disk, and whether they differ."""
    saved = load_saved_watchlist()
    return {
        "status": "ok",
        "symbols": list(PORTFOLIO),
        "saved": saved,
        # Order matters: the watchlist renders in this order, so a reorder is a
        # real unsaved change, not a no-op.
        "dirty": saved is None or saved != list(PORTFOLIO),
        "path": WATCHLIST_FILE,
    }


@app.post("/api/watchlist/save")
async def post_save_watchlist():
    """Persists the current watchlist so it survives a restart."""
    ok, detail = save_watchlist(PORTFOLIO)
    if not ok:
        return {"status": "error", "message": f"Could not save: {detail}"}
    logger.info(f"Watchlist saved: {len(PORTFOLIO)} symbols -> {detail}")
    return {
        "status": "success",
        "count": len(PORTFOLIO),
        "path": detail,
        "message": f"Saved {len(PORTFOLIO)} symbols — this list will load on restart.",
    }


# --- TICKER API ENDPOINTS ---
@app.get("/api/feed/status")
async def get_feed_status():
    session = us_market_session()
    now_et = datetime.now(US_EASTERN)
    last = feed_state["last_quote_at"]
    quote_age = round(time.monotonic() - last, 1) if last else None

    payload = {
        "session": session or "unknown",
        "delayed": feed_state["delayed"],
        "session_time_et": now_et.strftime("%H:%M ET"),
        "quote_age_seconds": quote_age,
        "subscribed_symbols": len(active_contracts),
        # Unknown session takes the lenient threshold: without IBKR's calendar we
        # cannot tell RTH from a holiday, and crying "Feed stalled" is the worse error.
        "stale_after_seconds": FEED_STALE_RTH if session == "open" else FEED_STALE_EXTENDED,
        "schedule_source": market_schedule["source"] or "unavailable",
        # The session the current prices came from, when the market is shut.
        "last_close_day": last_completed_session(now_et),
        "farms": dict(feed_state["farms"]),
        "data_errors": [
            {"code": e["code"], "message": e["message"]} for e in feed_state["data_errors"][-3:]
        ],
    }

    if not ib.isConnected():
        payload.update(
            state="disconnected",
            headline="Disconnected from IB Gateway",
            detail="Market status is unknown while disconnected — it comes from IBKR. "
                   "Use the connection panel to reconnect."
        )
        return payload

    down_farms = [name for name, st in feed_state["farms"].items() if st == "down"]

    # A shut market is checked BEFORE entitlement, because it explains silence more
    # fundamentally. Reversed, a stale entitlement error from hours earlier made every
    # weekend report "No market data subscription" — which is unknowable while closed,
    # since no quote would be flowing either way.
    if session in ("closed", "weekend", "holiday"):
        payload.update(
            state="market_closed",
            headline="US market closed" if session != "holiday" else "US market holiday",
            detail=(
                f"It is {payload['session_time_et']}. Regular hours are 09:30–16:00 ET, "
                f"Mon–Fri. Quotes resume at the open."
            )
        )
        return payload

    if quote_age is None and feed_state["data_errors"]:
        newest = feed_state["data_errors"][-1]
        payload.update(
            state="no_entitlement",
            headline="No market data subscription",
            detail=f"IBKR error {newest['code']}: {newest['message']}"
        )
        return payload

    if quote_age is None and session in ("premarket", "afterhours"):
        payload.update(
            state="market_closed",
            headline=f"US market {'pre-market' if session == 'premarket' else 'after hours'}",
            detail=(
                f"It is {payload['session_time_et']}. Delayed data is generally only served "
                f"during regular hours (09:30–16:00 ET)."
            )
        )
        return payload

    if quote_age is None:
        payload.update(
            state="no_data",
            headline="No quotes received",
            detail=(
                f"Subscribed to {len(active_contracts)} symbols but IBKR has sent nothing"
                + (f". Market data farms not connected: {', '.join(down_farms)}." if down_farms else ".")
            )
        )
        return payload

    if quote_age > payload["stale_after_seconds"]:
        payload.update(
            state="stale",
            headline="Feed stalled",
            detail=f"No quote for {int(quote_age)}s."
                   + (f" Farms not connected: {', '.join(down_farms)}." if down_farms else "")
        )
        return payload

    payload.update(
        state="live",
        headline="Streaming (delayed)" if feed_state["delayed"] else "Streaming",
        detail=(
            f"Last quote {quote_age:.0f}s ago."
            + (" Delayed feed — no real-time subscription, quotes lag the market."
               if feed_state["delayed"] else "")
        )
    )
    return payload

@app.get("/api/prices")
async def get_prices():
    """Returns the most recent IBKR prices from memory, filling gaps with static state."""
    if ib.isConnected():
        for t in ib.tickers():
            symbol = t.contract.symbol
            if symbol in latest_prices:
                # same ladder the tick handler uses, so both writers agree
                apply_quote(symbol, *resolve_quote(t))

        for item in ib.portfolio():
            symbol = item.contract.symbol
            if symbol in latest_prices and latest_prices[symbol]["price"] == 0.0:
                p_price = safe_float(item.marketPrice)
                if p_price > 0.0:
                    latest_prices[symbol]["price"] = p_price

    # age_seconds, not the raw monotonic clock, which is meaningless to a browser.
    now = time.monotonic()
    now_utc = datetime.now(timezone.utc)
    sessions = {sym: symbol_session(sym, now_utc) for sym in latest_prices}
    return {
        symbol: {
            # Divided here, not at ingestion: returns-based analytics are immune
            # to the factor, so only what the user reads as money needs it.
            "price": in_currency(symbol, e["price"]),
            "bid": in_currency(symbol, e["bid"]),
            "ask": in_currency(symbol, e["ask"]),
            "currency": e["currency"],
            "price_magnifier": price_magnifiers.get(symbol, 1),
            "age_seconds": round(now - e["last_tick_at"], 1) if e.get("last_tick_at") else None,
            # This symbol's OWN exchange, not the US session — a London listing
            # trades while NYSE is shut and must not be labelled closed.
            "session": sessions[symbol][0],
            "last_close_day": sessions[symbol][1],
            "exchange": exchange_of(symbol),
            # Non-empty only when IBKR lists this symbol in several currencies.
            "currencies_available": contract_alternatives.get(symbol) or [],
            # A USD-denominated fund on a London exchange is still USD. Rather
            # than mislabel it, show what it is worth in the account's currency.
            "base_currency": base_currency(),
            "base_twin": base_twins.get(symbol),
            "price_base": (to_base(in_currency(symbol, e["price"]), e["currency"])[0]
                           if e["price"] and (e["currency"] or "").upper() != base_currency()
                           else None),
        }
        for symbol, e in latest_prices.items()
    }

@app.post("/api/tickers/{symbol}")
async def add_ticker(symbol: str):
    """Adds a new ticker dynamically to the IBKR market data feed."""
    if not ib.isConnected():
        return {"status": "error", "message": "IBKR Gateway is not connected"}

    symbol = symbol.upper()
    if symbol in latest_prices:
        return {"status": "already tracking"}

    try:
        # Resolve currency/exchange from IBKR rather than assuming a US listing.
        qualified, error = await resolve_contract(symbol)
        if qualified is None:
            return {"status": "error", "message": error}

        # Start tracking in our global dictionaries
        PORTFOLIO.append(symbol)
        latest_prices[symbol] = new_quote_entry(qualified.currency)
        active_contracts[symbol] = qualified

        # Request live market data feed for this new contract
        ib.reqMktData(qualified, '', snapshot=False, regulatorySnapshot=False)

        # Pull daily history in the background so the new ticker gets volatility, beta, and other analytics without blocking the API response.
        if ANALYTICS_ENABLED and analytics_backfill is not None:
            spawn_background(analytics_backfill(symbol), f"analytics-backfill-{symbol}")

        saved = autosave_watchlist(f"adding {symbol}")

        twin = await base_currency_twin(symbol, qualified.currency)
        return {
            "status": "success",
            "symbol": symbol,
            "currency": qualified.currency,
            "exchange": qualified.primaryExchange,
            "currencies_available": contract_alternatives.get(symbol) or [],
            # Same ISIN, your currency, different ticker — the thing a ticker
            # search can never surface on its own.
            "base_currency_twin": twin,
            "saved": saved,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.delete("/api/tickers/{symbol}")
async def delete_ticker(symbol: str):
    """Removes a ticker from the IBKR feed and active prices."""
    if not ib.isConnected():
        return {"status": "error", "message": "IBKR Gateway is not connected"}

    symbol = symbol.upper()
    if symbol not in latest_prices:
        return {"status": "error", "message": "Ticker not tracked"}

    try:
        # Cancel the market data stream. A contract left unqualified by an older build
        contract = active_contracts.get(symbol)
        if contract:
            try:
                if contract.conId:
                    ib.cancelMktData(contract)
            except Exception as e:
                logger.warning(f"cancelMktData failed for {symbol}, removing anyway: {e}")
            active_contracts.pop(symbol, None)

        # Remove from global state
        if symbol in PORTFOLIO:
            PORTFOLIO.remove(symbol)
        if symbol in latest_prices:
            del latest_prices[symbol]

        saved = autosave_watchlist(f"removing {symbol}")
        return {"status": "success", "symbol": symbol, "saved": saved}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# --- STATIC UI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


if __name__ == '__main__':
    util.patchAsyncio()
    import uvicorn

    logger.info(f"Dashboard UI:  http://{SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"IBKR Gateway:  {IBKR_HOST}:{IBKR_PORT} (clientId {IBKR_CLIENT_ID})")
    logger.info(f"Watchlist:     {', '.join(PORTFOLIO)}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
