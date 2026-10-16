"""FastAPI surface for the decision model, plus the IBKR historical-data fetcher.

Wired into server.py through attach(), which injects the live IB handle, a getter
for the mutable watchlist, and server.py's own resolve_contract — the last one
shared rather than reimplemented so analytics inherits non-US symbol support
instead of re-deriving it (and re-introducing the hardcoded-USD bug).
"""

import asyncio
import json
import logging
import os
import time
from datetime import date

import numpy as np

from fastapi import APIRouter
from pydantic import BaseModel

from . import backtest as BT
from . import metrics as M
from . import store
from .benchmarks import (
    BENCHMARK_LABELS,
    MARKET_BENCHMARK,
    all_benchmarks,
    benchmark_for,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

_ib = None
_get_portfolio = lambda: []
_resolve_contract = None
_get_positions = lambda: []
# How many quoted units make one currency unit. 100 for pence-quoted LSE lines.
_get_magnifier = lambda sym: 1
_get_currency = lambda sym: None

# IBKR allows ~60 historical requests per rolling 10 minutes, and separately
# rejects near-identical requests issued within a couple of seconds. A full
# refresh is ~15 requests, so 2s spacing clears both limits with room to spare.
PACING_DELAY = 2.0
BAR_DURATION = "3 Y"      # 3 years: skew needs ~500+ observations to mean anything
IV_DURATION = "1 Y"       # VRP only reads the last ~60 days
HIST_TIMEOUT = 90.0

# Skip a symbol whose newest stored bar is this recent. 4 calendar days clears a
# weekend plus one holiday; a stale read only costs a redundant fetch, so this
# errs toward refetching.
BAR_FRESH_DAYS = 4
# Below this, treat the history as a partial backfill and redo it.
MIN_BARS_FOR_CURRENT = 200

# --- OPTIONAL SEPARATE DATA CONNECTION ---
# Daily bars are account-independent, so a paper login can supply them while trading
# stays on the live gateway. Worth having because some live accounts lack the EOD data
# entitlement and answer every historical request with error 162, while the paper
# login for the same person serves it fine.
#
# Set ANALYTICS_IBKR_PORT=4002 (with a paper Gateway running) to use it. Unset means
# "use the trading connection", i.e. previous behaviour.
DATA_HOST = os.getenv("ANALYTICS_IBKR_HOST", os.getenv("IBKR_HOST", "127.0.0.1"))
DATA_PORT = os.getenv("ANALYTICS_IBKR_PORT")
DATA_CLIENT_ID = int(os.getenv("ANALYTICS_IBKR_CLIENT_ID", "2"))

_data_ib = None
_data_state = {"configured": bool(DATA_PORT), "connected": False, "last_error": None}


async def data_connection():
    """The dedicated data connection, or None to fall back to the trading one."""
    global _data_ib
    if not DATA_PORT:
        return None
    if _data_ib is not None and _data_ib.isConnected():
        return _data_ib

    from ib_async import IB
    if _data_ib is None:
        _data_ib = IB()
    try:
        await asyncio.wait_for(
            _data_ib.connectAsync(DATA_HOST, int(DATA_PORT), clientId=DATA_CLIENT_ID),
            timeout=10.0,
        )
    except Exception as e:
        _data_state.update(connected=False, last_error=str(e) or type(e).__name__)
        # Once per outage, not once per request — a refresh makes ~49 of these.
        if not _data_state.get("warned"):
            logger.warning(
                f"Analytics data connection to {DATA_HOST}:{DATA_PORT} failed "
                f"({_data_state['last_error']}) — falling back to the trading "
                f"connection. Is a paper Gateway running on {DATA_PORT}?"
            )
            _data_state["warned"] = True
        else:
            logger.debug(f"Analytics data connection still down: {_data_state['last_error']}")
        return None

    _data_state.update(connected=True, last_error=None, warned=False)
    logger.info(
        f"Analytics data connection up: {DATA_HOST}:{DATA_PORT} (clientId {DATA_CLIENT_ID})"
    )
    return _data_ib


def close_data_connection():
    """Called from the app's shutdown so clientId {DATA_CLIENT_ID} is released."""
    global _data_ib
    if _data_ib is not None and _data_ib.isConnected():
        try:
            _data_ib.disconnect()
            logger.info("Analytics data connection closed")
        except Exception as e:
            logger.warning(f"Analytics data disconnect was not clean: {e}")
    _data_state["connected"] = False

# One refresh at a time. Two overlapping backfills would double the request rate
# and trip pacing, which IBKR punishes by silently returning empty bar lists.
_refresh_lock = asyncio.Lock()


def attach(ib, get_portfolio, resolve_contract, get_positions=None, get_magnifier=None,
           get_currency=None):
    """Inject the live handles. `get_positions` is optional so analytics still
    mounts on a build that does not supply it — the holdings book just stays null."""
    global _ib, _get_portfolio, _resolve_contract, _get_positions, _get_magnifier
    global _get_currency
    _ib, _get_portfolio, _resolve_contract = ib, get_portfolio, resolve_contract
    if get_positions is not None:
        _get_positions = get_positions
    if get_magnifier is not None:
        _get_magnifier = get_magnifier
    if get_currency is not None:
        _get_currency = get_currency
    store.connect()


# ----------------------------------------------------------------------------
# fetching
# ----------------------------------------------------------------------------
async def _fetch(symbol, what_to_show, duration):
    """One historical request. Returns (bars, error_message)."""
    # Contracts are resolved on the TRADING connection because that is what has the
    # symbol-search plumbing; the conId it returns is account-independent, so the
    # resolved contract is valid on the data connection too.
    if _ib is None or not _ib.isConnected():
        return None, "IBKR gateway not connected"

    contract, error = await _resolve_contract(symbol)
    if contract is None:
        return None, error

    handle = await data_connection() or _ib

    try:
        bars = await asyncio.wait_for(
            handle.reqHistoricalDataAsync(
                contract, endDateTime="", durationStr=duration,
                barSizeSetting="1 day", whatToShow=what_to_show,
                useRTH=True, formatDate=1,
            ),
            timeout=HIST_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return None, f"timed out after {HIST_TIMEOUT:.0f}s"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

    if not bars:
        # IBKR answers both a pacing violation AND a missing entitlement with an empty
        # list, putting the real reason (commonly error 162) only on the error stream.
        where = f"{DATA_HOST}:{DATA_PORT}" if handle is _data_ib else "the trading connection"
        return None, (
            f"IBKR returned no bars via {where} — check the server log for the reason "
            f"(error 162 means this login has no historical-data entitlement)"
        )
    return bars, None


def _is_current(last_day, today=None):
    """True if last_day is recent enough that a refetch would return the same bars."""
    if not last_day:
        return False
    try:
        stored = date.fromisoformat(last_day)
    except (TypeError, ValueError):
        return False
    return ((today or date.today()) - stored).days <= BAR_FRESH_DAYS


def needs_bars(symbol, force=False):
    return (force
            or store.bar_count(symbol) < MIN_BARS_FOR_CURRENT
            or not _is_current(store.last_bar_day(symbol)))


def needs_iv(symbol, force=False):
    return force or not _is_current(store.last_vol_day(symbol, "IV"))


async def backfill_symbol(symbol, include_iv=True):
    """Price bars, plus IBKR's implied-vol series for watchlist names."""
    symbol = symbol.upper()
    bars, error = await _fetch(symbol, "TRADES", BAR_DURATION)
    if error:
        return {"symbol": symbol, "error": error}
    n = store.save_bars(symbol, bars)

    iv_n = 0
    if include_iv:
        await asyncio.sleep(PACING_DELAY)
        iv_bars, iv_err = await _fetch(symbol, "OPTION_IMPLIED_VOLATILITY", IV_DURATION)
        if iv_err:
            # Not fatal: a name with no listed options still scores on pillars 1-3.
            logger.info(f"No implied-vol series for {symbol}: {iv_err}")
        else:
            iv_n = store.save_vol(symbol, iv_bars, "IV")

    logger.info(f"Backfilled {symbol}: {n} bars, {iv_n} IV points")
    return {"symbol": symbol, "error": None, "bars": n, "iv_points": iv_n}


async def backfill_new_symbol(symbol):
    """Called by server.py's add_ticker so a new name gets risk numbers on its own."""
    try:
        async with _refresh_lock:
            return await backfill_symbol(symbol)
    except Exception as e:
        logger.error(f"Backfill failed for {symbol}: {e}", exc_info=e)
        return {"symbol": symbol, "error": str(e)}


# ----------------------------------------------------------------------------
# computation helpers
# ----------------------------------------------------------------------------
def _months_back(months, today=None):
    """ISO date `months` calendar months before today."""
    end = today or date.today()
    y, m = end.year, end.month - months
    while m <= 0:
        m += 12
        y -= 1
    import calendar as _cal
    return date(y, m, min(end.day, _cal.monthrange(y, m)[1])).isoformat()


def watch_universe():
    """Watchlist plus anything held — the names every analytic should cover.

    A position outside the watchlist is still a position: leaving it out gave it
    no risk row, no refresh and no price, each of which had to be fixed
    separately before this existed.
    """
    out = [s.upper() for s in _get_portfolio()]
    try:
        for pos in _get_positions() or []:
            sym = str(pos.get("symbol", "")).upper()
            if sym and sym not in out:
                out.append(sym)
    except Exception as e:
        logger.warning(f"Could not add held positions to the analytics universe: {e}")
    return out


def resolve_window(key):
    """(key, ISO cutoff) for a requested window, falling back to the default.

    Anchored on TODAY, not on each symbol's newest bar: a window that ends wherever
    a symbol's data happens to stop is not the same window for two symbols, and the
    whole point of these is comparing names on equal terms. A symbol whose history
    is stale therefore shows fewer bars in the window, which is the honest reading.
    """
    key = (key or M.DEFAULT_WINDOW).lower()
    if key not in M.WINDOW_MONTHS:
        key = M.DEFAULT_WINDOW
    return key, M.window_start(date.today().isoformat(), key)


# Roughly 21 trading days a month. Used only to tell two very different failures
# apart: a window too short to ever hold enough sessions, versus history that is
# merely stale. Sending someone to Refresh history when refreshing cannot help is
# worse than saying nothing.
SESSIONS_PER_MONTH = 21


def window_capacity(key):
    """About how many sessions a window can hold at best."""
    return M.WINDOW_MONTHS.get(key, 36) * SESSIONS_PER_MONTH


def shortfall_hint(key, needed):
    """Why a statistic could not be computed, and what would actually fix it."""
    if window_capacity(key) < needed:
        return (f"a {M.WINDOW_LABELS[key]} window holds about {window_capacity(key)} "
                f"sessions and this needs {needed} — choose a longer window")
    return "history is stale — use Refresh history"


def _slice(days, values, start):
    """The tail of a (days, values) pair from `start` onward."""
    if not start:
        return days, values
    for i, d in enumerate(days):
        if d >= start:
            return days[i:], values[i:]
    return [], values[:0]


def _lead_in(days, start):
    """Index to slice from: the window, plus one bar before it.

    The extra bar is what makes the window's FIRST return a real day-over-day move
    instead of a gap across the boundary. Every series in a comparison has to use
    this same rule — mismatched lengths are what silently null out beta and IR.
    Returns (slice_index, bars_inside_window), or (None, 0) if nothing qualifies.
    """
    if not start:
        return 0, len(days)
    first = next((i for i, d in enumerate(days) if d >= start), None)
    if first is None:
        return None, 0
    return max(0, first - 1), len(days) - first


def _window_bars(bars, start):
    """(bars with their lead-in, count actually inside the window)."""
    lo, inside = _lead_in([b["day"] for b in bars], start)
    return ([] if lo is None else bars[lo:]), inside


def _returns_for(symbol, start=None):
    days, closes = store.load_closes(symbol)
    lo, _ = _lead_in(days, start)
    if lo is None:
        return [], M.to_returns([])
    days, closes = days[lo:], closes[lo:]
    return days[1:], M.to_returns(closes)


def _window_meta(key, start, bars, stored_first, inside=None):
    """What this symbol's window actually covered, for honest labelling."""
    first = bars[0]["day"] if bars else None
    # Truncated = the stored history begins after the requested cutoff, so these
    # numbers are "everything we have" rather than the full window asked for.
    truncated = bool(start and stored_first and stored_first > start)
    return {
        "window": key,
        "window_label": M.WINDOW_LABELS.get(key, key),
        "window_start": start,
        "window_first_day": first,
        # Bars inside the window; `bars` carries one extra lead-in bar on top.
        "window_bars": len(bars) if inside is None else inside,
        "window_truncated": truncated,
    }


def _scorecards(window_key=None):
    """Runs the model for every watchlist symbol over one window. Never raises."""
    key, start = resolve_window(window_key)
    portfolio = watch_universe()
    mkt_days, mkt_r = _returns_for(MARKET_BENCHMARK, start)

    rows = []
    for sym in portfolio:
        stored = store.load_bars(sym)
        stored_first = stored[0]["day"] if stored else None
        bars, inside = _window_bars(stored, start)
        meta = _window_meta(key, start, bars, stored_first, inside)

        if len(bars) < M.MIN_SERIES + 1:
            if not stored:
                err = "no stored history — use Refresh history"
            elif window_capacity(key) < M.MIN_SERIES + 1:
                err = f"a {meta['window_label']} window is too short for any estimate"
            else:
                err = (f"only {len(bars)} bars in the last {meta['window_label']} "
                       f"(history ends {stored[-1]['day']}) — use Refresh history")
            rows.append({"symbol": sym, "error": err, **meta})
            continue

        etf, is_sector = benchmark_for(sym)
        sym_days, sym_r = _returns_for(sym, start)
        b_days, b_r = _returns_for(etf, start)

        # Align on dates so a later listing date can never shift the series.
        a_r, bench_r = M.align(sym_days, sym_r, b_days, b_r)
        # Second alignment against the market leg: the sector overlap and the SPY
        # overlap are not the same set of days when listing dates differ.
        a_mkt, mkt_al = M.align(sym_days, sym_r, mkt_days, mkt_r)
        # The IV leg needs the same slice: comparing 60 days of implied vol against
        # 20 days of realized would not be a variance risk premium.
        iv_days, iv_vals = store.load_vol(sym, "IV")
        _, iv = _slice(iv_days, iv_vals, start)

        card = M.build_scorecard(
            sym, bars, bench_r, etf, is_sector, iv,
            window=meta,
            aligned={"bench": (a_r, bench_r), "market": (a_mkt, mkt_al)},
        )
        # The gates were computed on the aligned overlap; say how big that was.
        card["benchmark_overlap_days"] = int(a_r.size)
        card["benchmark_label"] = BENCHMARK_LABELS.get(etf, etf)
        rows.append(card)
    return rows


# ----------------------------------------------------------------------------
# endpoints
# ----------------------------------------------------------------------------
@router.get("/windows")
async def get_windows():
    """The lookback windows the Risk tab offers, and today's cutoff for each."""
    today = date.today().isoformat()
    return {
        "status": "ok",
        "default": M.DEFAULT_WINDOW,
        "windows": [
            {"key": k, "label": M.WINDOW_LABELS[k],
             "months": M.WINDOW_MONTHS[k], "start": M.window_start(today, k)}
            for k in M.WINDOW_MONTHS
        ],
        "min_tail_sample": M.MIN_TAIL_SAMPLE,
        "min_series": M.MIN_SERIES,
    }


@router.get("/metrics")
async def get_metrics(window: str = None):
    """Per-symbol risk stats and the four-pillar verdict, over one lookback window."""
    try:
        key, start = resolve_window(window)
        return {"status": "ok", "window": key, "window_start": start,
                "window_label": M.WINDOW_LABELS[key], "metrics": _scorecards(key)}
    except Exception as e:
        logger.error(f"/metrics failed: {e}", exc_info=e)
        return {"status": "error", "message": str(e), "metrics": []}


@router.get("/scorecard")
async def get_scorecard(window: str = None):
    """The decision model, with the thresholds that produced each verdict."""
    try:
        key, start = resolve_window(window)
        return {
            "status": "ok",
            "window": key,
            "window_start": start,
            "window_label": M.WINDOW_LABELS[key],
            "scorecards": _scorecards(key),
            "thresholds": {
                "sortino_target": M.SORTINO_TARGET,
                "info_ratio_target": M.INFO_RATIO_TARGET,
                "skew_floor": M.SKEW_FLOOR,
                "cvar_limit_pct": M.CVAR_LIMIT_PCT,
                "vrp_min_points": M.VRP_MIN_POINTS,
                "vrp_max_points": M.VRP_MAX_POINTS,
            },
            "pillar_labels": {
                "momentum": "Momentum & drift",
                "efficiency": "Risk-adjusted efficiency",
                "tail_risk": "Distribution & tail risk",
                "sentiment": "Implied sentiment (VRP)",
            },
        }
    except Exception as e:
        logger.error(f"/scorecard failed: {e}", exc_info=e)
        return {"status": "error", "message": str(e), "scorecards": []}


# --- WATCHLIST GROUPS ---
# benchmarks.THEMES is the seed. Once the user edits anything the complete config
# is written to the store and becomes the single source of truth — otherwise a
# rename would be silently resurrected by the defaults underneath it.
THEMES_META_KEY = "watchlist_themes"
OTHER_GROUP = "Other"


def default_theme_config():
    from .benchmarks import THEMES
    return {
        "groups": [name for name, _ in THEMES],
        "assign": {sym: name for name, syms in THEMES for sym in syms},
    }


def load_theme_config():
    """Stored config if the user has edited, else the built-in defaults."""
    raw = store.get_meta(THEMES_META_KEY)
    if not raw:
        return default_theme_config()
    try:
        cfg = json.loads(raw)
        groups = [str(g) for g in cfg.get("groups") or []]
        assign = {str(k).upper(): str(v) for k, v in (cfg.get("assign") or {}).items()}
        if not groups:
            return default_theme_config()
        # Drop assignments pointing at a group that no longer exists, so a deleted
        # group cannot leave symbols stranded in an invisible bucket.
        assign = {k: v for k, v in assign.items() if v in groups}
        return {"groups": groups, "assign": assign}
    except Exception as e:
        logger.warning(f"Stored watchlist groups unreadable, using defaults: {e}")
        return default_theme_config()


def save_theme_config(cfg):
    store.set_meta(THEMES_META_KEY, json.dumps(cfg))


def theme_of(symbol, cfg=None):
    cfg = cfg or load_theme_config()
    return cfg["assign"].get(symbol.upper(), OTHER_GROUP)


def grouped(symbols, cfg=None):
    """[(group, [symbols])] in the config's group order, 'Other' last."""
    cfg = cfg or load_theme_config()
    seen = [s.upper() for s in symbols]
    out = []
    for g in cfg["groups"]:
        hit = [s for s in seen if cfg["assign"].get(s) == g]
        if hit:
            out.append((g, hit))
    leftover = [s for s in seen if s not in cfg["assign"]]
    if leftover:
        out.append((OTHER_GROUP, leftover))
    return out


@router.get("/themes")
async def get_themes():
    cfg = load_theme_config()
    return {"status": "ok", "customised": bool(store.get_meta(THEMES_META_KEY)), **cfg}


class ThemeAssign(BaseModel):
    symbol: str
    group: str


@router.post("/themes/assign")
async def assign_theme(req: ThemeAssign):
    """Puts a symbol in a group, creating the group if it is new."""
    sym = req.symbol.strip().upper()
    group = req.group.strip()
    if not sym or not group:
        return {"status": "error", "message": "Both a symbol and a group are required."}

    cfg = load_theme_config()
    if group not in cfg["groups"]:
        # New group goes before 'Other' so the catch-all stays at the bottom.
        at = len(cfg["groups"])
        if OTHER_GROUP in cfg["groups"]:
            at = cfg["groups"].index(OTHER_GROUP)
        cfg["groups"].insert(at, group)
    cfg["assign"][sym] = group
    save_theme_config(cfg)
    logger.info(f"Watchlist group: {sym} -> {group}")
    return {"status": "ok", **cfg}


class ThemeRename(BaseModel):
    old: str
    new: str


@router.post("/themes/rename")
async def rename_theme(req: ThemeRename):
    """Renames a group, carrying its members with it."""
    old, new = req.old.strip(), req.new.strip()
    if not old or not new:
        return {"status": "error", "message": "Both names are required."}
    cfg = load_theme_config()
    if old not in cfg["groups"]:
        return {"status": "error", "message": f"No group called {old!r}."}
    if new != old and new in cfg["groups"]:
        return {"status": "error", "message": f"A group called {new!r} already exists."}

    cfg["groups"] = [new if g == old else g for g in cfg["groups"]]
    cfg["assign"] = {k: (new if v == old else v) for k, v in cfg["assign"].items()}
    save_theme_config(cfg)
    logger.info(f"Watchlist group renamed: {old!r} -> {new!r}")
    return {"status": "ok", **cfg}


# A hand-picked matrix is for reading, not for dumping the whole watchlist into.
# Past this many rows the cells are too small to compare anyway.
MAX_CORRELATION_SYMBOLS = 20


def parse_symbol_list(raw):
    """'nvda, amd ,NVDA' -> ['NVDA', 'AMD'], order kept, capped."""
    if not raw:
        return []
    out = []
    for part in raw.replace(" ", ",").split(","):
        sym = part.strip().upper()
        if sym and sym not in out:
            out.append(sym)
    return out[:MAX_CORRELATION_SYMBOLS]


@router.get("/correlation")
async def get_correlation(window: str = None, symbols: str = None):
    """Pairwise correlation, across the watchlist or a chosen subset.

    `symbols` is a comma-separated list; empty means the whole watchlist. The
    subset is what makes a 21x21 grid readable — the user picks the few names
    they actually want to compare.

    Always returns `symbols` and `matrix` keys, even on failure: the UI indexes
    matrix[i][j] directly, so an error payload missing those fields would crash
    the render rather than show the message.
    """
    empty = {"symbols": [], "matrix": [], "average_pairwise": None,
             "observations": 0, "excluded": [], "unknown": [], "custom": False}
    try:
        key, start = resolve_window(window)
        empty = {**empty, "window": key, "window_label": M.WINDOW_LABELS[key]}
        requested = parse_symbol_list(symbols)
        custom = bool(requested)
        empty["custom"] = custom

        if custom:
            # Anything with stored bars is fair game, not just the watchlist —
            # a benchmark ETF is often exactly what you want to compare against.
            known = set(store.stored_symbols())
            unknown = [s for s in requested if s not in known]
            universe = [s for s in requested if s in known]
            empty["unknown"] = unknown
        else:
            unknown = []
            universe = watch_universe()

        if custom and len(universe) < 2:
            return {**empty, "status": "error", "unknown": unknown,
                    "message": ("Pick at least 2 symbols with stored history"
                                + (f" — no bars for {', '.join(unknown)}." if unknown else "."))}

        series, excluded = {}, []
        for sym in universe:
            days, closes = store.load_closes(sym)
            days, closes = _slice(days, closes, start)
            if len(closes) < M.MIN_TAIL_SAMPLE + 1:
                excluded.append(sym)
            else:
                series[sym] = (days, closes)

        if len(series) < 2:
            return {**empty, "status": "error", "excluded": excluded, "unknown": unknown,
                    "window_limited": window_capacity(key) < M.MIN_TAIL_SAMPLE + 1,
                    "message": (f"Correlation needs {M.MIN_TAIL_SAMPLE + 1}+ sessions per symbol: "
                                f"{shortfall_hint(key, M.MIN_TAIL_SAMPLE + 1)}.")}

        symbols, matrix, avg = M.correlation_matrix(series)
        if not matrix:
            return {**empty, "status": "error", "excluded": excluded, "unknown": unknown,
                    "window_limited": window_capacity(key) < M.MIN_TAIL_SAMPLE + 1,
                    "message": (f"Symbols share fewer than {M.MIN_TAIL_SAMPLE} common sessions: "
                                f"{shortfall_hint(key, M.MIN_TAIL_SAMPLE + 1)}.")}

        # Correlation is computed on RETURNS, which number one fewer than the
        # common days — report what was actually measured.
        observations = len(set.intersection(*[set(series[s][0]) for s in symbols])) - 1
        return {"status": "ok", "symbols": symbols, "matrix": matrix,
                "average_pairwise": avg, "observations": observations,
                "window": key, "window_label": M.WINDOW_LABELS[key],
                "window_start": start, "excluded": excluded,
                "unknown": unknown, "custom": custom}
    except Exception as e:
        logger.error(f"/correlation failed: {e}", exc_info=e)
        return {**empty, "status": "error", "message": str(e)}


def _book(aligned, mkt_days, mkt_r, start):
    """Equal-weighted aggregate over the symbols that share a common calendar.

    Intersecting dates first rather than trusting each symbol's own alignment: a
    name listed mid-window would otherwise contribute a shorter series and
    silently distort the basket.
    """
    if len(aligned) < 2:
        return None
    common = sorted(set.intersection(*[set(d.keys()) for d, _ in aligned.values()]))
    if len(common) < 2 * M.MIN_CAPTURE_DAYS:
        return None
    bench_index = dict(zip(mkt_days, mkt_r))
    if any(d not in bench_index for d in common):
        common = [d for d in common if d in bench_index]
        if len(common) < 2 * M.MIN_CAPTURE_DAYS:
            return None
    series = {s: [d[k] for k in common] for s, (d, _) in aligned.items()}
    return M.book_summary(series, [bench_index[k] for k in common])


def _holdings_book(start, mkt_days, mkt_r):
    """The same aggregate, weighted by what is actually held.

    Weights are |market value| shares of the total. A position is dropped — and
    named — when it has no price or too little stored history to contribute a
    return series, because silently weighting it to zero would overstate how much
    of the book this figure covers.
    """
    try:
        held = _get_positions() or []
    except Exception as e:
        logger.warning(f"Could not read positions for the holdings book: {e}")
        return None
    if not held:
        return None

    series, weights, excluded, estimated = {}, {}, [], []
    for pos in held:
        sym = str(pos.get("symbol", "")).upper()
        mv = abs(float(pos.get("market_value") or 0.0))
        if not sym or mv <= 0:
            excluded.append({"symbol": sym or "?", "reason": "no market value"})
            continue
        days, rets = _returns_for(sym, start)
        if len(rets) < 2 * M.MIN_CAPTURE_DAYS:
            excluded.append({"symbol": sym,
                             "reason": f"only {len(rets)} sessions of stored history"})
            continue
        series[sym] = dict(zip(days, rets))
        weights[sym] = mv
        if pos.get("estimated"):
            estimated.append(sym)

    if not series:
        return {"symbols": 0, "excluded": excluded, "weighting": "market_value",
                "covered_pct": 0.0}

    # Values arrive already converted to one base currency. Anything that could
    # NOT be converted is the only remaining apples-to-oranges risk, so that is
    # what gets reported — not the mere presence of a second currency.
    inc = [p for p in held if str(p.get("symbol", "")).upper() in series]
    base = next((p.get("base_currency") for p in inc if p.get("base_currency")), None)
    unconverted = sorted({str(p.get("symbol")).upper() for p in inc if p.get("no_fx")})
    fx_used = {
        str(p.get("currency")).upper(): {"rate": round(p["fx_rate"], 5), "source": p.get("fx_source")}
        for p in inc if p.get("converted") and p.get("fx_rate")
    }

    bench = dict(zip(mkt_days, mkt_r))
    common = sorted(set(bench) & set.intersection(*[set(v) for v in series.values()]))
    if len(common) < 2 * M.MIN_CAPTURE_DAYS:
        return {"symbols": len(series), "excluded": excluded,
                "weighting": "market_value", "sessions": len(common),
                "error": "not enough overlapping history"}

    total = sum(weights.values())
    w = {k: v / total for k, v in weights.items()}
    b = [bench[d] for d in common]
    import numpy as _np
    port = _np.sum([w[k] * _np.array([series[k][d] for d in common]) for k in series], axis=0)

    out = M.book_summary({"_": port, "__": port}, b) or {}
    # book_summary equal-weights what it is given, so a duplicated single series
    # reproduces the weighted portfolio exactly — but its diversification fields
    # would then be meaningless. Recompute those from the real constituents.
    for k in ("avg_pairwise_corr", "effective_bets", "symbols"):
        out.pop(k, None)
    if len(series) > 1:
        Rm = _np.array([[series[k][d] for d in common] for k in series])
        C = _np.corrcoef(Rm)
        out["avg_pairwise_corr"] = M._finite(float(_np.mean(C[_np.triu_indices(len(series), 1)])))
        out["effective_bets"] = M.effective_bets(C)
    else:
        out["avg_pairwise_corr"] = None
        out["effective_bets"] = 1.0

    out.update({
        "symbols": len(series),
        "weighting": "market_value",
        "weights": {k: round(v * 100, 1) for k, v in sorted(w.items(), key=lambda kv: -kv[1])},
        "excluded": excluded,
        # Weighted off cost basis because no live quote was available.
        "estimated_weights": estimated,
        "base_currency": base,
        "fx_used": fx_used or None,
        "unconverted": unconverted or None,
    })
    return out


# Comparison windows. Longer than the single-symbol chart offers, because the
# point here is relative performance over a meaningful stretch.
COMPARE_RANGES = {
    "1d": {"duration": "1 D", "bar_size": "5 mins", "rth": True,  "label": "1 day"},
    "1w": {"duration": "1 W", "bar_size": "30 mins", "rth": True, "label": "1 week"},
    "1m": {"duration": "1 M", "bar_size": "1 day",  "rth": True,  "label": "1 month"},
    "6m": {"duration": "6 M", "bar_size": "1 day",  "rth": True,  "label": "6 months"},
    "1y": {"duration": "1 Y", "bar_size": "1 day",  "rth": True,  "label": "1 year"},
    "3y": {"duration": "3 Y", "bar_size": "1 day",  "rth": True,  "label": "3 years"},
}
COMPARE_DAILY = ("1m", "6m", "1y", "3y")


async def ensure_history(symbol, min_bars=5):
    """Guarantees stored daily bars for a symbol. Returns (ok, error).

    Always fetches BAR_DURATION, never the caller's window: filling this store
    from a 1-month request is what once left VUAG with 20 bars and no risk
    numbers, because every other consumer reads it as the whole record.
    """
    if store.bar_count(symbol) >= min_bars:
        return True, None
    bars, err = await _fetch_bars(symbol, BAR_DURATION, "1 day", True)
    if bars is None:
        return False, err
    store.save_bars(symbol, bars)
    return True, None
MAX_COMPARE_SYMBOLS = 8


@router.get("/compare")
async def compare_symbols(symbols: str = None, range: str = "1y"):
    """Several symbols rebased to a common start, for visual comparison.

    Rebased to 100, not plotted raw: SNDK trades near $1,800 and FCEL near $18,
    so a shared price axis would render one as a flat line at the bottom. Index
    them and the question becomes "which one moved more", which is the question.
    """
    spec = COMPARE_RANGES.get(range)
    if spec is None:
        return {"status": "error", "series": [],
                "message": f"Unknown range {range!r}. Use one of: {', '.join(COMPARE_RANGES)}"}

    wanted = parse_symbol_list(symbols)[:MAX_COMPARE_SYMBOLS]
    if not wanted:
        return {"status": "ok", "series": [], "range": range,
                "label": spec["label"], "message": "Add a ticker to compare."}

    raw, errors = {}, []
    for sym in wanted:
        try:
            if range in COMPARE_DAILY:
                # Daily ranges come off the store, so they are instant and work
                # with the gateway down. Cold symbols get the full history once.
                ok, err = await ensure_history(sym)
                if not ok:
                    errors.append({"symbol": sym, "error": err})
                    continue
                # window_start only covers the risk windows (1y/2y/3y), so the
                # comparison ranges derive their own cutoff.
                start = _months_back({"1m": 1, "6m": 6, "1y": 12, "3y": 36}[range])
                rows = [b for b in store.load_bars(sym) if b["day"] >= start]
                if len(rows) < 2:
                    errors.append({"symbol": sym, "error": "not enough stored history"})
                    continue
                raw[sym] = [(f"{b['day']}T00:00:00+00:00", float(b["close"])) for b in rows]
            else:
                bars, err = await _fetch_bars(sym, spec["duration"], spec["bar_size"], spec["rth"])
                if bars is None:
                    errors.append({"symbol": sym, "error": err})
                    continue
                raw[sym] = [(_bar_time(b.date), float(b.close)) for b in bars if b.close]
        except Exception as e:
            errors.append({"symbol": sym, "error": str(e)})

    if not raw:
        return {"status": "error", "series": [], "range": range, "label": spec["label"],
                "errors": errors, "message": "None of those symbols could be loaded."}

    # Intersect timestamps so every line starts at the same moment — rebasing on
    # different start dates would make the comparison meaningless.
    common = sorted(set.intersection(*[{t for t, _ in v} for v in raw.values()]))
    if len(common) < 2:
        return {"status": "error", "series": [], "range": range, "label": spec["label"],
                "errors": errors,
                "message": ("These symbols share fewer than 2 common sessions — most likely "
                            "different exchanges or trading hours.")}

    series = []
    for sym, pairs in raw.items():
        idx = dict(pairs)
        closes = [idx[t] for t in common]
        base = closes[0]
        if not base:
            errors.append({"symbol": sym, "error": "zero price at the window start"})
            continue
        series.append({
            "symbol": sym,
            "points": [round(c / base * 100.0, 4) for c in closes],
            "first_close": round(base, 4),
            "last_close": round(closes[-1], 4),
            "change_pct": round((closes[-1] / base - 1.0) * 100.0, 2),
        })
    series.sort(key=lambda x: -x["change_pct"])
    return {"status": "ok", "range": range, "label": spec["label"],
            "bar_size": spec["bar_size"], "times": common,
            "series": series, "errors": errors, "rebased_to": 100}


# ----------------------------------------------------------------------------
# research
# ----------------------------------------------------------------------------
# Display ranges for the Research tab. The rolling window's lead-in is taken
# from history BEFORE these cutoffs, so a 252-session window over a 1y display
# range is a real 252 sessions, not a truncated one.
RESEARCH_RANGES = {"6m": 6, "1y": 12, "2y": 24, "3y": 36}
RESEARCH_WINDOWS = (20, 60, 120, 252)
MAX_BACKTEST_SYMBOLS = 12


def _dated_returns(symbol):
    """(days, returns) over the whole stored history."""
    days, closes = store.load_closes(symbol)
    if len(closes) < 2:
        return [], []
    return days[1:], list(M.to_returns(closes))


def _align_dated(days_a, vals_a, days_b, vals_b):
    """Intersect two dated series, keeping the days. M.align drops them."""
    idx_b = dict(zip(days_b, vals_b))
    rows = [(d, v, idx_b[d]) for d, v in zip(days_a, vals_a) if d in idx_b]
    if not rows:
        return [], [], []
    d, a, b = zip(*rows)
    return list(d), list(a), list(b)


def _research_range(key):
    key = (key or "1y").lower()
    if key not in RESEARCH_RANGES:
        key = "1y"
    return key, _months_back(RESEARCH_RANGES[key])


def _research_window(value, default=60):
    try:
        w = int(value)
    except (TypeError, ValueError):
        return default
    return w if w in RESEARCH_WINDOWS else default


@router.get("/rolling/{symbol}")
async def get_rolling(symbol: str, window: int = 60, range: str = "1y", vs: str = "SPY"):
    """Sigma, beta and correlation as they drifted, rather than one final number.

    Returns-based throughout, so the price magnifier never enters: a pence-quoted
    close divided by the previous pence-quoted close is the same ratio either way.
    """
    symbol, bench = symbol.upper(), (vs or "SPY").upper()
    win = _research_window(window)
    rkey, start = _research_range(range)

    for sym in ({symbol, bench}):
        ok, err = await ensure_history(sym)
        if not ok:
            return {"status": "error", "symbol": symbol, "benchmark": bench,
                    "message": f"No history for {sym}: {err}"}

    sd, sr = _dated_returns(symbol)
    bd, br = _dated_returns(bench)
    days, a, b = _align_dated(sd, sr, bd, br)
    if len(days) < win + 1:
        return {"status": "error", "symbol": symbol, "benchmark": bench, "window": win,
                "message": (f"{symbol} and {bench} share only {len(days)} sessions; a "
                            f"{win}-session rolling window needs at least {win + 1}.")}

    # Computed over the FULL overlap, then sliced for display, so the first
    # plotted point already has a complete window behind it.
    roll = M.rolling_stats(a, b, win)
    if roll is None:
        return {"status": "error", "symbol": symbol, "benchmark": bench, "window": win,
                "message": "Not enough aligned history for a rolling window."}

    points = [
        {"day": days[i], "sigma_annual_pct": sg, "beta": be, "correlation": co}
        for i, sg, be, co in zip(roll["index"], roll["sigma_annual_pct"],
                                 roll["beta"], roll["correlation"])
    ]
    shown = [p for p in points if p["day"] >= start]
    truncated = len(shown) < len([d for d in days if d >= start])
    if not shown:
        shown = points[-1:]

    def _last(key):
        vals = [p[key] for p in shown if p[key] is not None]
        return vals[-1] if vals else None

    return {
        "status": "ok", "symbol": symbol, "benchmark": bench, "window": win,
        "range": rkey, "start": start,
        "days": [p["day"] for p in shown],
        "sigma_annual_pct": [p["sigma_annual_pct"] for p in shown],
        "beta": [p["beta"] for p in shown],
        "correlation": [p["correlation"] for p in shown],
        "latest": {"sigma_annual_pct": _last("sigma_annual_pct"),
                   "beta": _last("beta"), "correlation": _last("correlation")},
        "sessions": len(shown), "overlap_sessions": len(days),
        "partial_window": truncated,
    }


@router.get("/pair")
async def get_pair(a: str = None, b: str = None, range: str = "1y", window: int = 60):
    """One name priced in the other, with a trailing-window z-score."""
    sa = (a or "").strip().upper()
    sb = (b or "").strip().upper()
    if not sa or not sb:
        return {"status": "error", "message": "Give two tickers to compare."}
    if sa == sb:
        return {"status": "error", "message": "Pick two different tickers."}

    win = _research_window(window)
    rkey, start = _research_range(range)
    for sym in (sa, sb):
        ok, err = await ensure_history(sym)
        if not ok:
            return {"status": "error", "message": f"No history for {sym}: {err}"}

    # Divided by the magnifier per leg BEFORE the ratio: a pence-quoted leg
    # against a dollar-quoted one is otherwise wrong by exactly 100x.
    mag_a = _get_magnifier(sa) or 1
    mag_b = _get_magnifier(sb) or 1
    da, ca = store.load_closes(sa)
    db, cb = store.load_closes(sb)
    days, pa, pb = _align_dated(da, [c / mag_a for c in ca], db, [c / mag_b for c in cb])
    if len(days) < win + 1:
        return {"status": "error", "a": sa, "b": sb, "window": win,
                "message": (f"{sa} and {sb} share only {len(days)} sessions; a "
                            f"{win}-session window needs at least {win + 1}. "
                            "Different exchanges often overlap less than you expect.")}

    spread = M.pair_spread(pa, pb, win)
    if spread is None:
        return {"status": "error", "a": sa, "b": sb, "window": win,
                "message": "Could not compute a ratio from these two series."}

    keep = [i for i, d in enumerate(days) if d >= start] or [len(days) - 1]
    lo = keep[0]

    def cut(seq):
        return list(seq[lo:])

    z = cut(spread["z"])
    live = [v for v in z if v is not None]
    ccy_a, ccy_b = _get_currency(sa), _get_currency(sb)
    cross = bool(ccy_a and ccy_b and ccy_a != ccy_b)
    return {
        "status": "ok", "a": sa, "b": sb, "range": rkey, "window": win, "start": start,
        "days": cut(days), "ratio": cut(spread["ratio"]), "mean": cut(spread["mean"]),
        "sd": cut(spread["sd"]), "z": z,
        "latest": {
            "ratio": next((v for v in reversed(cut(spread["ratio"])) if v is not None), None),
            "z": live[-1] if live else None,
            "a_close": round(pa[-1], 4), "b_close": round(pb[-1], 4),
        },
        "currency_a": ccy_a, "currency_b": ccy_b, "cross_currency": cross,
        "price_magnifier_a": mag_a, "price_magnifier_b": mag_b,
        "sessions": len(cut(days)), "overlap_sessions": len(days),
        "note": ("These legs are quoted in different currencies, so the ratio moves "
                 f"with {ccy_a}/{ccy_b} as well as with the two names."
                 if cross else None),
    }


@router.get("/backtest")
async def get_backtest(symbols: str = None, start: str = None, end: str = None,
                       benchmarks: str = "SPY"):
    """Equal-weight buy-and-hold over the stored bars. See the in-sample warning."""
    wanted = parse_symbol_list(symbols)[:MAX_BACKTEST_SYMBOLS]
    if not wanted:
        return {"status": "error", "message": "Add at least one ticker to backtest."}
    bench = parse_symbol_list(benchmarks)[:3]

    start_day = (start or "").strip() or _months_back(12)
    try:
        date.fromisoformat(start_day)
        if end:
            date.fromisoformat(end)
    except ValueError:
        return {"status": "error", "message": "Dates must be YYYY-MM-DD."}

    errors = []
    for sym in list(dict.fromkeys(wanted + bench)):
        ok, err = await ensure_history(sym)
        if not ok:
            errors.append({"symbol": sym, "error": err})
    usable = [s for s in wanted if store.bar_count(s) >= 2]
    if not usable:
        return {"status": "error", "errors": errors,
                "message": "None of those symbols have stored history."}

    result = BT.compare(usable, [b for b in bench if store.bar_count(b) >= 2],
                        start_day, end or None)
    if "error" in result:
        return {"status": "error", "errors": errors, "message": result["error"],
                "skipped": result.get("skipped", [])}

    # Entry/exit are prices, so they carry the magnifier; return_pct and the
    # equity curve are ratios and are already correct.
    for leg in result["legs"]:
        m = _get_magnifier(leg["symbol"]) or 1
        leg["entry"] = round(leg["entry"] / m, 4)
        leg["exit"] = round(leg["exit"] / m, 4)
        leg["return_pct"] = round(leg["return_pct"], 2)
        leg["currency"] = _get_currency(leg["symbol"])
    for key in ("portfolio_return_pct", "max_drawdown_pct", "annualised_vol_pct"):
        if result.get(key) is not None:
            result[key] = round(result[key], 2)
    for stats in result.get("benchmarks", {}).values():
        for key in ("return_pct", "excess_pct", "max_drawdown_pct"):
            if stats.get(key) is not None:
                stats[key] = round(stats[key], 2)
        if stats.get("curve"):
            stats["curve"] = [None if v is None else round(v, 6) for v in stats["curve"]]
    result["curve"] = [round(v, 6) for v in result["curve"]]

    result.update({
        "status": "ok", "requested": wanted, "errors": errors,
        "benchmark_symbols": bench,
        "warning": ("In-sample. Backtesting names you already chose, over a period "
                    "you chose them in, measures what they did — it is not evidence "
                    "the selection rule predicts anything. Equal weight at entry, "
                    "no rebalancing, no costs, no dividends."),
    })
    return result


@router.get("/position-context")
async def get_position_context(window: str = None):
    """Where each name stands versus its own peak, and how it behaves up vs down.

    Replaces the market-neutral theme betas, which measured almost nothing: the
    theme ETFs are up to 79% explained by SPY, so removing the market from both
    sides left near-zero residuals that still rendered as full-size bars.
    """
    try:
        key, start = resolve_window(window)
        mkt_days, mkt_r = _returns_for(MARKET_BENCHMARK, start)
        rows, aligned = [], {}
        theme_cfg = load_theme_config()
        for sym in watch_universe():
            stored = store.load_bars(sym)
            bars, inside = _window_bars(stored, start)
            if len(bars) < M.MIN_SERIES + 1:
                rows.append({"symbol": sym, "error": "not enough stored history"})
                continue

            sym_days, sym_r = _returns_for(sym, start)
            a, b = M.align(sym_days, sym_r, mkt_days, mkt_r)
            # Keep the aligned series for the book aggregate, keyed on the dates
            # so every member can be cut to one common calendar below.
            # Only watchlist names feed the equal-weighted book; a held extra would
            # quietly change what "watchlist, equal weight" means.
            if sym in {s.upper() for s in _get_portfolio()}:
                aligned[sym] = (dict(zip([d for d in sym_days if d in set(mkt_days)], a)), b)
            rows.append({
                "symbol": sym,
                "error": None,
                "theme": theme_of(sym, theme_cfg),
                "sessions": inside,
                "drawdown": M.drawdown_profile(bars),
                "capture": M.capture_ratios(a, b),
                "benchmark": MARKET_BENCHMARK,
            })

        book = _book(aligned, mkt_days, mkt_r, start)
        holdings = _holdings_book(start, mkt_days, mkt_r)
        return {"status": "ok", "window": key, "window_label": M.WINDOW_LABELS[key],
                "window_start": start, "benchmark": MARKET_BENCHMARK,
                "rows": rows, "book": book, "holdings": holdings}
    except Exception as e:
        logger.error(f"/position-context failed: {e}", exc_info=e)
        return {"status": "error", "message": str(e), "rows": []}


@router.post("/refresh")
async def refresh_history(force: bool = False):
    """Re-downloads only what is stale.

    Daily bars for a closed session never change, so a repeat refresh normally has
    nothing to do. Without this the watchlist cost ~49 IBKR requests every time,
    and two refreshes inside ten minutes tripped the pacing limit — which IBKR
    signals by silently returning empty bar lists rather than an error.

    Pass force=true to redownload regardless.
    """
    if _ib is None or not _ib.isConnected():
        return {"status": "error", "message": "IBKR gateway is not connected",
                "fetched": [], "skipped": [], "errors": []}
    if _refresh_lock.locked():
        return {"status": "error", "message": "A refresh is already running.",
                "fetched": [], "skipped": [], "errors": []}

    async with _refresh_lock:
        started = time.monotonic()
        portfolio = watch_universe()

        # Benchmarks first: every gate compares against them, so a watchlist name
        # backfilled without its comparator would score on nothing.
        jobs, skipped = [], []
        for sym in all_benchmarks():
            if needs_bars(sym, force):
                jobs.append((sym, "bars"))
            else:
                skipped.append(sym)
        for sym in portfolio:
            want_bars, want_iv = needs_bars(sym, force), needs_iv(sym, force)
            if want_bars:
                jobs.append((sym, "bars"))
            if want_iv:
                jobs.append((sym, "iv"))
            if not (want_bars or want_iv):
                skipped.append(sym)

        fetched, errors = [], []
        for i, (sym, kind) in enumerate(jobs):
            # Pace between real requests only — skipped symbols must not cost time.
            if i:
                await asyncio.sleep(PACING_DELAY)

            if kind == "bars":
                bars, error = await _fetch(sym, "TRADES", BAR_DURATION)
                if error:
                    errors.append({"symbol": sym, "error": error})
                else:
                    store.save_bars(sym, bars)
                    fetched.append(sym)
            else:
                iv_bars, iv_err = await _fetch(sym, "OPTION_IMPLIED_VOLATILITY", IV_DURATION)
                if iv_err:
                    # Not fatal: a name with no listed options still scores pillars 1-3.
                    logger.info(f"No implied-vol series for {sym}: {iv_err}")
                else:
                    store.save_vol(sym, iv_bars, "IV")

        # last_refresh means "data was actually stored", nothing weaker. Stamping it
        # unconditionally made a refresh that failed on every symbol — no market data
        # entitlement, say — indistinguishable from one that worked, so the UI reported
        # fresh data that was days old. The attempt is recorded separately.
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        stored_anything = bool(fetched)
        if stored_anything:
            store.set_meta("last_refresh", now)
        store.set_meta("last_attempt", now)
        store.set_meta(
            "last_attempt_result",
            "ok" if (stored_anything and not errors)
            else "partial" if stored_anything
            else "no_data" if jobs
            else "nothing_to_do",
        )
        store.set_meta("last_attempt_detail",
                       f"{len(fetched)} stored, {len(errors)} failed, {len(skipped)} already current")

        elapsed = round(time.monotonic() - started, 1)
        level = logger.info if stored_anything or not jobs else logger.error
        level(
            f"Analytics refresh: {len(jobs)} requests, {len(fetched)} stored, "
            f"{len(skipped)} already current, {len(errors)} failed, {elapsed}s"
        )
        return {"status": "ok", "fetched": sorted(set(fetched)), "skipped": sorted(skipped),
                "errors": errors, "requests": len(jobs), "elapsed_seconds": elapsed,
                "stored_anything": stored_anything}


@router.get("/coverage")
async def get_coverage():
    """What history is actually stored — the first thing to check when a gate reads null."""
    return {
        "status": "ok",
        # last_refresh = data was stored. last_attempt = the button was pressed.
        # They differ whenever a refresh failed, which is the case worth seeing.
        "last_refresh": store.get_meta("last_refresh"),
        "data_connection": {
            "configured": _data_state["configured"],
            "target": f"{DATA_HOST}:{DATA_PORT}" if DATA_PORT else "trading connection",
            "connected": _data_state["connected"],
            "last_error": _data_state["last_error"],
            "client_id": DATA_CLIENT_ID if DATA_PORT else None,
        },
        "last_attempt": store.get_meta("last_attempt"),
        "last_attempt_result": store.get_meta("last_attempt_result"),
        "last_attempt_detail": store.get_meta("last_attempt_detail"),
        "database": store.DB_PATH,
        "watchlist": [s.upper() for s in _get_portfolio()],
        "benchmarks": all_benchmarks(),
        "coverage": store.coverage(),
    }


# ---------------------------------------------------------------------------
# price history for the watchlist chart
# ---------------------------------------------------------------------------
# Each range maps to (IBKR duration, bar size, regular-trading-hours only, how many
# of the newest bars to keep). The hour view asks for a whole day of minute bars and
# keeps the last 60: a plain "3600 S" request returns nothing once the session has
# closed, which is exactly when someone is most likely to be looking.
RANGE_SPECS = {
    "1h": {"duration": "1 D",  "bar_size": "1 min",  "rth": False, "keep": 60,  "label": "1 hour"},
    "1d": {"duration": "1 D",  "bar_size": "5 mins", "rth": True,  "keep": None, "label": "1 day"},
    "1m": {"duration": "1 M",  "bar_size": "1 day",  "rth": True,  "keep": 23,  "label": "1 month"},
    "1y": {"duration": "1 Y",  "bar_size": "1 day",  "rth": True,  "keep": 253, "label": "1 year"},
}

# Ranges answerable from the daily bars already on disk — no IBKR request at all.
STORE_RANGES = ("1m", "1y")

# Intraday bars are re-fetched at most this often per symbol+range. Clicking around
# the watchlist otherwise burns the ~60-requests-per-10-minutes budget in seconds.
INTRADAY_TTL = {"1h": 45.0, "1d": 180.0}
_history_cache = {}

# IBKR rejects bursts of historical requests for the same contract. One at a time,
# spaced, so a fast click-through degrades to slow rather than to empty responses.
_history_lock = asyncio.Lock()
_history_gap = 0.35
_last_history_at = 0.0


# Symbols with a top-up already queued, so repeated clicks on a stale chart do not
# queue the same fetch over and over.
_topup_pending = set()


def _schedule_topup(symbol):
    """Background refresh of one symbol's daily bars. Never blocks the caller."""
    if symbol in _topup_pending:
        return

    async def run():
        try:
            async with _refresh_lock:
                # Re-check under the lock: a full refresh may have covered it while
                # this task sat waiting.
                if needs_bars(symbol):
                    result = await backfill_symbol(symbol, include_iv=False)
                    if result.get("error"):
                        logger.info(f"Chart top-up for {symbol} failed: {result['error']}")
        except Exception as e:
            logger.warning(f"Chart top-up for {symbol} failed: {e}")
        finally:
            _topup_pending.discard(symbol)

    _topup_pending.add(symbol)
    asyncio.create_task(run())


async def _fetch_bars(symbol, duration, bar_size, rth):
    """A historical request at an arbitrary bar size. Returns (bars, error)."""
    if _ib is None or not _ib.isConnected():
        return None, "IBKR gateway not connected"

    contract, error = await _resolve_contract(symbol)
    if contract is None:
        return None, error

    handle = await data_connection() or _ib
    global _last_history_at
    async with _history_lock:
        wait = _history_gap - (time.monotonic() - _last_history_at)
        if wait > 0:
            await asyncio.sleep(wait)
        try:
            bars = await asyncio.wait_for(
                handle.reqHistoricalDataAsync(
                    contract, endDateTime="", durationStr=duration,
                    barSizeSetting=bar_size, whatToShow="TRADES",
                    useRTH=rth, formatDate=2,   # 2 = epoch seconds, zone-unambiguous
                ),
                timeout=HIST_TIMEOUT,
            )
        except asyncio.TimeoutError:
            return None, f"timed out after {HIST_TIMEOUT:.0f}s"
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"
        finally:
            _last_history_at = time.monotonic()

    if not bars:
        return None, "IBKR returned no bars (market may not have traded in this window)"
    return bars, None


def _bar_time(value):
    """ib_async hands back a datetime, a date, or epoch seconds depending on bar size."""
    from datetime import date as date_cls
    from datetime import datetime as dt_cls
    from datetime import timezone as tz_cls

    if isinstance(value, (int, float)):
        return dt_cls.fromtimestamp(value, tz_cls.utc).isoformat()
    if isinstance(value, dt_cls):
        return (value if value.tzinfo else value.replace(tzinfo=tz_cls.utc)).isoformat()
    if isinstance(value, date_cls):
        return dt_cls(value.year, value.month, value.day, tzinfo=tz_cls.utc).isoformat()
    return str(value)


def _from_store(symbol, keep):
    """Daily bars already on disk, newest `keep` of them, oldest first.

    Divided by the price magnifier on the way out: bars are stored exactly as
    IBKR sends them (pence for many LSE lines) so that returns-based analytics
    stay untouched, but a chart shows money.
    """
    rows = store.load_bars(symbol, limit=keep)
    m = _get_magnifier(symbol) or 1
    d = (lambda v: v / m) if m != 1 else (lambda v: v)
    return [
        {"t": f"{r['day']}T00:00:00+00:00", "o": d(r["open"]), "h": d(r["high"]),
         "l": d(r["low"]), "c": d(r["close"]), "v": r["volume"] or 0}
        for r in rows
    ]


@router.get("/history/{symbol}")
async def get_price_history(symbol: str, range: str = "1d", refresh: bool = False):
    """OHLCV bars for one symbol over a preset window.

    Month and year come off the stored daily bars, so they render instantly and keep
    working while the gateway is down. Hour and day need intraday resolution, which
    is not stored, so those go to IBKR and are cached briefly.
    """
    symbol = symbol.upper()
    spec = RANGE_SPECS.get(range)
    if spec is None:
        return {"status": "error", "symbol": symbol, "range": range, "bars": [],
                "message": f"Unknown range {range!r}. Use one of: {', '.join(RANGE_SPECS)}"}

    if range in STORE_RANGES:
        bars = _from_store(symbol, spec["keep"])
        if bars:
            last_day = store.last_bar_day(symbol)
            fresh = _is_current(last_day)
            if not fresh:
                # Serve what is on disk now and top it up behind the request: a chart
                # that appears instantly and is current next time beats one that
                # blocks for several seconds on every click.
                _schedule_topup(symbol)
            return {
                "status": "ok", "symbol": symbol, "range": range, "source": "store",
                "barSize": spec["bar_size"], "label": spec["label"], "bars": bars,
                "lastStoredDay": last_day, "stale": not fresh,
                "refreshing": not fresh,
            }
        ok, error = await ensure_history(symbol, min_bars=M.MIN_SERIES)
        if not ok:
            return {"status": "error", "symbol": symbol, "range": range, "bars": [],
                    "message": f"No stored history for {symbol} and the fetch failed: {error}"}
        return {
            "status": "ok", "symbol": symbol, "range": range, "source": "ibkr",
            "barSize": spec["bar_size"], "label": spec["label"],
            "bars": _from_store(symbol, spec["keep"]),
            "lastStoredDay": store.last_bar_day(symbol),
        }

    key = (symbol, range)
    now = time.monotonic()
    if not refresh:
        cached = _history_cache.get(key)
        if cached and cached[0] > now:
            return {**cached[1], "cached": True}

    bars, error = await _fetch_bars(symbol, spec["duration"], spec["bar_size"], spec["rth"])
    if bars is None:
        stale = _history_cache.get(key)
        if stale:
            # Better a slightly old chart, clearly labelled, than an empty panel.
            return {**stale[1], "cached": True, "stale": True, "message": error}
        return {"status": "error", "symbol": symbol, "range": range, "bars": [],
                "message": error}

    m = _get_magnifier(symbol) or 1
    d = (lambda v: v / m) if m != 1 else (lambda v: v)
    rows = [
        {"t": _bar_time(b.date), "o": d(b.open), "h": d(b.high), "l": d(b.low),
         "c": d(b.close), "v": b.volume or 0}
        for b in bars
    ]
    if spec["keep"]:
        rows = rows[-spec["keep"]:]

    payload = {
        "status": "ok", "symbol": symbol, "range": range, "source": "ibkr",
        "barSize": spec["bar_size"], "label": spec["label"], "bars": rows,
        "price_magnifier": m,
    }
    _history_cache[key] = (now + INTRADAY_TTL.get(range, 60.0), payload)
    return payload
