"""The quantitative decision model.

Pure functions over price series no IBKR, no database, no FastAPI so every
number here is testable against a series you construct by hand.
Four pillars, each a pass/fail gate against a stated target:

  1. Momentum & drift        annualized drift ABOVE the sector baseline
  2. Risk-adjusted efficiency Sortino > 1.0 and information ratio > 0
  3. Distributional/tail risk skew not meaningfully negative, CVaR inside limit
  4. Implied sentiment        IV/HV spread positive but not event inflated
"""

import calendar
import math
from datetime import date

import numpy as np

TRADING_DAYS = 252

# Lookback windows offered on the Risk tab, as calendar months back from the newest
# stored bar. Calendar rather than a bar count so the label means what it says: a
# holiday-shortened month is still a month.
# Windows shorter than a year were dropped: a 50/200 MA cross needs 201 bars, and
# below ~250 sessions beta, CVaR and skew are too noisy to read. Measured, not assumed.
WINDOW_MONTHS = {"1y": 12, "2y": 24, "3y": 36}
WINDOW_LABELS = {"1y": "1 year", "2y": "2 years", "3y": "3 years"}
DEFAULT_WINDOW = "1y"

# Below this many returns, a gate's verdict disagrees with the full-sample verdict
# often enough to flag. Measured on rolling windows across the watchlist: at 252
# sessions momentum and efficiency flip ~1 time in 3, tail risk ~1 in 4; by ~500 that
# falls to ~20%, ~20% and ~8%. Set just under two calendar years of sessions (~502),
# so a real 2y window clears it instead of missing by a holiday.
WEAK_SAMPLE_SESSIONS = 480

# Below this many returns a tail statistic is not an estimate. At n=20 the 5% tail
# of CVaR95 is a SINGLE observation and skewness carries a standard error near 0.55
# — larger than the -0.25 floor it would be judged against. Those metrics return
# null below it rather than a number that looks like a measurement.
MIN_TAIL_SAMPLE = 30

'''
Information ratio = (portfolio return - benchmark return) / tracking error
Sortino ratio = (portfolio return - target return) / downside deviation 
'''
SORTINO_TARGET = 1.0
INFO_RATIO_TARGET = 0.0
# "Positive or neutral skew". Exactly 0.0 would fail almost every equity on sampling noise alone, so 'neutral' carries a tolerance band.
SKEW_FLOOR = -0.25
# Daily 95% expected shortfall, in percent. Raised from 6.0 after measurement:
# 6% rejected 17 of 21 names, which makes it a volatility filter rather than a
# discriminator. Portfolio median CVaR is ~7.9%. Set this from what you can
# actually absorb on a position — available funds x tolerable daily loss — not
# from what the current holdings happen to score.
CVAR_LIMIT_PCT = 10.0

# Variance risk premium, in annualized vol POINTS (IV% - HV%).
#
# The band is symmetric because the signal is an EXTREME in either direction: far
# positive means the option market has priced an event the share price has not,
# far negative means realized vol is outrunning implied. The old 0..15 band
# assumed positive VRP was the normal state — measurement said otherwise, with 13
# of 21 names negative. Comparing a 60-day trailing IV mean against 60-day
# trailing realized vol turns negative for any name in a strong trend, so the old
# floor rejected momentum rather than risk.
VRP_MIN_POINTS = -15.0
VRP_MAX_POINTS = 15.0


# ----------------------------------------------------------------------------
# series plumbing
# ----------------------------------------------------------------------------
def window_start(last_day, key):
    """ISO cutoff for a trailing window, or None for 'everything stored'."""
    months = WINDOW_MONTHS.get(key)
    if months is None or not last_day:
        return None
    try:
        end = date.fromisoformat(str(last_day))
    except (TypeError, ValueError):
        return None
    y, m = end.year, end.month - months
    while m <= 0:
        m += 12
        y -= 1
    # Clamp the day: 31 Mar minus one month is 28/29 Feb, not an invalid date.
    return date(y, m, min(end.day, calendar.monthrange(y, m)[1])).isoformat()


def to_returns(closes):
    """Simple daily returns. Length is len(closes) - 1."""
    c = np.asarray(closes, dtype=float)
    if c.size < 2:
        return np.array([])
    # Guard against a zero/negative close corrupting the whole series.
    valid = c[:-1] > 0
    r = np.zeros(c.size - 1)
    r[valid] = c[1:][valid] / c[:-1][valid] - 1.0
    return r


def align(days_a, vals_a, days_b, vals_b):
    """Intersects two dated series on their common days.

    Positional zipping is the bug waiting to happen here: two symbols can differ
    in bar count from a later listing date or a halted session, and a one-day
    offset silently turns beta and the information ratio into noise.
    """
    index_b = {d: v for d, v in zip(days_b, vals_b)}
    common = [(v, index_b[d]) for d, v in zip(days_a, vals_a) if d in index_b]
    if not common:
        return np.array([]), np.array([])
    a, b = zip(*common)
    return np.asarray(a, dtype=float), np.asarray(b, dtype=float)


def _finite(x):
    """None for anything a JSON consumer cannot render — nan/inf never reach the UI."""
    if x is None:
        return None
    f = float(x)
    return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)


# ----------------------------------------------------------------------------
# pillar 1 — momentum & drift
# ----------------------------------------------------------------------------
def annualized_drift(returns):
    """mean daily return scaled to a year, as a percentage."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return None
    return float(np.mean(r) * TRADING_DAYS * 100.0)


def moving_average_state(closes, fast=50, slow=200):
    """Golden/death cross state plus how long it has held.

    Reported as supporting context rather than a gate of its own: a crossover is
    a lagging restatement of the drift already measured above, so counting it
    twice would double-weight one piece of evidence.
    """
    c = np.asarray(closes, dtype=float)
    if c.size < slow + 1:
        return {"state": None, "days_since_cross": None,
                "reason": f"needs {slow + 1} bars, have {c.size}"}

    def sma(arr, n):
        # Cumulative-sum rolling mean; returns one value per window position.
        cs = np.cumsum(np.insert(arr, 0, 0.0))
        return (cs[n:] - cs[:-n]) / n

    f = sma(c, fast)[-(c.size - slow + 1):]
    s = sma(c, slow)
    above = f > s
    state = "golden" if above[-1] else "death"

    days = 0
    for flag in above[::-1]:
        if flag != above[-1]:
            break
        days += 1
    return {"state": state, "days_since_cross": int(days), "reason": None}


# ----------------------------------------------------------------------------
# pillar 2 — risk-adjusted efficiency
# ----------------------------------------------------------------------------
def sortino_ratio(returns, mar_annual_pct=0.0):
    """Return per unit of DOWNSIDE deviation.

    Uses the full-length denominator (Sortino/Satchell), not the count of losing
    days: dividing by only the negative observations inflates the ratio for
    anything that rarely falls, which is precisely the name you must not flatter.
    """
    r = np.asarray(returns, dtype=float)
    if r.size < 30:
        return None
    mar_daily = mar_annual_pct / 100.0 / TRADING_DAYS
    excess = r - mar_daily
    downside = np.minimum(excess, 0.0)
    dd = math.sqrt(float(np.mean(downside ** 2)))
    if dd == 0:
        return None
    return float((np.mean(excess) * TRADING_DAYS) / (dd * math.sqrt(TRADING_DAYS)))


def information_ratio(returns, bench_returns):
    """Annualized active return divided by tracking error."""
    a = np.asarray(returns, dtype=float)
    b = np.asarray(bench_returns, dtype=float)
    if a.size < 30 or a.size != b.size:
        return None
    active = a - b
    te = float(np.std(active, ddof=1))
    if te == 0:
        return None
    return float((np.mean(active) * TRADING_DAYS) / (te * math.sqrt(TRADING_DAYS)))


# ----------------------------------------------------------------------------
# pillar 3 — distribution & tail risk
# ----------------------------------------------------------------------------
def skewness(returns):
    """Bias-corrected sample skewness (Fisher-Pearson G1), matching scipy's bias=False."""
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < MIN_TAIL_SAMPLE:
        return None
    m = r - r.mean()
    m2 = float(np.mean(m ** 2))
    m3 = float(np.mean(m ** 3))
    if m2 <= 0:
        return None
    g1 = m3 / (m2 ** 1.5)
    return float(math.sqrt(n * (n - 1)) / (n - 2) * g1)


def expected_shortfall(returns, confidence=0.95):
    """Historical CVaR: the mean loss on the worst (1-confidence) share of days.

    Empirical rather than Gaussian on purpose — a normal-distribution CVaR would
    understate exactly the fat left tail this pillar exists to detect. Returned
    as a POSITIVE percentage, so 6.2 means "the bad days average -6.2%".
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 30:
        return None, None
    tail_n = max(1, int(math.floor(n * (1.0 - confidence))))
    worst = np.sort(r)[:tail_n]
    var = float(-np.sort(r)[tail_n - 1] * 100.0)
    cvar = float(-np.mean(worst) * 100.0)
    return cvar, var


# ----------------------------------------------------------------------------
# pillar 4 — implied sentiment (variance risk premium)
# ----------------------------------------------------------------------------
def variance_risk_premium(iv_series, returns, window=60):
    """IV minus realized vol, in annualized vol points.

    IBKR serves both legs off the UNDERLYING via reqHistoricalData, so this needs
    no option chain and no OPRA entitlement. Put-call skew — the other half of
    the sentiment pillar — does, and is deliberately not attempted here.
    """
    iv = np.asarray([v for v in iv_series if v is not None], dtype=float)
    r = np.asarray(returns, dtype=float)
    if iv.size < 5 or r.size < 30:
        return {"iv_pct": None, "hv_pct": None, "vrp_points": None}

    iv_recent = float(np.mean(iv[-window:]) * 100.0)          # IBKR gives decimals
    hv_recent = float(np.std(r[-window:], ddof=1) * math.sqrt(TRADING_DAYS) * 100.0)
    return {"iv_pct": iv_recent, "hv_pct": hv_recent, "vrp_points": iv_recent - hv_recent}


# ----------------------------------------------------------------------------
# supporting statistics (these feed the existing Risk tab)
# ----------------------------------------------------------------------------
def volatility(returns):
    r = np.asarray(returns, dtype=float)
    if r.size < 5:
        return None, None, None
    sigma_d = float(np.std(r, ddof=1) * 100.0)
    # Standard error of a standard deviation ~ sigma/sqrt(2n). Shown so two
    # decimals of sigma are not mistaken for two decimals of confidence.
    stderr = sigma_d / math.sqrt(2 * r.size)
    return sigma_d, sigma_d * math.sqrt(TRADING_DAYS), stderr


def beta(returns, bench_returns):
    a = np.asarray(returns, dtype=float)
    b = np.asarray(bench_returns, dtype=float)
    if a.size < 30 or a.size != b.size:
        return None
    var_b = float(np.var(b, ddof=1))
    if var_b == 0:
        return None
    return float(np.cov(a, b, ddof=1)[0][1] / var_b)


def atr(bars, period=14):
    """Wilder's average true range, in price units."""
    if len(bars) < period + 1:
        return None
    high = np.array([b["high"] for b in bars], dtype=float)
    low = np.array([b["low"] for b in bars], dtype=float)
    close = np.array([b["close"] for b in bars], dtype=float)
    prev = close[:-1]
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - prev), np.abs(low[1:] - prev)))
    # Wilder smoothing: seed with a simple mean, then recursively blend.
    a = float(np.mean(tr[:period]))
    for x in tr[period:]:
        a = (a * (period - 1) + float(x)) / period
    return a


def range_position(bars, window=252):
    """Where the last close sits in its 52-week range: 0.0 = at the low, 1.0 = at the high."""
    recent = bars[-window:]
    if len(recent) < 20:
        return None
    hi = max(b["high"] for b in recent)
    lo = min(b["low"] for b in recent)
    if hi <= lo:
        return None
    return float((recent[-1]["close"] - lo) / (hi - lo))


def z_bucket(last_return_pct, sigma_daily_pct):
    """Buckets the latest move against the name's own daily range.

    Words, not a printed z: sigma carries ~16% sampling error at these windows and
    returns are fat-tailed enough that a decimal z would imply confidence the
    estimate cannot support.
    """
    if not sigma_daily_pct or sigma_daily_pct <= 0 or last_return_pct is None:
        return "normal"
    z = abs(last_return_pct) / sigma_daily_pct
    if z < 1.0:
        return "normal"
    if z < 2.0:
        return "notable"
    if z < 3.0:
        return "unusual"
    return "extreme"


def correlation_matrix(series_by_symbol):
    """Pairwise correlation over the days every symbol shares.

    Intersecting on common days rather than truncating to a common LENGTH: the
    latter silently compares different calendar periods when listing dates differ.
    """
    symbols = sorted(series_by_symbol)
    if len(symbols) < 2:
        return symbols, [], None

    day_sets = [set(series_by_symbol[s][0]) for s in symbols]
    common = sorted(set.intersection(*day_sets)) if day_sets else []
    if len(common) < 30:
        return symbols, [], None

    cols = []
    for s in symbols:
        days, closes = series_by_symbol[s]
        index = dict(zip(days, closes))
        cols.append(to_returns([index[d] for d in common]))

    m = np.corrcoef(np.vstack(cols))
    matrix = [[round(float(v), 4) if np.isfinite(v) else 0.0 for v in row] for row in m]

    n = len(symbols)
    pairs = [matrix[i][j] for i in range(n) for j in range(i + 1, n)]
    avg = round(float(np.mean(pairs)), 4) if pairs else None
    return symbols, matrix, avg


# ----------------------------------------------------------------------------
# the scorecard
# ----------------------------------------------------------------------------
def _gate(passed, detail):
    return {"pass": bool(passed) if passed is not None else None, "detail": detail}


# Below this even a standard deviation is not worth printing.
MIN_SERIES = 10


def build_scorecard(symbol, bars, bench_returns, sector_etf, sector_specific,
                    iv_series, market_returns=None, cvar_limit_pct=CVAR_LIMIT_PCT,
                    window=None, aligned=None):
    """Runs all four pillars for one symbol over one trailing window.

    `bars`, `bench_returns`, `market_returns` and `iv_series` must ALREADY be sliced
    to the window — the caller owns that, because only it can align the benchmark on
    the same dates. Metrics whose own sample floor the window cannot meet return
    null rather than a number, so a short window understates confidence instead of
    inventing it.
    """
    days = [b["day"] for b in bars]
    closes = [b["close"] for b in bars]
    r = to_returns(closes)

    window = window or {}
    if r.size < MIN_SERIES:
        return {"symbol": symbol,
                "error": f"only {r.size + 1} bars in this window — need {MIN_SERIES + 1}+",
                **window}

    sigma_d, sigma_a, sigma_se = volatility(r)

    # Single-series stats use the whole window; anything comparing two series uses
    # their COMMON dates instead. Requiring equal raw lengths silently nulled beta
    # and the information ratio whenever a benchmark was one session staler than
    # the symbol — which, with per-symbol refreshes, is most of the time.
    aligned = aligned or {}
    sym_b, bench_b = aligned.get("bench", (r, bench_returns))
    sym_m, mkt_m = aligned.get("market", (None, None))

    # Market beta is separate from the sector comparator: the Risk tab shows both,
    # because a name can look uncorrelated with its peers and still fall with the
    # index in a selloff.
    beta_spy = beta(sym_m, mkt_m) if sym_m is not None and len(sym_m) else None
    drift = annualized_drift(r)
    # Both legs off the overlap, so "beat the benchmark" is measured on days both
    # actually traded.
    sym_drift = annualized_drift(sym_b) if len(sym_b) else None
    bench_drift = annualized_drift(bench_b) if len(bench_b) else None
    excess_drift = ((sym_drift - bench_drift)
                    if (sym_drift is not None and bench_drift is not None) else None)

    ma = moving_average_state(closes)
    sortino = sortino_ratio(r)
    ir = information_ratio(sym_b, bench_b)
    skew = skewness(r)
    cvar, var95 = expected_shortfall(r)
    vrp = variance_risk_premium(iv_series, r)

    last_ret = float(r[-1] * 100.0)

    # --- gates ---
    p1 = _gate(
        None if excess_drift is None else excess_drift > 0,
        f"drift {sym_drift:.1f}%/yr vs {sector_etf} {bench_drift:.1f}%/yr"
        if excess_drift is not None else "no benchmark history",
    )
    p2_ok = None if (sortino is None or ir is None) else (
        sortino > SORTINO_TARGET and ir > INFO_RATIO_TARGET)
    p2 = _gate(p2_ok,
               f"Sortino {sortino:.2f} (>{SORTINO_TARGET}), IR {ir:.2f} vs {sector_etf} (>0)"
               if p2_ok is not None else "insufficient history")
    p3_ok = None if (skew is None or cvar is None) else (
        skew >= SKEW_FLOOR and cvar <= cvar_limit_pct)
    p3 = _gate(p3_ok,
               f"skew {skew:+.2f} (>={SKEW_FLOOR}), CVaR95 {cvar:.1f}% (<={cvar_limit_pct}%)"
               if p3_ok is not None else "insufficient history")
    vp = vrp["vrp_points"]
    p4_ok = None if vp is None else (VRP_MIN_POINTS <= vp <= VRP_MAX_POINTS)
    p4 = _gate(p4_ok,
               f"IV {vrp['iv_pct']:.1f}% vs HV {vrp['hv_pct']:.1f}% = {vp:+.1f} pts "
               f"(band {VRP_MIN_POINTS:+.0f} to {VRP_MAX_POINTS:+.0f})"
               if vp is not None else "no implied-vol history stored")

    # Flag the gates whose verdict this sample cannot hold steady. Sentiment is left
    # unflagged: its VRP leg is a level comparison, not a ratio of means, and its
    # stability was not measured.
    weak_n = r.size < WEAK_SAMPLE_SESSIONS
    WEAK_WHY = {
        "momentum": "annualised drift over a short sample; this verdict flips about "
                    "1 time in 3 against a longer window",
        "efficiency": "Sortino and the information ratio are ratios of annualised means "
                      "— the noisiest statistics here; this verdict flips about 1 time in 3",
        "tail_risk": "skew needs years to settle; this verdict flips about 1 time in 4",
    }
    for key, gate in (("momentum", p1), ("efficiency", p2), ("tail_risk", p3)):
        if weak_n and gate["pass"] is not None:
            gate["weak"] = True
            gate["weak_reason"] = (
                f"{r.size} sessions — {WEAK_WHY[key]}. "
                f"About 2 years of sessions steadies it."
            )

    gates = [p1, p2, p3, p4]
    decided = [g["pass"] for g in gates if g["pass"] is not None]

    return {
        "symbol": symbol,
        "error": None,
        "bars": len(bars),
        "benchmark": sector_etf,
        "benchmark_is_sector": sector_specific,
        "pillars": {
            "momentum":   {**p1, "drift_annual_pct": _finite(drift),
                           "benchmark_drift_annual_pct": _finite(bench_drift),
                           "excess_drift_pct": _finite(excess_drift),
                           "ma_state": ma["state"], "ma_days": ma["days_since_cross"]},
            "efficiency": {**p2, "sortino": _finite(sortino), "information_ratio": _finite(ir)},
            "tail_risk":  {**p3, "skew": _finite(skew), "cvar95_pct": _finite(cvar),
                           "var95_pct": _finite(var95), "cvar_limit_pct": cvar_limit_pct},
            "sentiment":  {**p4, "iv_pct": _finite(vrp["iv_pct"]),
                           "hv_pct": _finite(vrp["hv_pct"]),
                           "vrp_points": _finite(vp)},
        },
        "gates_passed": sum(1 for g in decided if g),
        "gates_evaluated": len(decided),
        # Supporting stats the Risk tab already renders.
        "sigma_daily_pct": _finite(sigma_d),
        "sigma_annual_pct": _finite(sigma_a),
        "sigma_stderr_pct": _finite(sigma_se),
        "beta_spy": _finite(beta_spy),
        "vol_window": int(r.size),
        "atr14": _finite(atr(bars)),
        "range_position": _finite(range_position(bars)),
        # range_position takes min(252, window). At a 1-month window it is a
        # 1-month range, so the UI must not label it "52-week".
        "range_position_days": min(len(bars), 252),
        **window,
        "last_return_pct": _finite(last_ret),
        "last_z_bucket": z_bucket(last_ret, sigma_d),
        "last_day": days[-1] if days else None,
    }


# ----------------------------------------------------------------------------
# position context — where a name stands, and how it behaves in each direction
# ----------------------------------------------------------------------------
def drawdown_profile(bars):
    """Distance from the running peak, now and at worst, over the bars given.

    Computed on closes rather than intraday lows: a stop is not triggered by a
    wick on a chart you are not watching, and close-to-close is what the rest of
    the risk tab measures.
    """
    if len(bars) < 10:
        return None
    closes = np.array([b["close"] for b in bars], dtype=float)
    days = [b["day"] for b in bars]
    if not np.all(np.isfinite(closes)) or closes.min() <= 0:
        return None

    peak = np.maximum.accumulate(closes)
    dd = closes / peak - 1.0
    under = dd < -1e-9

    # Longest unbroken stretch below a prior peak — the "how long was I wrong"
    # number that a max-drawdown figure alone never tells you.
    longest = run = 0
    for flag in under:
        run = run + 1 if flag else 0
        longest = max(longest, run)

    trough = int(np.argmin(dd))
    last_peak = int(np.max(np.where(closes >= peak - 1e-12)[0]))
    return {
        "current_pct": _finite(dd[-1] * 100.0),
        "max_pct": _finite(dd.min() * 100.0),
        "trough_day": days[trough],
        "peak_day": days[last_peak],
        "days_under": int(under.sum()),
        "longest_under": int(longest),
        "sessions": len(closes),
    }


# Below this many same-signed days, a capture ratio is averaging too few
# observations in one direction to mean anything.
MIN_CAPTURE_DAYS = 20


def capture_ratios(returns, bench_returns):
    """Share of the benchmark's up moves and down moves the symbol captures.

    Beta averages both directions into one number, so it cannot distinguish a
    name that rises with the market and holds up when it falls from one that does
    the reverse. These split it. Over 100% means amplifying the benchmark.
    """
    a = np.asarray(returns, dtype=float)
    b = np.asarray(bench_returns, dtype=float)
    if a.size != b.size or a.size < 2 * MIN_CAPTURE_DAYS:
        return {"up_pct": None, "down_pct": None, "asymmetry_pct": None,
                "up_days": 0, "down_days": 0}

    up, down = b > 0, b < 0
    out = {"up_days": int(up.sum()), "down_days": int(down.sum())}
    up_c = down_c = None
    if up.sum() >= MIN_CAPTURE_DAYS:
        mb = float(np.mean(b[up]))
        if mb > 0:
            up_c = float(np.mean(a[up]) / mb) * 100.0
    if down.sum() >= MIN_CAPTURE_DAYS:
        mb = float(np.mean(b[down]))
        if mb < 0:
            down_c = float(np.mean(a[down]) / mb) * 100.0

    out["up_pct"] = _finite(up_c)
    out["down_pct"] = _finite(down_c)
    # Positive = rises more than it falls, the asymmetry worth owning.
    out["asymmetry_pct"] = _finite(up_c - down_c) if (up_c is not None and down_c is not None) else None
    return out


def effective_bets(corr):
    """How many independent positions a correlated book really amounts to.

    Participation ratio of the correlation matrix eigenvalues: 1/sum(p^2) where p
    is each eigenvalue's share. N uncorrelated names give N; N identical names
    give 1. Answers "am I diversified" in a way a count of tickers cannot.
    """
    c = np.asarray(corr, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] < 2:
        return None
    ev = np.linalg.eigvalsh(c)
    ev = ev[ev > 0]
    if ev.size == 0:
        return None
    p = ev / ev.sum()
    return _finite(1.0 / float(np.sum(p ** 2)))


def book_summary(returns_by_symbol, bench_returns):
    """Aggregate risk for an EQUAL-WEIGHTED basket of the given symbols.

    Equal weight, not position weight: this describes the kind of names on the
    watchlist, most of which are not held. Beta is linear so it equals the mean of
    the individual betas, but capture is a ratio — it has to be computed from the
    basket's own return series, not averaged.
    """
    names = [s for s, r in returns_by_symbol.items() if len(r) == len(bench_returns)]
    if len(names) < 2 or len(bench_returns) < 2 * MIN_CAPTURE_DAYS:
        return None

    R = np.array([np.asarray(returns_by_symbol[s], dtype=float) for s in names])
    b = np.asarray(bench_returns, dtype=float)
    basket = R.mean(axis=0)

    var_b = float(np.var(b, ddof=1))
    beta = float(np.cov(basket, b, ddof=1)[0][1] / var_b) if var_b > 0 else None
    cap = capture_ratios(basket, b)
    sigma_d, sigma_a, _ = volatility(basket)

    return {
        "symbols": len(names),
        "sessions": int(basket.size),
        "beta": _finite(beta),
        "sigma_daily_pct": _finite(sigma_d),
        "sigma_annual_pct": _finite(sigma_a),
        "up_capture_pct": cap["up_pct"],
        "down_capture_pct": cap["down_pct"],
        "asymmetry_pct": cap["asymmetry_pct"],
        "avg_pairwise_corr": _finite(float(np.mean(
            np.corrcoef(R)[np.triu_indices(len(names), 1)]))) if len(names) > 1 else None,
        "effective_bets": effective_bets(np.corrcoef(R)),
        "weighting": "equal",
    }


# ----------------------------------------------------------------------------
# research — how a statistic moved, and how two names moved against each other
# ----------------------------------------------------------------------------
def rolling_stats(sym_returns, bench_returns, window=60):
    """Trailing sigma, beta and correlation at each point in the series.

    Definitions are kept identical to volatility() and beta() above rather than
    re-derived, so the last rolling point and the Risk tab's point-in-time figure
    for the same window cannot disagree.
    """
    a = np.asarray(sym_returns, dtype=float)
    b = np.asarray(bench_returns, dtype=float)
    w = int(window)
    if a.size != b.size or a.size < w + 1 or w < 5:
        return None

    out = {"index": [], "sigma_annual_pct": [], "beta": [], "correlation": []}
    for i in range(w, a.size + 1):
        sa, sb = a[i - w:i], b[i - w:i]
        out["index"].append(i - 1)
        out["sigma_annual_pct"].append(
            _finite(float(np.std(sa, ddof=1) * 100.0 * math.sqrt(TRADING_DAYS))))
        var_b = float(np.var(sb, ddof=1))
        out["beta"].append(
            _finite(float(np.cov(sa, sb, ddof=1)[0][1] / var_b)) if var_b > 0 else None)
        sda, sdb = float(np.std(sa, ddof=1)), float(np.std(sb, ddof=1))
        out["correlation"].append(
            _finite(float(np.corrcoef(sa, sb)[0][1])) if sda > 0 and sdb > 0 else None)
    return out


def pair_spread(a_closes, b_closes, window=60):
    """Price ratio of two names, with a rolling mean, band and z-score.

    The ratio, not the difference: a difference between a $230 name and a $17 one
    is dominated by the larger leg and says nothing about relative value.
    """
    a = np.asarray(a_closes, dtype=float)
    b = np.asarray(b_closes, dtype=float)
    w = int(window)
    if a.size != b.size or a.size < w + 1 or w < 5:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(b > 0, a / b, np.nan)
    if not np.any(np.isfinite(ratio)):
        return None

    mean, sd, z = [], [], []
    for i in range(ratio.size):
        if i + 1 < w:
            mean.append(None); sd.append(None); z.append(None)
            continue
        seg = ratio[i + 1 - w:i + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size < max(5, w // 2):
            mean.append(None); sd.append(None); z.append(None)
            continue
        m, s = float(np.mean(seg)), float(np.std(seg, ddof=1))
        mean.append(_finite(m))
        sd.append(_finite(s))
        # z against the TRAILING window, so it answers "stretched versus its own
        # recent behaviour", not versus a full-sample mean the pair never revisits.
        z.append(_finite((float(ratio[i]) - m) / s) if s > 0 and np.isfinite(ratio[i]) else None)

    return {
        "ratio": [_finite(float(x)) if np.isfinite(x) else None for x in ratio],
        "mean": mean, "sd": sd, "z": z, "window": w,
    }
