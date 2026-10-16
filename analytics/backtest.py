"""Buy-and-hold backtest over the stored daily bars.

IMPORTANT — what this can and cannot tell you.

Running this on names the scorecard selected, over a period the scorecard's
thresholds were calibrated on, is an IN-SAMPLE test. It measures what those
holdings did; it is not evidence the model predicts anything. A result that
looks good here would look good even if the gates were chosen at random and
then tuned until the winners passed.

To make it evidence you would need the selection to happen strictly before the
holding period, using only data available at that moment (see walk_forward()).
"""

import math

import numpy as np

from . import store


def _series(symbol, start_day, end_day=None):
    """(days, closes) inside the window, or ([], []) when nothing is stored."""
    days, closes = store.load_closes(symbol)
    pairs = [
        (d, c) for d, c in zip(days, closes)
        if d >= start_day and (end_day is None or d <= end_day)
    ]
    if not pairs:
        return [], []
    d, c = zip(*pairs)
    return list(d), list(c)


def _drawdown(values):
    """Largest peak-to-trough fall of an equity curve, as a positive percent."""
    peak, worst = values[0], 0.0
    for v in values:
        peak = max(peak, v)
        worst = min(worst, v / peak - 1.0)
    return abs(worst) * 100.0


def hold(symbols, start_day, end_day=None):
    """Equal-weight buy-and-hold: buy at the first close on/after start_day.

    Weights are set once and left to drift, which is what "hold" means — no
    rebalancing, so a winner grows into a larger share of the book.
    """
    legs, skipped = [], []
    for s in symbols:
        days, closes = _series(s, start_day, end_day)
        if len(closes) < 2:
            skipped.append(s)
            continue
        legs.append({
            "symbol": s,
            "entry_day": days[0], "exit_day": days[-1],
            "entry": closes[0], "exit": closes[-1],
            "return_pct": (closes[-1] / closes[0] - 1.0) * 100.0,
            "days": days, "closes": closes,
        })

    if not legs:
        return {"error": f"no stored bars for any symbol on/after {start_day}",
                "skipped": skipped}

    # Every leg is restated on the COMMON sessions before anything is measured.
    # A leg whose data runs later than the others would otherwise be ranked over a
    # longer period than the portfolio it is part of.
    common = sorted(set.intersection(*[set(l["days"]) for l in legs]))
    if len(common) < 2:
        return {"error": "these symbols share fewer than 2 common sessions in the window",
                "skipped": skipped}
    for l in legs:
        idx = {d: c for d, c in zip(l["days"], l["closes"])}
        l["days"] = common
        l["closes"] = [idx[d] for d in common]
        l["entry_day"], l["exit_day"] = common[0], common[-1]
        l["entry"], l["exit"] = l["closes"][0], l["closes"][-1]
        l["return_pct"] = (l["exit"] / l["entry"] - 1.0) * 100.0

    curve = []
    for i, _ in enumerate(common):
        curve.append(float(np.mean([l["closes"][i] / l["entry"] for l in legs])))

    port_ret = (curve[-1] - 1.0) * 100.0
    daily = np.diff(curve) / np.array(curve[:-1])
    vol_ann = float(np.std(daily, ddof=1) * math.sqrt(252) * 100.0) if daily.size > 1 else None

    legs.sort(key=lambda l: -l["return_pct"])
    return {
        "start": common[0], "end": common[-1], "sessions": len(common), "days": common,
        "symbols": [l["symbol"] for l in legs], "skipped": skipped,
        "portfolio_return_pct": port_ret,
        "best": legs[0]["symbol"], "worst": legs[-1]["symbol"],
        "winners": sum(1 for l in legs if l["return_pct"] > 0),
        "max_drawdown_pct": _drawdown(curve),
        "annualised_vol_pct": vol_ann,
        "legs": [{k: v for k, v in l.items() if k not in ("days", "closes")} for l in legs],
        "curve": curve,
    }


def compare(symbols, benchmarks, start_day, end_day=None):
    """The basket against one or more single-ticker benchmarks."""
    result = hold(symbols, start_day, end_day)
    if "error" in result:
        return result
    result["benchmarks"] = {}
    for b in benchmarks:
        bench = hold([b], start_day, end_day)
        if "error" in bench:
            continue
        # Re-based on the BASKET's sessions, not its own: two curves drawn against
        # one x-axis have to be sampled on the same days, and a benchmark that
        # trades days the basket does not would otherwise run ahead of it.
        idx = dict(zip(bench["days"], bench["curve"]))
        shared = [d for d in result["days"] if d in idx]
        curve = None
        if len(shared) >= 2 and idx[shared[0]]:
            base = idx[shared[0]]
            curve = {d: idx[d] / base for d in shared}
            curve = [curve.get(d) for d in result["days"]]
        result["benchmarks"][b] = {
            "return_pct": bench["portfolio_return_pct"],
            "excess_pct": result["portfolio_return_pct"] - bench["portfolio_return_pct"],
            "max_drawdown_pct": bench["max_drawdown_pct"],
            "curve": curve,
        }
    return result
