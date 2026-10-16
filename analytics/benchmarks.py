"""Per symbol sector benchmarks.

Both pillar 1 (drift vs sector baseline) and pillar 2 (information ratio) are
meaningless without the RIGHT comparator: measuring Rocket Lab against a
semiconductor ETF produces a confident number about nothing. Mapping is explicit
and manually maintained for exactly that reason, IBKR's own industry categories are
too coarse to separate a satellite imaging company from a defense prime.

Edit SECTOR_BENCHMARK when you add a name. An unmapped symbol falls back to
MARKET_BENCHMARK and is reported as such (benchmark_is_sector = False), so a
silent mis-benchmark is not possible: the API always says which comparator
produced the number, and the UI flags the fallback.
"""

# Beta and the market leg of every regression price off this.
MARKET_BENCHMARK = "SPY"

SECTOR_BENCHMARK = {
    # --- AI cloud infrastructure ---
    # 'Neocloud' GPU-capacity operators. SKYY is the closest established sector
    # mandate; see the note at the bottom of this file, because these two may
    # empirically load harder on SMH than on cloud.
    "NBIS": "SKYY",
    "CRWV": "SKYY",

    # --- Space / national defense ---
    # XAR rather than ITA deliberately: ITA is cap-weighted and dominated by
    # Boeing, RTX and Lockheed, so a small-cap launch company would be measured
    # against megacap primes. XAR is equal-weighted, which gives mid- and
    # small-cap space names real influence on the baseline.
    "RKLB": "XAR",
    "PL":   "XAR",
    # PLTR is defense-ADJACENT software, grouped by end-market as requested. The data
    # disagrees: over 1y it correlates 0.70 with IGV against 0.38 with XAR.
    "PLTR": "XAR",

    # --- Semiconductors ---
    "NVDA": "SMH", "AMD":  "SMH", "AVGO": "SMH", "TSM":  "SMH",
    "MRVL": "SMH", "INTC": "SMH", "SNDK": "SMH",

    # --- Data / infrastructure software ---
    "MDB":  "IGV", "SNOW": "IGV", "ESTC": "IGV",

    # --- Megacap platforms ---
    # XLK for all three. AMZN is consumer discretionary under GICS, but its
    # price behaviour is driven by AWS and ad tech, so a retail benchmark
    # would be the worse distortion of the two.
    "MSFT": "XLK", "GOOGL": "XLK", "AMZN": "XLK",

    # --- Clean energy ---
    # FSLR sits here rather than in TAN on purpose: it is typically TAN's single
    # largest holding, so benchmarking it against TAN measures it largely
    # against itself and drives the information ratio toward zero by construction.
    "BE": "ICLN", "FCEL": "ICLN", "FSLR": "ICLN",
}

# Human-readable, for the UI. Keyed by ETF so it survives symbol remapping.
BENCHMARK_LABELS = {
    "SPY":  "S&P 500",
    "SMH":  "Semiconductors",
    "SKYY": "Cloud infrastructure",
    "XAR":  "Aerospace & defense (equal weight)",
    "IGV":  "Infrastructure software",
    "XLK":  "Technology sector",
    "ICLN": "Clean energy",
}


def benchmark_for(symbol):
    """Returns (etf, is_sector_specific). False means we fell back to the market."""
    etf = SECTOR_BENCHMARK.get(symbol.upper())
    return (etf, True) if etf else (MARKET_BENCHMARK, False)


def all_benchmarks():
    """Every ETF that must be backfilled alongside the watchlist."""
    return sorted({MARKET_BENCHMARK, *SECTOR_BENCHMARK.values()})


def unmapped(symbols):
    """Watchlist names with no sector mapping — these silently benchmark to SPY."""
    return sorted(s.upper() for s in symbols if s.upper() not in SECTOR_BENCHMARK)

# --- A note on verifying these choices ---
# Every mapping here is a judgment call, and the data can check it:
#   /api/analytics/correlation?symbols=NBIS,SKYY,SMH&window=1y
# accepts any symbol with stored bars, so a name can be compared against its
# assigned benchmark and the candidates it lost to. If NBIS tracks SMH harder than
# SKYY, the mapping is wrong and the matrix says so.


# --- Display themes ---
# Deliberately separate from SECTOR_BENCHMARK: that map answers "what should this
# be scored against", which is a statistical question, while this one answers "how
# does the owner think about the book". They disagree on purpose — AI Infrastructure
# holds both neoclouds and chipmakers, which benchmark to SKYY and SMH respectively.
THEMES = (
    ("AI Infrastructure",     ["NBIS", "CRWV", "NVDA", "AMD", "AVGO", "TSM", "MRVL", "INTC", "SNDK"]),
    ("Spacecraft & Satellite", ["RKLB", "PL", "PLTR"]),
    ("Renewable Energy",      ["BE", "FCEL", "FSLR"]),
    ("Software & Data",       ["MDB", "SNOW", "ESTC"]),
    ("Megacap Platforms",     ["AMZN", "MSFT", "GOOGL"]),
    # Index and sector funds, including the benchmark ETFs — if one is ever watched
    # or held it belongs here rather than beside the single names it is measured
    # against. A real ETF group, not a rename of the catch-all: an unclassified
    # STOCK must not end up filed under "ETFs".
    ("ETFs",                  ["VUAG"] + sorted(set(SECTOR_BENCHMARK.values()) | {MARKET_BENCHMARK})),
)

THEME_OF = {sym: name for name, syms in THEMES for sym in syms}
OTHER_THEME = "Other"


def theme_for(symbol):
    """Display theme for a symbol. Anything unmapped lands in 'Other'."""
    return THEME_OF.get(symbol.upper(), OTHER_THEME)


def group_by_theme(symbols):
    """[(theme, [symbols])] in THEMES order, skipping empty groups.

    Returned as a list rather than a dict so the UI renders themes in a stable,
    intentional order instead of whatever the watchlist happens to be in.
    """
    seen = [s.upper() for s in symbols]
    out = []
    for name, members in THEMES:
        hit = [s for s in members if s in seen]
        if hit:
            out.append((name, hit))
    leftover = [s for s in seen if s not in THEME_OF]
    if leftover:
        out.append((OTHER_THEME, leftover))
    return out
