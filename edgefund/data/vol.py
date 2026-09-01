"""Realised-volatility estimators and trend scoring.

The variance risk premium is only meaningful if the realised-vol side of the
ratio is measured well, so this uses Yang-Zhang for daily data (it accounts for
overnight gaps and is far more efficient than close-to-close) and true
intraday realised vol for short-dated options.

Horizon matching matters more than the estimator choice: pricing a 1-DTE option
against 20-day daily vol compares two different things. `realized_vol` picks the
estimator that matches the option's actual life.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

TRADING_DAYS = 252


def _logret(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return math.log(a / b)


def close_to_close_vol(bars: Sequence[dict[str, Any]], window: int = 20) -> float:
    """Annualised close-to-close volatility. The simple baseline."""
    closes = [b["c"] for b in bars if b.get("c")]
    if len(closes) < 3:
        return 0.0
    rets = [_logret(closes[i], closes[i - 1]) for i in range(1, len(closes))][-window:]
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var * TRADING_DAYS)


def yang_zhang_vol(bars: Sequence[dict[str, Any]], window: int = 20) -> float:
    """Annualised Yang-Zhang volatility.

    Decomposes into overnight, open-to-close and Rogers-Satchell components,
    which makes it robust to both opening gaps and intraday drift -- exactly the
    two things that distort a close-to-close number on an index ETF.
    """
    rows = [b for b in bars if all(b.get(k) for k in ("o", "h", "l", "c"))][-(window + 1):]
    n = len(rows) - 1
    if n < 3:
        return close_to_close_vol(bars, window)

    overnight, open_close, rs = [], [], []
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        o, h, l, c = cur["o"], cur["h"], cur["l"], cur["c"]
        overnight.append(_logret(o, prev["c"]))
        open_close.append(_logret(c, o))
        rs.append(_logret(h, c) * _logret(h, o) + _logret(l, c) * _logret(l, o))

    def _var(xs: list[float]) -> float:
        m = sum(xs) / len(xs)
        return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)

    var_o = _var(overnight)
    var_c = _var(open_close)
    var_rs = sum(rs) / len(rs)

    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = var_o + k * var_c + (1 - k) * var_rs
    if var <= 0:
        return close_to_close_vol(bars, window)
    return math.sqrt(var * TRADING_DAYS)


def short_horizon_rv(daily_bars: Sequence[dict[str, Any]],
                     intraday_bars: Sequence[dict[str, Any]],
                     days: int = 5) -> float:
    """Annualised full-day realised vol built from intraday bars.

    Total daily variance is the sum of two pieces:

        var_day = var_overnight + var_intraday

    Measuring only the intraday piece is a trap -- it silently drops every
    opening gap and understates SPY's realised vol by roughly a third, which
    then inflates the VRP ratio and makes premium look richer than it is. A
    1-DTE option sold today lives through tonight's gap, so the gap belongs in
    the denominator.
    """
    by_day: dict[str, list[float]] = defaultdict(list)
    for b in intraday_bars:
        ts = b.get("t", "")
        if ts and b.get("c"):
            by_day[ts[:10]].append(b["c"])

    # previous session close, for the overnight leg
    prev_close: dict[str, float] = {}
    ordered = [b for b in daily_bars if b.get("c") and b.get("t")]
    for i in range(1, len(ordered)):
        prev_close[ordered[i]["t"][:10]] = ordered[i - 1]["c"]

    daily_var: list[float] = []
    for day in sorted(by_day)[-days:]:
        closes = by_day[day]
        if len(closes) < 5:
            continue
        rets = [_logret(closes[i], closes[i - 1]) for i in range(1, len(closes))]
        var_intraday = sum(r * r for r in rets)

        var_overnight = 0.0
        if day in prev_close and prev_close[day] > 0:
            gap = _logret(closes[0], prev_close[day])
            var_overnight = gap * gap

        daily_var.append(var_overnight + var_intraday)

    if not daily_var:
        return 0.0
    return math.sqrt(sum(daily_var) / len(daily_var) * TRADING_DAYS)


def realized_vol(daily_bars: Sequence[dict[str, Any]],
                 intraday_bars: Sequence[dict[str, Any]] | None,
                 dte: int) -> tuple[float, str]:
    """Horizon-matched realised vol. Returns (annualised_vol, estimator_name).

    For short-dated contracts we take the *higher* of the intraday-derived
    figure and a short-window Yang-Zhang. Being wrong in the direction of
    understating realised vol is the expensive error here: it makes premium
    look rich when it is not and talks us into selling cheap options.
    """
    if dte <= 2 and intraday_bars:
        rv_intra = short_horizon_rv(daily_bars, intraday_bars)
        rv_yz = yang_zhang_vol(daily_bars, window=10)
        if rv_intra > 0 and rv_yz > 0:
            return (max(rv_intra, rv_yz),
                    "max(intraday_5m+overnight, yang_zhang_10d)")
        if rv_intra > 0:
            return rv_intra, "intraday_5m+overnight"

    window = 10 if dte <= 2 else 20
    rv = yang_zhang_vol(daily_bars, window=window)
    if rv > 0:
        return rv, f"yang_zhang_{window}d"

    return close_to_close_vol(daily_bars, 20), "close_to_close_20d"


def trend_score(bars: Sequence[dict[str, Any]]) -> float:
    """Normalised trend in [-1, 1] from EMA alignment plus position in range.

    Used to choose the *structure* (put-credit in an uptrend, call-credit in a
    downtrend, condor when flat) rather than to decide whether an edge exists.
    """
    closes = [b["c"] for b in bars if b.get("c")]
    if len(closes) < 21:
        return 0.0

    def ema(values: list[float], span: int) -> float:
        a = 2 / (span + 1)
        out = values[0]
        for v in values[1:]:
            out = a * v + (1 - a) * out
        return out

    fast, slow = ema(closes[-30:], 8), ema(closes[-60:] or closes, 21)
    spot = closes[-1]
    if slow <= 0:
        return 0.0

    # EMA separation, scaled so ~2% divergence saturates the signal
    sep = max(-1.0, min(1.0, ((fast - slow) / slow) / 0.02))

    lookback = closes[-20:]
    lo, hi = min(lookback), max(lookback)
    pos = 0.0 if hi <= lo else ((spot - lo) / (hi - lo)) * 2 - 1

    return round(max(-1.0, min(1.0, 0.6 * sep + 0.4 * pos)), 4)
