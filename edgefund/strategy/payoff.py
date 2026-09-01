"""Expected value of an option structure under our own volatility forecast.

This is where the edge stops being a number on a dashboard and starts driving
decisions.

The market prices options off implied volatility. Our whole thesis is that
implied is richer than what will actually be realised. So when we evaluate a
structure we integrate its expiry payoff against a distribution built from
*realised* vol, not implied. If the thesis is right, structures the market
considers fairly priced show positive expected value to us, and the size of
that gap is exactly the variance risk premium we set out to harvest.

This replaces the credit-to-width heuristic that a first pass used. That rule
of thumb could not distinguish a good spread from a bad one -- measured live it
rejected every candidate at 18-delta short strikes on 1-2 DTE, including ones
with clearly positive expected value, purely because short-dated OTM verticals
structurally collect a small fraction of their width.

Payoff is computed generically from the legs, so verticals, iron condors and
debit spreads all go through the same code path:

    payoff(S) = net_credit + sum(long intrinsics) - sum(short intrinsics)
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

TRADING_DAYS = 252
GRID_POINTS = 601
GRID_SIGMAS = 5.0


def _intrinsic(spot_grid: np.ndarray, strike: float, opt_type: str) -> np.ndarray:
    if opt_type == "call":
        return np.maximum(spot_grid - strike, 0.0)
    return np.maximum(strike - spot_grid, 0.0)


def payoff_curve(legs: Sequence, net_credit: float, spot_grid: np.ndarray) -> np.ndarray:
    """Expiry P&L per share across a grid of terminal prices.

    `net_credit` is positive when premium was received and negative when paid,
    which makes the same expression correct for credit and debit structures.
    """
    total = np.full_like(spot_grid, float(net_credit))
    for leg in legs:
        intr = _intrinsic(spot_grid, float(leg.strike), leg.opt_type)
        total = total + intr if leg.side == "buy" else total - intr
    return total


def terminal_distribution(spot: float, vol: float, dte: int,
                          points: int = GRID_POINTS) -> tuple[np.ndarray, np.ndarray]:
    """Lognormal terminal price grid and its probability weights.

    `vol` is annualised realised volatility -- the forecast we actually believe,
    which is the whole point. Drift is set to zero: over a 1-4 day horizon any
    expected return is noise next to the volatility term, and assuming one would
    quietly turn a volatility strategy into a directional bet.
    """
    t = max(dte, 1) / TRADING_DAYS
    sigma = max(vol, 1e-4) * math.sqrt(t)

    lo = math.log(spot) - GRID_SIGMAS * sigma - 0.5 * sigma ** 2
    hi = math.log(spot) + GRID_SIGMAS * sigma - 0.5 * sigma ** 2
    log_grid = np.linspace(lo, hi, points)
    spot_grid = np.exp(log_grid)

    mu = math.log(spot) - 0.5 * sigma ** 2
    density = np.exp(-((log_grid - mu) ** 2) / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))

    weights = density * np.gradient(log_grid)
    total = weights.sum()
    if total > 0:
        weights = weights / total          # renormalise for grid truncation
    return spot_grid, weights


def evaluate(legs: Sequence, net_credit: float, spot: float, rv: float,
             dte: int, multiplier: int = 100) -> dict[str, float]:
    """Expected value and shape statistics for one structure, per contract.

    Returns dollars per contract, so results are directly comparable across
    underlyings with very different share prices.
    """
    spot_grid, weights = terminal_distribution(spot, rv, dte)
    payoff = payoff_curve(legs, net_credit, spot_grid)

    ev = float(np.sum(payoff * weights)) * multiplier
    pop = float(np.sum(weights[payoff > 0]))
    max_profit = float(payoff.max()) * multiplier
    max_loss = float(payoff.min()) * multiplier

    downside = payoff < 0
    expected_loss = (float(np.sum(payoff[downside] * weights[downside]))
                     * multiplier) if downside.any() else 0.0

    return {
        "ev": round(ev, 2),
        "pop": round(pop, 4),
        "max_profit": round(max_profit, 2),
        "max_loss": round(max_loss, 2),
        "expected_loss": round(expected_loss, 2),
        # EV as a fraction of capital at risk -- the comparable quality metric.
        "ev_per_risk": round(ev / abs(max_loss), 4) if max_loss < 0 else 0.0,
    }


def implied_vs_realized_gap(atm_iv: float, rv: float) -> float:
    """How much cheaper our vol forecast is than the market's, in vol points."""
    return round(atm_iv - rv, 4)
