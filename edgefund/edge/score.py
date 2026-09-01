"""The edge calculation -- the core of the whole strategy.

Thesis: implied volatility is usually richer than the volatility the underlying
subsequently realises. That gap (the variance risk premium) is the edge, it is
directly measurable from data we already have, and *its sign tells us which way
to trade*: sell premium when implied is rich, buy convexity when it is cheap.

Two inputs, both available from a single chain call plus stock bars:

  vrp_ratio  = atm_iv / realised_vol    >1 means the premium is rich
  term_slope = front_iv / short_end_iv  >1 means backwardation, i.e. stress

Backwardation is the warning sign. A high vrp_ratio during a stress event is
not free money -- it is the market correctly pricing a coming move. So the
score rewards a rich premium and penalises backwardation.

**Normalisation is cross-sectional first.** The naive approach -- z-scoring
against hardcoded constants -- fails immediately in practice: measured live,
every symbol in the universe showed a term slope near 1.4, so a prior centred
on 1.0 flagged the entire market as "stressed" and would have refused to trade.
A market-wide level is not an edge. What is tradeable is *relative* richness:
which names carry unusually rich premium compared with their peers right now.
Median and MAD across the universe give us that on day one with no history at
all, and a time-series z-score is blended in as the agent accumulates its own
observations.

Absolute floors still apply on top, so that "least cheap in a uniformly cheap
market" can never be mistaken for a genuine edge.

Trend deliberately does NOT enter the score. It selects the *structure*
(put-credit in an uptrend, call-credit in a downtrend, iron condor when flat);
conflating a directional view with a volatility edge is how a vol strategy
quietly turns into a punt on direction.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

from edgefund.core.db import connect
from edgefund.core.models import EdgeSnapshot
from edgefund.data import vol as volmod
from edgefund.data.alpaca import AlpacaClient, parse_occ

log = logging.getLogger("edgefund.edge")

# Absolute gates. Cross-sectional ranking says which name is richest; these say
# whether the level is worth trading at all.
MIN_VRP_TO_SELL = 1.05
MAX_VRP_TO_BUY = 0.95

# Cold-start priors, used only when the universe is too small to rank.
DEFAULT_VRP_CENTRE = 1.25
VRP_DISPERSION = 0.25
DEFAULT_TERM_CENTRE = 1.30
TERM_DISPERSION = 0.15

W_VRP = 1.0
W_TERM = 0.6
MIN_HISTORY_FOR_ROLLING = 20
MIN_UNIVERSE_FOR_XSECTION = 4


@dataclass
class RawEdge:
    """Per-symbol measurements, before any normalisation."""

    underlying: str
    spot: float
    rv: float
    atm_iv: float
    vrp_ratio: float
    term_slope: float
    trend: float
    expiry: str
    dte: int
    detail: dict[str, Any] = field(default_factory=dict)


def _robust_centre_scale(values: list[float], fb_centre: float,
                         fb_scale: float) -> tuple[float, float, str]:
    """Median and MAD-derived sigma; falls back to priors on a thin universe."""
    clean = [v for v in values if v is not None and v == v]
    if len(clean) < MIN_UNIVERSE_FOR_XSECTION:
        return fb_centre, fb_scale, "prior"
    centre = statistics.median(clean)
    mad = statistics.median([abs(v - centre) for v in clean])
    scale = mad * 1.4826
    if scale < 1e-4:
        scale = fb_scale
    return centre, scale, "xsection"


def _history(underlying: str, column: str, limit: int = 120) -> list[float]:
    with connect() as conn:
        rows = conn.execute(
            f"SELECT {column} v FROM edge_snapshots WHERE underlying=? "
            "ORDER BY id DESC LIMIT ?",
            (underlying, limit),
        ).fetchall()
    return [r["v"] for r in rows if r["v"] is not None]


def _timeseries_z(value: float, underlying: str, column: str) -> float | None:
    hist = _history(underlying, column)
    if len(hist) < MIN_HISTORY_FOR_ROLLING:
        return None
    mean = statistics.fmean(hist)
    sd = statistics.pstdev(hist)
    if sd < 1e-6:
        return None
    return (value - mean) / sd


def atm_iv_from_chain(snapshots: dict[str, Any], spot: float) -> tuple[float, dict[str, Any]]:
    """Implied vol at the money, blended across the nearest strikes.

    Takes the closest strike either side of spot and, at each, averages the call
    and put IV. Single-contract IV off the indicative feed is noisy; averaging
    the straddle and interpolating across the two nearest strikes is materially
    steadier for very little work.
    """
    by_strike: dict[float, dict[str, float]] = {}
    for sym, snap in snapshots.items():
        iv = snap.get("impliedVolatility")
        if not iv or iv <= 0:
            continue
        try:
            meta = parse_occ(sym)
        except Exception:
            continue
        by_strike.setdefault(meta["strike"], {})[meta["opt_type"]] = iv

    if not by_strike:
        return 0.0, {"reason": "no IV in chain"}

    def blended(strike: float) -> float:
        legs = by_strike[strike]
        vals = [v for v in (legs.get("call"), legs.get("put")) if v]
        return sum(vals) / len(vals) if vals else 0.0

    strikes = sorted(by_strike)
    below = [s for s in strikes if s <= spot]
    above = [s for s in strikes if s >= spot]

    if below and above:
        lo, hi = below[-1], above[0]
        iv_lo, iv_hi = blended(lo), blended(hi)
        if iv_lo and iv_hi:
            iv = iv_lo if hi == lo else iv_lo + (iv_hi - iv_lo) * (spot - lo) / (hi - lo)
            return iv, {"strikes": [lo, hi], "iv_lo": round(iv_lo, 4), "iv_hi": round(iv_hi, 4)}
        nearest = lo if iv_lo else hi
    else:
        nearest = min(strikes, key=lambda s: abs(s - spot))

    return blended(nearest), {"strikes": [nearest]}


def compute_raw(client: AlpacaClient, underlying: str, target_expiry: str,
                dte: int) -> RawEdge | None:
    """Measure one underlying. No normalisation, no scoring."""
    today = datetime.now(timezone.utc).date()

    try:
        spot = client.latest_stock_price([underlying]).get(underlying, 0.0)
    except Exception as exc:
        log.warning("%s: spot fetch failed: %s", underlying, exc)
        return None
    if spot <= 0:
        return None

    daily = client.stock_bars(
        underlying, "1Day", start=(today - timedelta(days=90)).isoformat(),
    ).get(underlying, [])
    if len(daily) < 12:
        log.warning("%s: only %d daily bars, skipping", underlying, len(daily))
        return None

    intraday: list[dict[str, Any]] = []
    if dte <= 2:
        try:
            intraday = client.stock_bars(
                underlying, "5Min",
                start=(today - timedelta(days=9)).isoformat(), limit=5000,
            ).get(underlying, [])
        except Exception as exc:
            log.info("%s: intraday bars unavailable (%s), using daily", underlying, exc)

    rv, estimator = volmod.realized_vol(daily, intraday, dte)
    if rv <= 0.001:
        return None

    band = max(spot * 0.06, 3.0)
    front = client.option_chain(
        underlying, expiration_date=target_expiry,
        strike_gte=spot - band, strike_lte=spot + band,
    )
    if not front:
        log.warning("%s: empty chain for %s", underlying, target_expiry)
        return None

    atm_iv, iv_detail = atm_iv_from_chain(front, spot)
    if atm_iv <= 0:
        return None

    term_slope, far_iv = 1.0, 0.0
    try:
        far = client.option_chain(
            underlying,
            expiration_date_gte=(today + timedelta(days=7)).isoformat(),
            expiration_date_lte=(today + timedelta(days=16)).isoformat(),
            strike_gte=spot - band, strike_lte=spot + band,
        )
        if far:
            far_iv, _ = atm_iv_from_chain(far, spot)
            if far_iv > 0:
                term_slope = atm_iv / far_iv
    except Exception as exc:
        log.info("%s: term structure unavailable (%s), assuming flat", underlying, exc)

    return RawEdge(
        underlying=underlying,
        spot=round(spot, 4),
        rv=round(rv, 4),
        atm_iv=round(atm_iv, 4),
        vrp_ratio=round(atm_iv / rv, 4),
        term_slope=round(term_slope, 4),
        trend=volmod.trend_score(daily),
        expiry=target_expiry,
        dte=dte,
        detail={
            "rv_estimator": estimator,
            "far_iv": round(far_iv, 4),
            "atm_detail": iv_detail,
            "daily_bars": len(daily),
            "intraday_bars": len(intraday),
        },
    )


def score_universe(raws: list[RawEdge]) -> list[EdgeSnapshot]:
    """Turn raw measurements into comparable edge scores.

    Normalises across the universe so a market-wide vol level cancels out and
    only relative richness survives, then blends in each symbol's own history
    once enough observations exist.
    """
    if not raws:
        return []

    vrp_centre, vrp_scale, vrp_basis = _robust_centre_scale(
        [r.vrp_ratio for r in raws], DEFAULT_VRP_CENTRE, VRP_DISPERSION)
    term_centre, term_scale, term_basis = _robust_centre_scale(
        [r.term_slope for r in raws], DEFAULT_TERM_CENTRE, TERM_DISPERSION)

    out: list[EdgeSnapshot] = []
    for r in raws:
        xs_vrp = (r.vrp_ratio - vrp_centre) / vrp_scale
        xs_term = (r.term_slope - term_centre) / term_scale

        ts_vrp = _timeseries_z(r.vrp_ratio, r.underlying, "vrp_ratio")
        ts_term = _timeseries_z(r.term_slope, r.underlying, "term_slope")

        z_vrp = xs_vrp if ts_vrp is None else 0.5 * xs_vrp + 0.5 * ts_vrp
        z_term = xs_term if ts_term is None else 0.5 * xs_term + 0.5 * ts_term

        edge_score = round(W_VRP * z_vrp - W_TERM * z_term, 4)

        # Absolute gates: relative ranking must not talk us into selling
        # premium that is cheap in absolute terms, or buying premium that is
        # rich. Neutralise the score rather than let the ranking carry it.
        gated = ""
        if edge_score > 0 and r.vrp_ratio < MIN_VRP_TO_SELL:
            edge_score, gated = 0.0, (
                f"ranked rich vs peers but premium is not rich in absolute "
                f"terms (vrp {r.vrp_ratio:.2f} < {MIN_VRP_TO_SELL})")
        elif edge_score < 0 and r.vrp_ratio > MAX_VRP_TO_BUY:
            edge_score, gated = 0.0, (
                f"score negative (term structure) but vol is not cheap enough "
                f"to buy (vrp {r.vrp_ratio:.2f} > {MAX_VRP_TO_BUY}) -- no trade")

        if xs_term > 2.0:
            regime = "stressed"
        elif xs_vrp > 2.0 and xs_term > 1.0:
            regime = "event"
        elif abs(xs_vrp) < 0.6 and xs_term < 0.5:
            regime = "calm"
        else:
            regime = "normal"

        out.append(EdgeSnapshot(
            underlying=r.underlying, spot=r.spot, rv=r.rv, atm_iv=r.atm_iv,
            vrp_ratio=r.vrp_ratio, term_slope=r.term_slope, trend=r.trend,
            edge_score=edge_score, regime=regime,
            detail={
                **r.detail,
                "expiry": r.expiry,
                "dte": r.dte,
                "z_vrp": round(z_vrp, 3),
                "z_term": round(z_term, 3),
                "xs_vrp": round(xs_vrp, 3),
                "xs_term": round(xs_term, 3),
                "basis": {"vrp": vrp_basis, "term": term_basis},
                "universe_centre": {"vrp": round(vrp_centre, 4),
                                    "term": round(term_centre, 4)},
                "gated": gated,
            },
        ))
    return out


def pick_expiry(client: AlpacaClient, underlying: str, max_dte: int) -> tuple[str, int] | None:
    """Nearest expiry that still has meaningful life left.

    Same-day expiries are skipped: with the watchdog forced to flatten by 15:30
    there is not enough runway to open one, and gamma near the pin is the worst
    risk/reward on the board.
    """
    today = datetime.now(timezone.utc).date()
    try:
        expiries = client.expirations(underlying, within_days=max_dte + 6)
    except Exception as exc:
        log.warning("%s: expiry lookup failed: %s", underlying, exc)
        return None

    for exp in expiries:
        dte = (date.fromisoformat(exp) - today).days
        if 1 <= dte <= max_dte:
            return exp, dte
    return None


def scan_universe(client: AlpacaClient, symbols: list[str],
                  max_dte: int) -> list[EdgeSnapshot]:
    """Measure every symbol, then score them against each other."""
    raws: list[RawEdge] = []
    for sym in symbols:
        picked = pick_expiry(client, sym, max_dte)
        if not picked:
            log.info("%s: no expiry within %d DTE", sym, max_dte)
            continue
        raw = compute_raw(client, sym, picked[0], picked[1])
        if raw:
            raws.append(raw)
    return score_universe(raws)
