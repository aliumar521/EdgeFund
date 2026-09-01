"""Construct defined-risk option structures from a live chain.

Every structure here is defined-risk by construction. That is not only a risk
preference -- Alpaca permits no naked short options at any approval level, so a
short leg without a protective long leg would simply be rejected.

Selection separates the two questions cleanly:

  * the *edge score* decides whether we sell premium or buy convexity
  * the *trend* decides which side of the underlying we express it on

Within a chosen structure type, strikes are not picked by rule of thumb. Every
viable strike combination is enumerated and scored by expected value under our
own realised-vol forecast (see payoff.py), and the best one wins. That way the
same conviction -- implied vol is too rich here -- decides not just *whether*
to trade but *which* contracts give the most edge per dollar risked.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Iterable

from edgefund.core.config import SETTINGS
from edgefund.core.models import EdgeSnapshot, Leg, SpreadCandidate, StrategyDirective
from edgefund.data.alpaca import parse_occ
from edgefund.strategy import payoff as pf

log = logging.getLogger("edgefund.spreads")

# Liquidity gates. The indicative feed gives no size we can trust, so the
# bid-ask width relative to mid is our only usable liquidity proxy.
MIN_LEG_MID = 0.03
MAX_REL_SPREAD = 1.0

# Search space for the short strike of a credit structure.
SHORT_DELTA_MIN = 0.10
SHORT_DELTA_MAX = 0.36
# Long strike sits 1..N strikes further out.
MAX_STRIKES_OUT = 8
# Wings must be wide enough to be worth trading. Ranking purely on expected
# value per unit of risk drives the search toward the narrowest possible wings
# -- max loss shrinks faster than expected value does -- which produced $1-wide
# condors on a $296 underlying at 1 DTE. Those maximise the ratio while sitting
# on a knife edge of gamma risk and needing an exact fill on four strikes.
MIN_WIDTH_PCT = 0.006
MAX_WIDTH_PCT = 0.035
# Penny credits are not worth four legs of execution risk.
MIN_NET_CREDIT = 0.10

# A structure must clear these to be worth the buying power it consumes.
MIN_EV_PER_RISK = 0.02       # 2c of expected value per dollar at risk
MIN_POP = 0.55


class ChainView:
    """Indexed view over a chain snapshot: strike -> contract, per option type."""

    def __init__(self, snapshots: dict[str, Any], expiry: date):
        self.expiry = expiry
        self.calls: dict[float, dict[str, Any]] = {}
        self.puts: dict[float, dict[str, Any]] = {}

        for sym, snap in snapshots.items():
            try:
                meta = parse_occ(sym)
            except Exception:
                continue
            if meta["expiry"] != expiry:
                continue

            quote = snap.get("latestQuote") or {}
            greeks = snap.get("greeks") or {}
            bid, ask = quote.get("bp") or 0.0, quote.get("ap") or 0.0
            if bid <= 0 or ask <= 0 or ask < bid:
                continue

            row = {
                "symbol": sym,
                "strike": meta["strike"],
                "opt_type": meta["opt_type"],
                "bid": bid,
                "ask": ask,
                "mid": round((bid + ask) / 2, 4),
                "delta": greeks.get("delta") or 0.0,
                "iv": snap.get("impliedVolatility") or 0.0,
            }
            target = self.calls if meta["opt_type"] == "call" else self.puts
            target[meta["strike"]] = row

    def side(self, opt_type: str) -> dict[float, dict[str, Any]]:
        return self.calls if opt_type == "call" else self.puts

    def tradable(self, row: dict[str, Any]) -> bool:
        if row["mid"] < MIN_LEG_MID:
            return False
        return (row["ask"] - row["bid"]) / row["mid"] <= MAX_REL_SPREAD

    def ordered(self, opt_type: str, otm_first: bool) -> list[dict[str, Any]]:
        """Tradable contracts sorted by strike.

        For puts, "further out of the money" means a lower strike; for calls, a
        higher one. `otm_first` returns them in increasing OTM order so the
        long-leg search can simply walk forward.
        """
        rows = [r for r in self.side(opt_type).values() if self.tradable(r)]
        reverse = (opt_type == "put") if otm_first else (opt_type == "call")
        return sorted(rows, key=lambda r: r["strike"], reverse=reverse)


def _leg(row: dict[str, Any], side: str, intent: str, expiry: date) -> Leg:
    return Leg(
        symbol=row["symbol"], ratio_qty=1, side=side, position_intent=intent,
        strike=row["strike"], expiry=expiry, opt_type=row["opt_type"],
        bid=row["bid"], ask=row["ask"], delta=row["delta"], iv=row["iv"],
    )


def _width_ok(width: float, spot: float) -> bool:
    return spot * MIN_WIDTH_PCT <= width <= spot * MAX_WIDTH_PCT


def build_credit_spread(chain: ChainView, snap: EdgeSnapshot, opt_type: str,
                        dte: int) -> SpreadCandidate | None:
    """Best vertical credit spread on one side, by expected value.

    Enumerates every short strike in the target delta band against every
    protective strike within the allowed width, then keeps the combination with
    the most expected value per dollar of risk.
    """
    otm = chain.ordered(opt_type, otm_first=True)
    if len(otm) < 2:
        return None

    # Shorts sit nearest the money, so walk the OTM ordering in reverse.
    shorts = [r for r in otm
              if SHORT_DELTA_MIN <= abs(r["delta"]) <= SHORT_DELTA_MAX]
    if not shorts:
        return None

    best: tuple[float, SpreadCandidate] | None = None
    index = {r["strike"]: i for i, r in enumerate(otm)}

    for short_row in shorts:
        si = index[short_row["strike"]]
        for step in range(1, MAX_STRIKES_OUT + 1):
            li = si + step
            if li >= len(otm):
                break
            long_row = otm[li]
            width = abs(short_row["strike"] - long_row["strike"])
            if not _width_ok(width, snap.spot):
                continue

            net_credit = round(short_row["mid"] - long_row["mid"], 4)
            if net_credit < MIN_NET_CREDIT or net_credit >= width:
                continue

            legs = [
                _leg(short_row, "sell", "sell_to_open", chain.expiry),
                _leg(long_row, "buy", "buy_to_open", chain.expiry),
            ]
            stats = pf.evaluate(legs, net_credit, snap.spot, snap.rv, dte)
            if stats["ev"] <= 0 or stats["ev_per_risk"] < MIN_EV_PER_RISK:
                continue
            if stats["pop"] < MIN_POP:
                continue

            if best is None or stats["ev_per_risk"] > best[0]:
                structure = ("put_credit_spread" if opt_type == "put"
                             else "call_credit_spread")
                best = (stats["ev_per_risk"], SpreadCandidate(
                    underlying=snap.underlying, structure=structure, sleeve="core",
                    legs=legs, net_credit=net_credit, width=width,
                    max_loss_per_contract=round(abs(stats["max_loss"]), 2),
                    max_profit_per_contract=round(stats["max_profit"], 2),
                    short_delta=abs(short_row["delta"]), dte=dte,
                    expiry=chain.expiry, edge_score=snap.edge_score,
                    features=_features(snap, stats, net_credit / width, short_row),
                ))

    return best[1] if best else None


def build_iron_condor(chain: ChainView, snap: EdgeSnapshot,
                      dte: int) -> SpreadCandidate | None:
    """Both wings at once.

    Capital-efficient: only one side can finish in the money, so margin is the
    wider wing minus the total credit while we collect premium from both. The
    combined structure is re-scored as a whole rather than assumed to be the
    sum of its wings.
    """
    put_side = build_credit_spread(chain, snap, "put", dte)
    call_side = build_credit_spread(chain, snap, "call", dte)
    if not put_side or not call_side:
        return None

    short_put = max(l.strike for l in put_side.legs)
    short_call = min(l.strike for l in call_side.legs)
    if short_put >= short_call:          # wings must not overlap
        return None

    legs = [*put_side.legs, *call_side.legs]
    net_credit = round(put_side.net_credit + call_side.net_credit, 4)
    width = max(put_side.width, call_side.width)
    if net_credit >= width or net_credit < MIN_NET_CREDIT:
        return None

    stats = pf.evaluate(legs, net_credit, snap.spot, snap.rv, dte)
    if stats["ev"] <= 0 or stats["ev_per_risk"] < MIN_EV_PER_RISK:
        return None

    return SpreadCandidate(
        underlying=snap.underlying, structure="iron_condor", sleeve="core",
        legs=legs, net_credit=net_credit, width=width,
        max_loss_per_contract=round(abs(stats["max_loss"]), 2),
        max_profit_per_contract=round(stats["max_profit"], 2),
        short_delta=max(put_side.short_delta, call_side.short_delta),
        dte=dte, expiry=chain.expiry, edge_score=snap.edge_score,
        features=_features(snap, stats, net_credit / width, None,
                           extra={"short_put": short_put, "short_call": short_call,
                                  "put_wing": put_side.width,
                                  "call_wing": call_side.width}),
    )


def build_debit_spread(chain: ChainView, snap: EdgeSnapshot, opt_type: str,
                       dte: int) -> SpreadCandidate | None:
    """Directional long-convexity structure for the satellite sleeve.

    Used when implied vol is measurably *cheap*, so we buy optionality at a
    discount rather than paying up for it. Scored on the same expected-value
    basis, which for a debit structure means the market's implied move has to be
    smaller than the one our realised-vol forecast expects.
    """
    otm = chain.ordered(opt_type, otm_first=True)
    if len(otm) < 2:
        return None

    longs = [r for r in otm if 0.30 <= abs(r["delta"]) <= 0.62]
    if not longs:
        return None

    best: tuple[float, SpreadCandidate] | None = None
    index = {r["strike"]: i for i, r in enumerate(otm)}

    for long_row in longs:
        li = index[long_row["strike"]]
        for step in range(1, MAX_STRIKES_OUT + 1):
            si = li + step
            if si >= len(otm):
                break
            short_row = otm[si]
            width = abs(short_row["strike"] - long_row["strike"])
            if not _width_ok(width, snap.spot):
                continue

            net_debit = round(long_row["mid"] - short_row["mid"], 4)
            if net_debit <= 0 or net_debit >= width:
                continue

            legs = [
                _leg(long_row, "buy", "buy_to_open", chain.expiry),
                _leg(short_row, "sell", "sell_to_open", chain.expiry),
            ]
            stats = pf.evaluate(legs, -net_debit, snap.spot, snap.rv, dte)
            if stats["ev"] <= 0 or stats["ev_per_risk"] < MIN_EV_PER_RISK:
                continue

            if best is None or stats["ev_per_risk"] > best[0]:
                structure = ("call_debit_spread" if opt_type == "call"
                             else "put_debit_spread")
                best = (stats["ev_per_risk"], SpreadCandidate(
                    underlying=snap.underlying, structure=structure,
                    sleeve="satellite", legs=legs,
                    net_credit=-net_debit, width=width,
                    max_loss_per_contract=round(abs(stats["max_loss"]), 2),
                    max_profit_per_contract=round(stats["max_profit"], 2),
                    short_delta=abs(short_row["delta"]), dte=dte,
                    expiry=chain.expiry, edge_score=snap.edge_score,
                    features=_features(snap, stats, net_debit / width, long_row),
                ))

    return best[1] if best else None


def _features(snap: EdgeSnapshot, stats: dict[str, float], credit_to_width: float,
              key_row: dict[str, Any] | None,
              extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Entry features, stored so the reflection loop can learn what worked."""
    out: dict[str, Any] = {
        "vrp_ratio": snap.vrp_ratio, "term_slope": snap.term_slope,
        "trend": snap.trend, "atm_iv": snap.atm_iv, "rv": snap.rv,
        "regime": snap.regime, "edge_score": snap.edge_score,
        "credit_to_width": round(credit_to_width, 4),
        "spot_at_entry": snap.spot,
        "ev": stats["ev"], "ev_per_risk": stats["ev_per_risk"],
        "pop": stats["pop"],
        "iv_rv_gap": pf.implied_vs_realized_gap(snap.atm_iv, snap.rv),
    }
    if key_row:
        out["short_iv"] = round(key_row.get("iv") or 0.0, 4)
    if extra:
        out.update(extra)
    return out


def build_for_snapshot(chain: ChainView, snap: EdgeSnapshot,
                       directive: StrategyDirective, dte: int) -> SpreadCandidate | None:
    """Pick and build the structure that matches the measured edge.

    Edge sign chooses the sleeve; trend (nudged by the directive's directional
    bias) chooses how to express it. Where more than one structure is viable the
    one with the higher expected value per unit of risk wins.
    """
    bias = snap.trend + directive.directional_bias * 0.5

    if snap.edge_score >= directive.min_edge_score:
        if bias > 0.30:
            wanted = ["put_credit_spread", "iron_condor"]
        elif bias < -0.30:
            wanted = ["call_credit_spread", "iron_condor"]
        else:
            wanted = ["iron_condor", "put_credit_spread", "call_credit_spread"]

        if directive.preferred_structures:
            narrowed = [c for c in wanted if c in directive.preferred_structures]
            wanted = narrowed or wanted

        built: list[SpreadCandidate] = []
        for structure in wanted:
            if structure == "iron_condor":
                c = build_iron_condor(chain, snap, dte)
            else:
                c = build_credit_spread(
                    chain, snap,
                    "put" if structure == "put_credit_spread" else "call", dte)
            if c:
                built.append(c)

        if not built:
            return None
        return max(built, key=lambda c: c.features.get("ev_per_risk", 0.0))

    if snap.edge_score <= -directive.min_edge_score:
        opt_type = "call" if bias >= 0 else "put"
        primary = build_debit_spread(chain, snap, opt_type, dte)
        if primary:
            return primary
        return build_debit_spread(
            chain, snap, "put" if opt_type == "call" else "call", dte)

    return None
