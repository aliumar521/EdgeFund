"""Order submission and lifecycle for multi-leg option structures.

Everything Alpaca-specific about `mleg` orders is confined here.

Two facts learned by probing the live paper API, both of which the code depends
on and neither of which is in the bundled skills:

1. `limit_price` on an mleg order is **always positive**, for a net credit and
   a net debit alike. Direction is carried by each leg's side and
   position_intent, never by the sign of the price.
2. The response is a *parent* order carrying a `legs[]` array. That parent id
   is the only handle Alpaca gives us for treating the spread as one thing --
   positions come back leg by leg with nothing tying them together, so the
   grouping lives in our own database keyed by `strategy_uid`.

Fills are pursued with a limit chase rather than a market order. Option quotes
here come from the indicative feed, which is model-derived and can be wide; a
market order into a wide book gives the fill away, while a limit that steps
toward the natural price over a couple of minutes usually gets filled near mid.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from edgefund.core import db
from edgefund.core.models import SpreadCandidate
from edgefund.data.alpaca import AlpacaClient, AlpacaError

log = logging.getLogger("edgefund.router")

# Limit chase: start at mid, concede this fraction of the bid-ask each step.
CHASE_STEPS = 4
CHASE_CONCESSION = 0.25
MAX_CHASE_SECONDS = 210

_CLOSE_INTENT = {
    "sell_to_open": ("buy", "buy_to_close"),
    "buy_to_open": ("sell", "sell_to_close"),
}


def new_uid(underlying: str) -> str:
    return f"ef-{underlying.lower()}-{uuid.uuid4().hex[:10]}"


def _natural_price(candidate: SpreadCandidate) -> float:
    """The price we would get by crossing the spread on every leg.

    Selling a credit spread, that means hitting each bid and lifting each ask --
    the worst credit we would accept. It bounds the chase.
    """
    total = 0.0
    for leg in candidate.legs:
        if leg.side == "sell":
            total += leg.bid          # we sell at the bid
        else:
            total -= leg.ask          # we buy at the ask
    return abs(total)


def chase_price(candidate: SpreadCandidate, step: int) -> float:
    """Limit price for chase step `step` (0 = mid).

    Walks from the mid-derived price toward the natural price, so each step
    concedes a little more to get filled without ever going past the worst
    price we already decided was acceptable.
    """
    mid = abs(candidate.net_credit)
    natural = _natural_price(candidate)
    frac = min(1.0, step * CHASE_CONCESSION)
    price = mid + (natural - mid) * frac
    return max(0.01, round(price, 2))


def submit_spread(client: AlpacaClient, candidate: SpreadCandidate, qty: int,
                  entry_features: dict[str, Any], dry_run: bool) -> str | None:
    """Open a structure. Returns the strategy uid, or None if it was rejected.

    The database row is written *before* the API call so that an order which
    succeeds but whose response we never see still leaves a trace to reconcile.
    """
    uid = new_uid(candidate.underlying)
    limit_price = chase_price(candidate, step=0)
    max_loss_total = candidate.max_loss_per_contract * qty

    db.open_strategy(
        uid,
        {
            "underlying": candidate.underlying,
            "structure": candidate.structure,
            "sleeve": candidate.sleeve,
            "status": "pending",
            "qty": qty,
            "expiry": candidate.expiry.isoformat(),
            "net_credit": candidate.net_credit,
            "max_loss_total": max_loss_total,
            "parent_order_id": None,
            "entry_features": {**entry_features, "limit_price": limit_price},
            "dry_run": dry_run,
        },
        [
            {
                "symbol": l.symbol, "ratio_qty": l.ratio_qty, "side": l.side,
                "position_intent": l.position_intent, "strike": l.strike,
                "expiry": l.expiry.isoformat(), "opt_type": l.opt_type,
                "entry_price": l.mid, "entry_delta": l.delta, "entry_iv": l.iv,
            }
            for l in candidate.legs
        ],
    )

    if dry_run:
        db.update_strategy(uid, status="closed", exit_reason="dry_run",
                           realized_pnl=0.0, ts_close=datetime.now(timezone.utc).isoformat())
        db.log_decision(
            "entry", "dry_run",
            f"would open {candidate.structure} x{qty} @ {limit_price:.2f} "
            f"(max loss ${max_loss_total:,.0f})",
            candidate.underlying,
            {"uid": uid, "legs": [l.symbol for l in candidate.legs]},
        )
        return uid

    try:
        order = client.submit_mleg(
            legs=[l.to_alpaca_leg() for l in candidate.legs],
            qty=qty,
            limit_price=limit_price,
            client_order_id=uid,
        )
    except AlpacaError as exc:
        db.update_strategy(uid, status="failed", exit_reason=f"submit failed: {exc}")
        db.log_decision("entry", "rejected", str(exc), candidate.underlying, {"uid": uid})
        log.error("%s: submit failed: %s", uid, exc)
        return None

    db.update_strategy(uid, parent_order_id=order.get("id"), status="pending")
    db.log_decision(
        "entry", "submitted",
        f"{candidate.structure} x{qty} @ {limit_price:.2f} "
        f"(credit ${candidate.max_profit_per_contract * qty:,.0f}, "
        f"max loss ${max_loss_total:,.0f}, edge {candidate.edge_score:.2f})",
        candidate.underlying,
        {"uid": uid, "order_id": order.get("id"),
         "legs": [l.symbol for l in candidate.legs]},
    )
    log.info("%s: submitted %s x%d @ %.2f", uid, candidate.structure, qty, limit_price)
    return uid


def close_spread(client: AlpacaClient, strategy: dict[str, Any], reason: str,
                 limit_price: float | None = None, dry_run: bool = False) -> bool:
    """Close a structure by submitting the mirrored mleg order.

    Every opening intent has exactly one closing counterpart, so the reverse
    order is derived mechanically from what we recorded at entry rather than
    re-deriving it from broker positions.
    """
    uid = strategy["strategy_uid"]
    legs = strategy.get("legs") or []
    if not legs:
        log.error("%s: no legs recorded, cannot close", uid)
        return False

    close_legs = []
    for lg in legs:
        mapped = _CLOSE_INTENT.get(lg["position_intent"])
        if not mapped:
            log.error("%s: unmappable intent %s", uid, lg["position_intent"])
            return False
        side, intent = mapped
        close_legs.append({
            "symbol": lg["symbol"],
            "ratio_qty": str(lg["ratio_qty"]),
            "side": side,
            "position_intent": intent,
        })

    if dry_run:
        db.update_strategy(uid, status="closed", exit_reason=f"dry_run: {reason}",
                           ts_close=datetime.now(timezone.utc).isoformat())
        return True

    if limit_price is None:
        limit_price = _mark_to_close(client, strategy)
        if limit_price is None:
            log.error("%s: cannot price close order", uid)
            return False

    close_uid = f"{uid}-x{uuid.uuid4().hex[:4]}"
    try:
        order = client.submit_mleg(
            legs=close_legs, qty=int(strategy["qty"]),
            limit_price=limit_price, client_order_id=close_uid,
        )
    except AlpacaError as exc:
        db.log_decision("exit", "close_failed", f"{reason}: {exc}",
                        strategy["underlying"], {"uid": uid})
        log.error("%s: close failed: %s", uid, exc)
        return False

    db.update_strategy(uid, status="closing", close_order_id=order.get("id"),
                       exit_reason=reason)
    db.log_decision("exit", "closing", f"{reason} @ {limit_price:.2f}",
                    strategy["underlying"], {"uid": uid, "order_id": order.get("id")})
    log.info("%s: closing (%s) @ %.2f", uid, reason, limit_price)
    return True


def _mark_to_close(client: AlpacaClient, strategy: dict[str, Any]) -> float | None:
    """Current cost to close, priced marketably.

    Deliberately concedes toward the natural price: an exit that does not fill
    is not an exit, and on a stop or an expiry-day flatten the cost of lingering
    dominates the cost of a few cents of slippage.
    """
    legs = strategy.get("legs") or []
    snaps = client.option_snapshots([lg["symbol"] for lg in legs])
    if not snaps:
        return None

    total = 0.0
    for lg in legs:
        snap = snaps.get(lg["symbol"])
        if not snap:
            return None
        q = snap.get("latestQuote") or {}
        bid, ask = q.get("bp") or 0.0, q.get("ap") or 0.0
        if bid <= 0 or ask <= 0:
            return None
        mid = (bid + ask) / 2
        # Closing reverses the original direction.
        if lg["position_intent"] == "sell_to_open":
            total += mid + (ask - mid) * 0.5      # buying back: lean toward ask
        else:
            total -= mid - (mid - bid) * 0.5      # selling out: lean toward bid
    return max(0.01, round(abs(total), 2))


def sync_strategy_status(client: AlpacaClient, strategy: dict[str, Any]) -> str:
    """Reconcile one strategy against the broker's view of its orders.

    Alpaca is the source of truth for whether an order filled; our database is
    the source of truth for what the legs mean together.
    """
    uid = strategy["strategy_uid"]
    status = strategy["status"]

    if status == "pending" and strategy.get("parent_order_id"):
        try:
            order = client.get_order(strategy["parent_order_id"])
        except AlpacaError as exc:
            log.warning("%s: order lookup failed: %s", uid, exc)
            return status

        if order.get("status") == "filled":
            fill = _avg_fill(order)
            db.update_strategy(uid, status="open", entry_fill_price=fill)
            db.log_decision("entry", "filled", f"filled @ {fill}", strategy["underlying"],
                            {"uid": uid})
            return "open"
        if order.get("status") in {"canceled", "expired", "rejected"}:
            db.update_strategy(uid, status="failed",
                               exit_reason=f"entry {order.get('status')}")
            return "failed"

    elif status == "closing" and strategy.get("close_order_id"):
        try:
            order = client.get_order(strategy["close_order_id"])
        except AlpacaError as exc:
            log.warning("%s: close order lookup failed: %s", uid, exc)
            return status

        if order.get("status") == "filled":
            exit_price = _avg_fill(order)
            entry = strategy.get("entry_fill_price") or strategy.get("net_credit") or 0.0
            qty = int(strategy["qty"])
            # Credit structures: profit is entry credit minus what we pay back.
            # Debit structures: entry is stored negative, so the same expression
            # holds once the sign is respected.
            pnl = round((abs(entry) - (exit_price or 0.0)) * 100 * qty, 2)
            if strategy.get("net_credit", 0) < 0:
                pnl = round(((exit_price or 0.0) - abs(entry)) * 100 * qty, 2)
            db.update_strategy(
                uid, status="closed", exit_fill_price=exit_price, realized_pnl=pnl,
                ts_close=datetime.now(timezone.utc).isoformat())
            db.log_decision("exit", "closed",
                            f"{strategy.get('exit_reason') or 'closed'}; "
                            f"realized ${pnl:,.2f}",
                            strategy["underlying"], {"uid": uid, "pnl": pnl})
            return "closed"

    return status


def _avg_fill(order: dict[str, Any]) -> float | None:
    """Net fill price of a multi-leg order.

    Alpaca reports `filled_avg_price` per leg, so the structure's net price is
    the signed sum across legs rather than anything on the parent.
    """
    if order.get("filled_avg_price"):
        return abs(float(order["filled_avg_price"]))

    legs = order.get("legs") or []
    total = 0.0
    seen = False
    for leg in legs:
        price = leg.get("filled_avg_price")
        if price is None:
            continue
        seen = True
        ratio = float(leg.get("ratio_qty") or 1)
        total += (float(price) * ratio) * (1 if leg.get("side") == "sell" else -1)
    return abs(round(total, 4)) if seen else None
