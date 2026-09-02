"""Position watchdog. Every ~60s through the session, every 30 minutes outside
it (an "idle" pass). Uses no AI at all.

This is the component that has to be reliable, so it is deliberately the dumbest
one: fixed thresholds, no model calls, no judgement. The AI layer can change
what we open; it can never change how an open position is protected.

Responsibilities, in priority order:

1. reconcile fills against the broker
2. flatten everything if the kill switch has tripped
3. close positions that hit a profit target, stop, delta stop, or expiry cutoff
4. chase entry limit orders that have not filled

The expiry cutoff is the one rule that is never skipped. A short option left
open into expiry risks assignment and pin gamma for no compensating edge, so
anything expiring today is closed at 15:30 ET regardless of P&L.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo

from edgefund.core import db, params
from edgefund.core.config import SETTINGS
from edgefund.data.alpaca import AlpacaClient, AlpacaError
from edgefund.execute.router import (
    CHASE_STEPS, close_spread, sync_strategy_status,
)
from edgefund.risk.limits import breaches_kill_switch, state_from_account

log = logging.getLogger("edgefund.watchdog")

ET = ZoneInfo("America/New_York")

# Anything expiring today is flat by this time, no exceptions.
EXPIRY_FLATTEN_TIME = time(15, 30)
# Entry orders are chased for a few minutes, then abandoned.
ENTRY_CHASE_AFTER_SECONDS = 45


def now_et() -> datetime:
    return datetime.now(ET)


def market_phase(client: AlpacaClient) -> dict[str, Any]:
    clock = client.clock()
    return {
        "is_open": bool(clock.get("is_open")),
        "now_et": now_et(),
        "next_open": clock.get("next_open"),
        "next_close": clock.get("next_close"),
    }


def current_close_cost(client: AlpacaClient, strategy: dict[str, Any]) -> float | None:
    """What it would cost right now to close the structure, per contract.

    Priced at mid rather than marketably: this figure drives exit *decisions*,
    and pricing it pessimistically would trip stops that the real market never
    reached. The marketable concession is applied later, when the closing order
    is actually sent.
    """
    legs = strategy.get("legs") or []
    if not legs:
        return None
    try:
        snaps = client.option_snapshots([lg["symbol"] for lg in legs])
    except AlpacaError as exc:
        log.warning("%s: quote fetch failed: %s", strategy["strategy_uid"], exc)
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
        # Closing reverses the opening direction: shorts get bought back.
        total += mid if lg["position_intent"] == "sell_to_open" else -mid
    return round(total, 4)


def short_leg_deltas(client: AlpacaClient, strategy: dict[str, Any]) -> list[float]:
    legs = [lg for lg in (strategy.get("legs") or [])
            if lg["position_intent"] == "sell_to_open"]
    if not legs:
        return []
    try:
        snaps = client.option_snapshots([lg["symbol"] for lg in legs])
    except AlpacaError:
        return []
    out = []
    for lg in legs:
        greeks = (snaps.get(lg["symbol"]) or {}).get("greeks") or {}
        if greeks.get("delta") is not None:
            out.append(abs(greeks["delta"]))
    return out


def evaluate_exit(strategy: dict[str, Any], close_cost: float,
                  short_deltas: list[float], now: datetime) -> str | None:
    """Return an exit reason, or None to hold.

    Credit and debit structures are handled separately because "profit" means
    opposite things: a credit structure wins as its cost to close falls toward
    zero, a debit structure wins as its value rises.
    """
    entry = float(strategy.get("entry_fill_price")
                  or abs(strategy.get("net_credit") or 0.0))
    if entry <= 0:
        return None

    is_credit = float(strategy.get("net_credit") or 0.0) > 0
    expiry = date.fromisoformat(strategy["expiry"])

    # 1. expiry cutoff -- unconditional
    if expiry <= now.date() and now.time() >= EXPIRY_FLATTEN_TIME:
        return f"expiry flatten ({expiry} expires today, past {EXPIRY_FLATTEN_TIME:%H:%M} ET)"

    if is_credit:
        # 2. profit target: bought back for a fraction of the credit received.
        # The epsilon keeps an exact-boundary hit from being lost to floating
        # point (1.00 * (1 - 0.55) evaluates just below 0.45).
        target = params.get("profit_target_pct", SETTINGS.profit_target_pct)
        if close_cost <= entry * (1 - target) + 1e-9:
            captured = (entry - close_cost) / entry
            return f"profit target ({captured:.0%} of credit captured)"
        # 3. stop: costs a multiple of the credit to close
        stop_mult = params.get("stop_loss_mult", SETTINGS.stop_loss_mult)
        if close_cost >= entry * stop_mult:
            return (f"stop loss (close cost {close_cost:.2f} >= "
                    f"{stop_mult}x credit {entry:.2f})")
        # 4. delta stop: short strike is being challenged
        delta_stop = params.get("delta_stop", SETTINGS.delta_stop)
        if short_deltas and max(short_deltas) >= delta_stop:
            return (f"delta stop (short leg delta {max(short_deltas):.2f} >= "
                    f"{delta_stop})")
    else:
        value = -close_cost          # debit structures are worth the reverse sign
        if value >= entry * 2.0:
            return f"profit target (value {value:.2f} >= 2x debit {entry:.2f})"
        if value <= entry * 0.45:
            return f"stop loss (value {value:.2f} <= 45% of debit {entry:.2f})"

    return None


def chase_entry(client: AlpacaClient, strategy: dict[str, Any]) -> None:
    """Nudge an unfilled entry order toward the natural price.

    Quotes on the indicative feed are wide, so an order resting at mid can sit
    unfilled indefinitely. Rather than crossing immediately and giving the
    spread away, the limit concedes a little at a time and the order is
    abandoned entirely once the concession budget is spent.
    """
    uid = strategy["strategy_uid"]
    order_id = strategy.get("parent_order_id")
    if not order_id:
        return

    try:
        order = client.get_order(order_id)
    except AlpacaError:
        return
    if order.get("status") not in {"new", "accepted", "partially_filled", "pending_new"}:
        return

    submitted = order.get("submitted_at") or order.get("created_at")
    if not submitted:
        return
    try:
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(submitted.replace("Z", "+00:00"))).total_seconds()
    except ValueError:
        return
    if age < ENTRY_CHASE_AFTER_SECONDS:
        return

    step = int(age // ENTRY_CHASE_AFTER_SECONDS)
    if step > CHASE_STEPS:
        try:
            client.cancel_order(order_id)
            db.update_strategy(uid, status="failed",
                               exit_reason="entry not filled within chase budget")
            db.log_decision("entry", "abandoned",
                            f"no fill after {int(age)}s of chasing",
                            strategy["underlying"], {"uid": uid})
        except AlpacaError as exc:
            log.warning("%s: cancel failed: %s", uid, exc)
        return

    features = strategy.get("entry_features") or {}
    original = float(features.get("limit_price") or 0.0)
    if original <= 0:
        return

    # Concede 8% of the original limit per step; for a credit structure that
    # means accepting progressively less premium.
    concession = 1 - 0.08 * step if float(strategy.get("net_credit") or 0) > 0 \
        else 1 + 0.08 * step
    new_price = max(0.01, round(original * concession, 2))
    try:
        client.replace_order(order_id, new_price)
        db.log_decision("entry", "chase",
                        f"step {step}: reprice {original:.2f} -> {new_price:.2f}",
                        strategy["underlying"], {"uid": uid})
    except AlpacaError as exc:
        log.info("%s: reprice failed (%s)", uid, exc)


def run_watchdog(client: AlpacaClient, dry_run: bool | None = None,
                 idle: bool = False) -> dict[str, Any]:
    """One watchdog pass. Safe to call when the market is closed.

    `idle` is the off-hours mode: it still marks equity and writes a heartbeat,
    which also proves the API credentials are alive, but skips reconciling every
    strategy against the broker. Nothing can fill or be exited while the options
    market is shut, so that per-strategy sweep is pure cost overnight.
    """
    dry_run = SETTINGS.dry_run if dry_run is None else dry_run
    phase = market_phase(client)
    now = phase["now_et"]

    summary: dict[str, Any] = {
        "market_open": phase["is_open"], "idle": idle, "checked": 0,
        "closed": 0, "chased": 0, "actions": [],
    }

    account = client.account()
    if not idle:
        for strategy in db.active_strategies():
            try:
                sync_strategy_status(client, strategy)
            except Exception as exc:
                log.warning("%s: sync failed: %s", strategy["strategy_uid"], exc)

    strategies = db.active_strategies()
    state = state_from_account(account, strategies)
    db.record_equity(
        equity=state.equity, cash=float(account.get("cash") or 0),
        options_bp=state.options_bp, day_pnl_pct=state.day_pnl_pct,
    )
    summary["equity"] = state.equity
    summary["day_pnl_pct"] = state.day_pnl_pct

    # Bail out before the kill switch, not after. day_pnl_pct does not reset
    # until the next session, so a day that ends past the threshold would
    # otherwise have us firing closing orders at a shut market every pass all
    # night. Nothing can be flattened while options are not trading; the switch
    # is evaluated on the first pass after the next open instead.
    if not phase["is_open"]:
        db.heartbeat("watchdog",
                     f"{'idle' if idle else 'market closed'}; "
                     f"{len(strategies)} tracked")
        summary["actions"].append("market closed, no exits evaluated")
        return summary

    # Kill switch outranks every other consideration.
    if breaches_kill_switch(state, SETTINGS.limits):
        db.log_decision("risk", "kill_switch",
                        f"day P&L {state.day_pnl_pct:.2%} -- flattening book")
        for strategy in strategies:
            if strategy["status"] == "open":
                close_spread(client, strategy, "kill switch", dry_run=dry_run)
                summary["closed"] += 1
        summary["halted"] = "kill_switch"
        return summary

    for strategy in strategies:
        summary["checked"] += 1
        status = strategy["status"]

        if status == "pending":
            chase_entry(client, strategy)
            summary["chased"] += 1
            continue
        if status != "open":
            continue

        close_cost = current_close_cost(client, strategy)
        if close_cost is None:
            continue

        deltas = short_leg_deltas(client, strategy)
        reason = evaluate_exit(strategy, close_cost, deltas, now)
        if not reason:
            continue

        if close_spread(client, strategy, reason, dry_run=dry_run):
            summary["closed"] += 1
            summary["actions"].append(f"{strategy['underlying']}: {reason}")

    db.heartbeat("watchdog",
                 f"checked={summary['checked']} closed={summary['closed']} "
                 f"equity={state.equity:,.0f}")
    return summary


def flatten_all(client: AlpacaClient, reason: str,
                dry_run: bool | None = None) -> int:
    """Close every open structure. Used by the kill switch and the final sweep."""
    dry_run = SETTINGS.dry_run if dry_run is None else dry_run
    count = 0
    for strategy in db.active_strategies():
        if strategy["status"] in {"open", "pending"}:
            if strategy["status"] == "pending" and strategy.get("parent_order_id"):
                try:
                    client.cancel_order(strategy["parent_order_id"])
                    db.update_strategy(strategy["strategy_uid"], status="failed",
                                       exit_reason=reason)
                except AlpacaError:
                    pass
                continue
            if close_spread(client, strategy, reason, dry_run=dry_run):
                count += 1
    db.log_decision("risk", "flatten_all", f"{reason}: {count} structure(s) closed")
    return count
