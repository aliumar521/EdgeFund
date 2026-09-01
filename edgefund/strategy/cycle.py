"""One full trading cycle: measure -> propose -> gate -> size -> submit.

Single-shot and stateless by design. Every run re-derives account state from
Alpaca rather than trusting anything cached in memory, so a crashed or
restarted process resumes correctly with no recovery logic -- the broker plus
our own strategy table are always the source of truth.

The stage order is deliberate and never varies:

    edge scan -> structure construction -> risk gate -> sizing -> submission

Skips are logged with their specific reason at whichever stage they occur.
Knowing *why* the agent declined a trade is as much a part of the audit trail
as knowing why it took one.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

from edgefund.brain.directive import load_directive, tradable_universe
from edgefund.core import db, params
from edgefund.core.config import SETTINGS
from edgefund.core.models import EdgeSnapshot, SpreadCandidate, StrategyDirective
from edgefund.data.alpaca import AlpacaClient
from edgefund.edge.score import scan_universe
from edgefund.execute.router import submit_spread, sync_strategy_status
from edgefund.risk.limits import (
    PortfolioState, breaches_daily_halt, breaches_kill_switch,
    size_position, state_from_account,
)
from edgefund.strategy.spreads import ChainView, build_for_snapshot

log = logging.getLogger("edgefund.cycle")


def run_cycle(client: AlpacaClient, dry_run: bool | None = None,
              ramp: float | None = None) -> dict[str, Any]:
    """Scan, decide, and place orders. Returns a summary for logging/tests."""
    dry_run = SETTINGS.dry_run if dry_run is None else dry_run
    ramp = SETTINGS.size_ramp if ramp is None else ramp

    directive = load_directive()
    account = client.account()

    # Reconcile before deciding: an order that filled since the last run changes
    # what the risk layer is allowed to do now.
    for strategy in db.active_strategies():
        try:
            sync_strategy_status(client, strategy)
        except Exception as exc:
            log.warning("%s: sync failed: %s", strategy["strategy_uid"], exc)

    open_strategies = db.active_strategies()
    state = state_from_account(account, open_strategies)

    db.record_equity(
        equity=state.equity, cash=float(account.get("cash") or 0),
        options_bp=state.options_bp, day_pnl_pct=state.day_pnl_pct,
    )

    summary: dict[str, Any] = {
        "equity": state.equity,
        "day_pnl_pct": state.day_pnl_pct,
        "open_count": state.open_count,
        "deployed_risk": state.deployed_risk,
        "directive": directive.source,
        "dry_run": dry_run,
        "ramp": ramp,
        "scanned": 0, "proposed": 0, "opened": 0, "skips": [],
    }

    if breaches_kill_switch(state, SETTINGS.limits):
        db.log_decision("risk", "kill_switch",
                        f"day P&L {state.day_pnl_pct:.2%} breached kill switch "
                        f"{SETTINGS.limits.kill_switch_pct:.0%}")
        summary["halted"] = "kill_switch"
        return summary

    if breaches_daily_halt(state, SETTINGS.limits):
        db.log_decision("risk", "daily_halt",
                        f"day P&L {state.day_pnl_pct:.2%} breached halt "
                        f"{SETTINGS.limits.daily_loss_halt_pct:.0%}; no new entries")
        summary["halted"] = "daily_loss"
        return summary

    universe = tradable_universe(directive)
    snapshots = scan_universe(client, universe, directive.max_dte)
    summary["scanned"] = len(snapshots)

    for snap in snapshots:
        db.record_edge(snap.model_dump(mode="json"))

    # Strongest measured edge first, so the best opportunities get the buying
    # power before the concentration and BP caps start biting.
    ranked = sorted(snapshots, key=lambda s: -abs(s.edge_score))

    for snap in ranked:
        result = _consider(client, snap, directive, state, dry_run, ramp)
        if result["action"] == "opened":
            summary["opened"] += 1
            summary["proposed"] += 1
            # Reflect the new exposure immediately so later candidates in the
            # same cycle are sized against an accurate book.
            state.open_count += 1
            state.deployed_risk += result["risk"]
            state.risk_by_underlying[snap.underlying] = (
                state.risk_by_underlying.get(snap.underlying, 0.0) + result["risk"])
        elif result["action"] == "proposed":
            summary["proposed"] += 1
            summary["skips"].append(f"{snap.underlying}: {result['reason']}")
        else:
            summary["skips"].append(f"{snap.underlying}: {result['reason']}")

    db.heartbeat("cycle", f"scanned={summary['scanned']} opened={summary['opened']}")
    return summary


def _consider(client: AlpacaClient, snap: EdgeSnapshot, directive: StrategyDirective,
              state: PortfolioState, dry_run: bool, ramp: float) -> dict[str, Any]:
    """Evaluate one underlying end to end."""
    sym = snap.underlying

    # The directive proposes a bar; a value learned by reflection overrides it,
    # so the agent's own results outrank a single model call.
    min_edge = params.get("min_edge_score", directive.min_edge_score)
    if abs(snap.edge_score) < min_edge:
        reason = (snap.detail.get("gated")
                  or f"edge {snap.edge_score:+.2f} inside threshold "
                     f"+/-{min_edge:.2f}")
        db.log_decision("scan", "skip", reason, sym, snap.detail)
        return {"action": "skip", "reason": reason}

    expiry_str = snap.detail.get("expiry")
    dte = int(snap.detail.get("dte") or 0)
    if not expiry_str:
        return {"action": "skip", "reason": "no expiry recorded"}

    # Pull a wider strike band than the edge calc used, since the short strike
    # sits well out of the money.
    band = max(snap.spot * 0.14, 8.0)
    try:
        chain_snaps = client.option_chain(
            sym, expiration_date=expiry_str,
            strike_gte=snap.spot - band, strike_lte=snap.spot + band,
        )
    except Exception as exc:
        reason = f"chain fetch failed: {exc}"
        db.log_decision("scan", "skip", reason, sym)
        return {"action": "skip", "reason": reason}

    chain = ChainView(chain_snaps, date.fromisoformat(expiry_str))
    candidate = build_for_snapshot(chain, snap, directive, dte)
    if candidate is None:
        reason = ("no structure met the construction filters "
                  "(delta target, wing width, credit/width or liquidity)")
        db.log_decision("scan", "skip", reason, sym, {"edge": snap.edge_score})
        return {"action": "skip", "reason": reason}

    decision = size_position(candidate, state, SETTINGS.limits,
                             aggression=directive.aggression, ramp=ramp)
    if not decision.approved:
        db.log_decision("risk", "skip", decision.reason, sym,
                        {"structure": candidate.structure,
                         "max_loss_per_contract": candidate.max_loss_per_contract})
        return {"action": "proposed", "reason": decision.reason}

    uid = submit_spread(client, candidate, decision.qty,
                        entry_features=candidate.features, dry_run=dry_run)
    if not uid:
        return {"action": "proposed", "reason": "submission rejected"}

    return {
        "action": "opened",
        "reason": decision.reason,
        "uid": uid,
        "risk": candidate.max_loss_per_contract * decision.qty,
    }
