"""Position sizing and hard risk limits.

Pure functions, no I/O, no API calls -- testable in isolation and the one
module the rest of the agent is not permitted to route around.

The governing idea, borrowed from ai-hedge-fund: **conviction requests, risk
disposes.** The strategy layer only ever proposes a structure; nothing it
returns can enlarge a position beyond what these functions allow, and the AI
layer's directive can shrink risk but never widen it (`aggression` is clamped
to 1.5 upstream and applies to a budget that is itself bounded here).

Sizing is derived from the structure's own max loss rather than picked first
and checked afterwards. A wider or riskier spread automatically produces a
smaller contract count, so dollar risk per position stays roughly constant even
as the structures vary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from edgefund.core.config import RiskLimits
from edgefund.core.models import RiskDecision, SpreadCandidate


@dataclass
class PortfolioState:
    """Everything the risk layer needs to know about the current book."""

    equity: float
    last_equity: float
    options_bp: float
    open_count: int = 0
    deployed_risk: float = 0.0                       # sum of max loss, all open
    risk_by_underlying: dict[str, float] = field(default_factory=dict)

    @property
    def day_pnl_pct(self) -> float:
        if self.last_equity <= 0:
            return 0.0
        return (self.equity - self.last_equity) / self.last_equity


def breaches_daily_halt(state: PortfolioState, limits: RiskLimits) -> bool:
    """True once the day's loss forbids opening anything new."""
    return state.day_pnl_pct <= limits.daily_loss_halt_pct


def breaches_kill_switch(state: PortfolioState, limits: RiskLimits) -> bool:
    """True once the book must be flattened outright."""
    return state.day_pnl_pct <= limits.kill_switch_pct


def risk_budget(candidate: SpreadCandidate, state: PortfolioState,
                limits: RiskLimits, aggression: float, ramp: float) -> float:
    """Dollar risk allowed for a single position of this sleeve."""
    pct = (limits.max_loss_per_position_core_pct
           if candidate.sleeve == "core"
           else limits.max_loss_per_position_satellite_pct)
    return state.equity * pct * max(0.0, aggression) * max(0.0, min(1.0, ramp))


def size_position(candidate: SpreadCandidate, state: PortfolioState,
                  limits: RiskLimits, aggression: float = 1.0,
                  ramp: float = 1.0) -> RiskDecision:
    """Decide contract count, or refuse with a reason.

    Applies, in order: per-position risk budget, remaining buying-power
    allowance, and the per-underlying concentration cap. Every rejection
    carries the specific reason so the decision log explains itself.
    """
    per_contract = candidate.max_loss_per_contract
    if per_contract <= 0:
        return RiskDecision(approved=False, reason="structure reports non-positive max loss")

    if state.open_count >= limits.max_concurrent_strategies:
        return RiskDecision(
            approved=False,
            reason=f"already at max concurrent strategies "
                   f"({state.open_count}/{limits.max_concurrent_strategies})")

    if breaches_daily_halt(state, limits):
        return RiskDecision(
            approved=False,
            reason=f"daily loss halt: {state.day_pnl_pct:.2%} <= "
                   f"{limits.daily_loss_halt_pct:.0%}")

    # 1. per-position risk budget
    budget = risk_budget(candidate, state, limits, aggression, ramp)
    qty = int(budget // per_contract)
    if qty < 1:
        return RiskDecision(
            approved=False,
            reason=f"risk budget ${budget:,.0f} below one contract's max loss "
                   f"${per_contract:,.0f}")

    # 2. remaining buying-power allowance
    bp_allowance = state.options_bp * limits.max_bp_deployed_pct - state.deployed_risk
    if bp_allowance <= 0:
        return RiskDecision(
            approved=False,
            reason=f"buying-power allowance exhausted "
                   f"(deployed ${state.deployed_risk:,.0f} of "
                   f"${state.options_bp * limits.max_bp_deployed_pct:,.0f})")
    qty = min(qty, int(bp_allowance // per_contract))
    if qty < 1:
        return RiskDecision(
            approved=False,
            reason=f"remaining buying power ${bp_allowance:,.0f} below one "
                   f"contract's max loss ${per_contract:,.0f}")

    # 3. per-underlying concentration
    #
    # Anchored to the *target* deployed book (options BP x max deployed), not to
    # currently-deployed risk. Using the live figure collapses the cap to almost
    # nothing on an empty book -- it would have capped the first position of the
    # day at $2,000 against a $3,000 per-position budget, quietly making the
    # configured per-position limit unreachable. The cap should constrain a
    # concentrated book, not block the first trade into an empty one.
    cap = (state.options_bp * limits.max_bp_deployed_pct
           * limits.max_risk_per_underlying_pct)
    existing = state.risk_by_underlying.get(candidate.underlying, 0.0)
    room = cap - existing
    if room <= 0:
        return RiskDecision(
            approved=False,
            reason=f"{candidate.underlying} already at concentration cap "
                   f"(${existing:,.0f} of ${cap:,.0f})")
    qty = min(qty, int(room // per_contract))
    if qty < 1:
        return RiskDecision(
            approved=False,
            reason=f"{candidate.underlying} concentration room ${room:,.0f} "
                   f"below one contract's max loss ${per_contract:,.0f}")

    return RiskDecision(
        approved=True,
        qty=qty,
        reason=(f"{qty} contract(s); risk ${qty * per_contract:,.0f} of budget "
                f"${budget:,.0f}; credit ${qty * candidate.max_profit_per_contract:,.0f}"),
    )


def state_from_account(account: dict[str, Any], open_strategies: list[dict[str, Any]]
                       ) -> PortfolioState:
    """Build the risk view from live account data plus our own strategy records.

    `last_equity` is Alpaca's own prior-close equity, so the daily-loss check
    always measures against the broker's number rather than anything we cached.
    """
    equity = float(account.get("equity") or 0.0)
    last_equity = float(account.get("last_equity") or equity)
    options_bp = float(account.get("options_buying_power") or 0.0)

    risk_by_underlying: dict[str, float] = {}
    deployed = 0.0
    for s in open_strategies:
        risk = float(s.get("max_loss_total") or 0.0)
        deployed += risk
        risk_by_underlying[s["underlying"]] = (
            risk_by_underlying.get(s["underlying"], 0.0) + risk)

    return PortfolioState(
        equity=equity,
        last_equity=last_equity,
        options_bp=options_bp,
        open_count=len(open_strategies),
        deployed_risk=deployed,
        risk_by_underlying=risk_by_underlying,
    )
