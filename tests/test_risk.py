"""Risk gate and exit rule tests.

These cover the two places where a bug costs real money: sizing a position too
large, and failing to close one that should be closed. Both are pure functions,
so they are tested directly with no network involved.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest
from zoneinfo import ZoneInfo

from edgefund.core.config import RiskLimits
from edgefund.core.models import Leg, SpreadCandidate
from edgefund.risk.limits import (
    PortfolioState, breaches_daily_halt, breaches_kill_switch,
    size_position, state_from_account,
)
from edgefund.watchdog.monitor import evaluate_exit

ET = ZoneInfo("America/New_York")
LIMITS = RiskLimits()


def make_candidate(underlying: str = "SPY", sleeve: str = "core",
                   max_loss: float = 400.0, credit: float = 1.0) -> SpreadCandidate:
    expiry = date(2026, 9, 4)
    return SpreadCandidate(
        underlying=underlying, structure="put_credit_spread", sleeve=sleeve,
        legs=[
            Leg(symbol="SPY260904P00760000", side="sell", position_intent="sell_to_open",
                strike=760, expiry=expiry, opt_type="put", bid=1.0, ask=1.1, delta=-0.2),
            Leg(symbol="SPY260904P00755000", side="buy", position_intent="buy_to_open",
                strike=755, expiry=expiry, opt_type="put", bid=0.4, ask=0.5, delta=-0.1),
        ],
        net_credit=credit, width=5.0,
        max_loss_per_contract=max_loss, max_profit_per_contract=credit * 100,
        short_delta=0.2, dte=2, expiry=expiry, edge_score=1.5,
    )


def base_state(**kw) -> PortfolioState:
    defaults = dict(equity=100_000.0, last_equity=100_000.0, options_bp=100_000.0)
    defaults.update(kw)
    return PortfolioState(**defaults)


# ----------------------------------------------------------------- sizing

def test_sizes_from_risk_budget_not_arbitrary_count():
    """3% of 100k = $3,000 budget; at $400 max loss that is 7 contracts."""
    decision = size_position(make_candidate(max_loss=400), base_state(), LIMITS)
    assert decision.approved
    assert decision.qty == 7


def test_wider_risk_produces_smaller_size():
    """Dollar risk stays roughly constant as the structure's risk changes."""
    small = size_position(make_candidate(max_loss=200), base_state(), LIMITS)
    large = size_position(make_candidate(max_loss=1500), base_state(), LIMITS)
    assert small.qty > large.qty
    assert small.qty * 200 <= 3000
    assert large.qty * 1500 <= 3000


def test_satellite_sleeve_gets_half_the_budget():
    core = size_position(make_candidate(sleeve="core", max_loss=100), base_state(), LIMITS)
    sat = size_position(make_candidate(sleeve="satellite", max_loss=100), base_state(), LIMITS)
    assert sat.qty < core.qty


def test_ramp_scales_size_down():
    full = size_position(make_candidate(), base_state(), LIMITS, ramp=1.0)
    ramped = size_position(make_candidate(), base_state(), LIMITS, ramp=0.15)
    assert ramped.qty < full.qty


def test_aggression_cannot_exceed_budget_without_bound():
    """Even at max clamped aggression, one position stays a bounded fraction."""
    decision = size_position(make_candidate(max_loss=400), base_state(),
                             LIMITS, aggression=1.5)
    assert decision.qty * 400 <= 100_000 * LIMITS.max_loss_per_position_core_pct * 1.5


# ----------------------------------------------------------------- rejections

def test_rejects_when_position_too_large_for_budget():
    decision = size_position(make_candidate(max_loss=50_000), base_state(), LIMITS)
    assert not decision.approved
    assert "below one contract" in decision.reason


def test_rejects_at_max_concurrent():
    state = base_state(open_count=LIMITS.max_concurrent_strategies)
    decision = size_position(make_candidate(), state, LIMITS)
    assert not decision.approved
    assert "max concurrent" in decision.reason


def test_rejects_when_daily_halt_breached():
    state = base_state(equity=89_000.0, last_equity=100_000.0)
    decision = size_position(make_candidate(), state, LIMITS)
    assert not decision.approved
    assert "daily loss halt" in decision.reason


def test_rejects_when_buying_power_allowance_exhausted():
    state = base_state(deployed_risk=80_000.0)
    decision = size_position(make_candidate(), state, LIMITS)
    assert not decision.approved
    assert "buying-power" in decision.reason.lower()


def test_concentration_cap_limits_a_single_underlying():
    """A symbol already carrying most of the book's risk gets no more."""
    state = base_state(
        open_count=4, deployed_risk=20_000.0,
        risk_by_underlying={"SPY": 19_000.0},
    )
    decision = size_position(make_candidate("SPY"), state, LIMITS)
    other = size_position(make_candidate("QQQ"), state, LIMITS)
    assert not decision.approved
    assert "concentration" in decision.reason
    assert other.approved


def test_rejects_non_positive_max_loss():
    decision = size_position(make_candidate(max_loss=0), base_state(), LIMITS)
    assert not decision.approved


# ----------------------------------------------------------------- breakers

@pytest.mark.parametrize("equity,halted,killed", [
    (100_000, False, False),
    (95_000, False, False),
    (89_500, True, False),
    (81_000, True, True),
])
def test_circuit_breakers(equity, halted, killed):
    state = base_state(equity=float(equity), last_equity=100_000.0)
    assert breaches_daily_halt(state, LIMITS) is halted
    assert breaches_kill_switch(state, LIMITS) is killed


def test_state_from_account_aggregates_risk():
    account = {"equity": "95000", "last_equity": "100000", "options_buying_power": "60000"}
    state = state_from_account(account, [
        {"underlying": "SPY", "max_loss_total": 1000.0},
        {"underlying": "SPY", "max_loss_total": 500.0},
        {"underlying": "QQQ", "max_loss_total": 800.0},
    ])
    assert state.deployed_risk == 2300.0
    assert state.risk_by_underlying["SPY"] == 1500.0
    assert state.open_count == 3
    assert state.day_pnl_pct == pytest.approx(-0.05)


# ----------------------------------------------------------------- exits

def credit_strategy(**kw) -> dict:
    base = {
        "strategy_uid": "ef-test-1", "expiry": "2026-09-04",
        "net_credit": 1.00, "entry_fill_price": 1.00, "qty": 1,
    }
    base.update(kw)
    return base


MIDDAY = datetime(2026, 9, 2, 12, 0, tzinfo=ET)


def test_profit_target_fires_at_55_percent_captured():
    """Credit of 1.00 bought back at 0.45 means 55% captured."""
    reason = evaluate_exit(credit_strategy(), close_cost=0.45,
                           short_deltas=[0.1], now=MIDDAY)
    assert reason and "profit target" in reason


def test_holds_when_only_partly_profitable():
    assert evaluate_exit(credit_strategy(), 0.70, [0.15], MIDDAY) is None


def test_stop_loss_fires_at_2x_credit():
    reason = evaluate_exit(credit_strategy(), 2.10, [0.30], MIDDAY)
    assert reason and "stop loss" in reason


def test_delta_stop_fires_when_short_strike_challenged():
    reason = evaluate_exit(credit_strategy(), 1.20, [0.42], MIDDAY)
    assert reason and "delta stop" in reason


def test_expiry_flatten_fires_after_cutoff_regardless_of_pnl():
    """The unconditional rule: expiring today and past 15:30 ET means close."""
    late = datetime(2026, 9, 4, 15, 45, tzinfo=ET)
    reason = evaluate_exit(credit_strategy(expiry="2026-09-04"),
                           close_cost=0.90, short_deltas=[0.05], now=late)
    assert reason and "expiry flatten" in reason


def test_expiry_flatten_does_not_fire_before_cutoff():
    early = datetime(2026, 9, 4, 14, 0, tzinfo=ET)
    assert evaluate_exit(credit_strategy(expiry="2026-09-04"), 0.90, [0.05], early) is None


def test_debit_structure_profit_and_stop():
    """Debit structures store a negative net_credit and invert both rules."""
    debit = credit_strategy(net_credit=-1.50, entry_fill_price=1.50)
    # close_cost is negative for a debit structure: it is worth that much to us
    assert "profit target" in (evaluate_exit(debit, -3.10, [], MIDDAY) or "")
    assert "stop loss" in (evaluate_exit(debit, -0.60, [], MIDDAY) or "")
    assert evaluate_exit(debit, -1.80, [], MIDDAY) is None
