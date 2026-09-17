"""Tests for the self-evolution loop and the dials that gate trading.

These cover the failure mode that took the fund offline between 2026-09-04 and
2026-09-16: not a bad trade, but a posture value that made trading structurally
impossible, and a reflection loop that kept "learning" from a sample which had
stopped changing. Both are silent -- nothing errors, the agent simply stops --
so they are worth pinning down in tests.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from edgefund.brain import reflect
from edgefund.core.models import StrategyDirective
from edgefund.watchdog.monitor import ET, unrealised_pnl


# --------------------------------------------------------------------------
# max_dte must stay tradable
# --------------------------------------------------------------------------

@pytest.mark.parametrize("proposed, expected", [
    (-5, 1),      # nonsense input
    (0, 1),       # the value that halted the fund
    (1, 1),
    (4, 4),
    (7, 7),
    (99, 7),
])
def test_max_dte_is_clamped_into_a_tradable_range(proposed, expected):
    assert StrategyDirective(max_dte=proposed).max_dte == expected


def test_max_dte_zero_cannot_survive_a_round_trip_through_storage():
    """load_directive() revalidates stored rows, so a persisted 0 self-heals.

    This is what stops a stuck directive from outliving the bug that wrote it:
    the brain can be down for days and the book still reloads as tradable.
    """
    stored = StrategyDirective(max_dte=0, rationale="stuck").model_dump(mode="json")
    stored["max_dte"] = 0                      # simulate a pre-fix row on disk
    assert StrategyDirective(**stored).max_dte == 1


# --------------------------------------------------------------------------
# TUNABLE must not contain knobs nothing reads
# --------------------------------------------------------------------------

def test_every_tunable_is_actually_read_somewhere():
    """A knob that is tuned but never read lets the model believe it acted.

    `target_short_delta` sat in TUNABLE for the whole competition and absorbed 8
    of ~20 adjustments while no live code path consulted it.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "edgefund"
    sources = "\n".join(p.read_text(encoding="utf-8")
                        for p in root.rglob("*.py")
                        if p.name not in {"reflect.py", "params.py"})
    for key in reflect.TUNABLE:
        assert f'"{key}"' in sources or f"'{key}'" in sources, (
            f"{key} is tunable but never read by any live code path")


# --------------------------------------------------------------------------
# reflection refuses to re-learn from an unchanged window
# --------------------------------------------------------------------------

def _closed(ts_close: datetime) -> dict:
    return {"realized_pnl": 10.0, "ts_close": ts_close.isoformat(),
            "underlying": "SPY", "structure": "put_credit_spread",
            "qty": 1, "dry_run": 0, "entry_features": {}}


def test_closed_today_matches_only_todays_et_date():
    now = datetime.now(ET)
    rows = [_closed(now), _closed(now - timedelta(days=3))]
    assert len(reflect._closed_today(rows)) == 1


def test_reflection_skips_when_nothing_closed_today(monkeypatch):
    """The exact shape of the 09-04 -> 09-16 waste: a stale rolling window."""
    stale = [_closed(datetime.now(ET) - timedelta(days=5)) for _ in range(60)]
    monkeypatch.setattr(reflect.db, "closed_strategies", lambda limit=60: stale)
    monkeypatch.setattr(reflect.db, "active_strategies", lambda: [])

    logged: list[tuple] = []
    monkeypatch.setattr(reflect.db, "log_decision",
                        lambda *a, **k: logged.append(a))
    monkeypatch.setattr(reflect, "claude_available",
                        lambda: pytest.fail("must not spend an AI call"))

    assert reflect.run_reflection(client=None) is None
    assert logged and logged[0][1] == "reflection_skipped"


# --------------------------------------------------------------------------
# parameter changes are auditable
# --------------------------------------------------------------------------

def test_apply_adjustments_records_the_previous_value(monkeypatch):
    writes: list[tuple] = []
    monkeypatch.setattr(reflect.db, "set_param",
                        lambda k, v, source: writes.append((k, v)))
    monkeypatch.setattr("edgefund.core.params.get", lambda key, default=None: 0.35)

    applied = reflect.apply_adjustments({"delta_stop": 0.45})
    assert applied == {"delta_stop": {"from": 0.35, "to": 0.45}}
    assert writes == [("delta_stop", 0.45)]


def test_apply_adjustments_drops_a_no_op_rewrite(monkeypatch):
    """Re-proposing the current value should not pollute the audit trail."""
    writes: list[tuple] = []
    monkeypatch.setattr(reflect.db, "set_param",
                        lambda k, v, source: writes.append((k, v)))
    monkeypatch.setattr("edgefund.core.params.get", lambda key, default=None: 0.35)

    assert reflect.apply_adjustments({"delta_stop": 0.35}) == {}
    assert writes == []


def test_apply_adjustments_clamps_and_ignores_protected_keys(monkeypatch):
    monkeypatch.setattr(reflect.db, "set_param", lambda k, v, source: None)
    monkeypatch.setattr("edgefund.core.params.get", lambda key, default=None: None)

    applied = reflect.apply_adjustments({
        "delta_stop": 9.0,                       # above the 0.50 ceiling
        "max_loss_per_position_core_pct": 0.5,   # a risk limit -- unreachable
    })
    assert applied["delta_stop"]["to"] == 0.50
    assert "max_loss_per_position_core_pct" not in applied


# --------------------------------------------------------------------------
# marked P&L agrees with the realised formula
# --------------------------------------------------------------------------

def test_credit_structure_marks_profit_as_cost_to_close_falls():
    s = {"net_credit": 1.00, "entry_fill_price": 1.00, "qty": 3}
    assert unrealised_pnl(s, close_cost=0.40) == pytest.approx(180.0)
    assert unrealised_pnl(s, close_cost=1.60) == pytest.approx(-180.0)


def test_debit_structure_marks_profit_as_value_rises():
    # net_credit negative => debit paid of 1.00 per contract.
    s = {"net_credit": -1.00, "entry_fill_price": 1.00, "qty": 2}
    assert unrealised_pnl(s, close_cost=-1.75) == pytest.approx(150.0)
    assert unrealised_pnl(s, close_cost=-0.25) == pytest.approx(-150.0)


def test_unmarkable_structure_contributes_nothing():
    assert unrealised_pnl({"net_credit": 0.0, "qty": 5}, close_cost=0.2) == 0.0
