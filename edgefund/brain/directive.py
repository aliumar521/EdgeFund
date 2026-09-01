"""Loading and persisting the strategy directive.

The directive is the only channel through which the AI layer influences
trading, and it is policy, not orders. Every field is clamped by the Pydantic
model; risk limits are not represented here at all, so no directive -- however
malformed or however badly the model behaves -- can widen them.

The fallback chain matters more than the happy path. A live competition cannot
stall because a subprocess timed out, so:

    Claude output -> previous stored directive -> static defaults

At worst the agent keeps trading yesterday's policy, or the conservative
built-in one. It never blocks and never trades without a policy.
"""
from __future__ import annotations

import logging

from edgefund.core import db
from edgefund.core.config import SETTINGS
from edgefund.core.models import StrategyDirective

log = logging.getLogger("edgefund.directive")


def static_default() -> StrategyDirective:
    """The policy used before the brain has ever run, and if all else fails."""
    return StrategyDirective(
        regime="normal",
        directional_bias=0.0,
        aggression=1.0,
        allowed_underlyings=list(SETTINGS.universe_daily) + list(SETTINGS.universe_weekly),
        vetoed_underlyings=[],
        preferred_structures=[],
        min_edge_score=SETTINGS.sell_vol_threshold,
        max_dte=SETTINGS.max_dte,
        rationale="static default: no directive from the brain yet",
        source="fallback",
    )


def load_directive() -> StrategyDirective:
    """Most recent valid directive, falling back to the static default."""
    stored = db.latest_directive()
    if not stored:
        return static_default()
    try:
        directive = StrategyDirective(**stored)
    except Exception as exc:
        log.warning("stored directive invalid (%s), using static default", exc)
        return static_default()

    if not directive.allowed_underlyings:
        directive.allowed_underlyings = (
            list(SETTINGS.universe_daily) + list(SETTINGS.universe_weekly))
    return directive


def save_directive(directive: StrategyDirective) -> None:
    db.save_directive(directive.model_dump(mode="json"), directive.source)
    db.log_decision(
        "brain", "directive",
        f"[{directive.source}] regime={directive.regime} "
        f"bias={directive.directional_bias:+.2f} aggr={directive.aggression:.2f} "
        f"min_edge={directive.min_edge_score:.2f} :: {directive.rationale[:200]}",
        payload=directive.model_dump(mode="json"),
    )


def tradable_universe(directive: StrategyDirective) -> list[str]:
    """Symbols the directive permits, in a stable order."""
    allowed = directive.allowed_underlyings or (
        list(SETTINGS.universe_daily) + list(SETTINGS.universe_weekly))
    vetoed = set(directive.vetoed_underlyings)
    return [s for s in allowed if s not in vetoed]
