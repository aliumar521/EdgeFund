"""The AI strategist: sets policy, never places orders.

Runs three times a day. It receives a compact briefing -- account state, the
current volatility edge across the universe, open positions, recent outcomes and
accumulated lessons -- and returns a StrategyDirective.

The separation is the point. Claude decides *posture*: how aggressive to be,
which underlyings to avoid, which structures suit the regime, how strong an edge
must be before it is worth trading. Deterministic code decides everything else,
and the risk limits are not reachable from here at all. A hallucinated field
cannot do worse than move a clamped dial.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from edgefund.brain.claude_client import ask_for_json, claude_available
from edgefund.brain.directive import (
    load_directive, save_directive, static_default, tradable_universe,
)
from edgefund.core import db
from edgefund.core.config import SETTINGS
from edgefund.core.models import StrategyDirective
from edgefund.data.alpaca import AlpacaClient
from edgefund.edge.score import scan_universe
from edgefund.watchdog.monitor import ET

log = logging.getLogger("edgefund.strategist")

SYSTEM_FRAME = """\
You are the strategist for EdgeFund, an autonomous options trading agent running \
on Alpaca paper trading during a five-day competition that ends Friday 2026-09-04 \
at 11:00 ET. The objective is to maximise P&L over that window.

The agent's thesis is the variance risk premium: implied volatility is usually \
richer than the volatility an underlying subsequently realises. It measures \
vrp_ratio (ATM implied / realised vol) and term_slope (front IV / short-end IV), \
normalises them across the universe, and produces an edge_score. Positive score \
means premium is rich, so it sells defined-risk credit spreads or iron condors. \
Negative means volatility is cheap, so it buys debit spreads for convexity.

You do NOT choose contracts, strikes or position sizes. Deterministic code does \
that, and a hard risk layer you cannot reach caps every position. Your job is \
posture only.

Constraints you must respect:
- Every structure is defined-risk. Alpaca permits no naked short options.
- Only 0-4 DTE contracts are used, so positions realise inside the competition.
- Short options are always flattened by 15:30 ET on their expiry day.
"""

OUTPUT_CONTRACT = """\
Reply with ONLY a JSON object, no prose, no code fence:

{
  "regime": "calm" | "normal" | "stressed" | "event",
  "directional_bias": -1.0 to 1.0,
  "aggression": 0.3 to 1.5,
  "vetoed_underlyings": ["SYM", ...],
  "preferred_structures": ["iron_condor" | "put_credit_spread" |
                           "call_credit_spread" | "call_debit_spread" |
                           "put_debit_spread", ...],
  "min_edge_score": 0.3 to 2.0,
  "max_dte": 0 to 7,
  "rationale": "two or three sentences on why this posture, citing the numbers"
}

Guidance:
- aggression above 1.0 only when the edge is broad and the regime is calm.
- min_edge_score is the bar a symbol must clear to trade. Raise it to be
  selective when signals are weak, lower it to trade more when they are strong.
- veto a symbol when it carries event risk the vol model cannot see, such as
  earnings, or when recent outcomes on it have been consistently poor.
- empty lists are fine and mean "no constraint".
"""


def _fmt_edges(snapshots: list[Any]) -> str:
    if not snapshots:
        return "  (no edge snapshots available)"
    rows = [f"  {'sym':<6}{'spot':>9}{'rv':>7}{'iv':>7}{'vrp':>7}"
            f"{'term':>7}{'trend':>7}{'edge':>7}  regime"]
    for s in sorted(snapshots, key=lambda x: -x.edge_score):
        rows.append(
            f"  {s.underlying:<6}{s.spot:>9.2f}{s.rv:>7.3f}{s.atm_iv:>7.3f}"
            f"{s.vrp_ratio:>7.2f}{s.term_slope:>7.2f}{s.trend:>7.2f}"
            f"{s.edge_score:>7.2f}  {s.regime}")
    return "\n".join(rows)


def _fmt_open(strategies: list[dict[str, Any]]) -> str:
    if not strategies:
        return "  (flat -- no open positions)"
    rows = []
    for s in strategies:
        rows.append(
            f"  {s['underlying']:<6} {s['structure']:<20} qty={s['qty']:<3} "
            f"exp={s['expiry']} status={s['status']:<8} "
            f"risk=${s['max_loss_total']:,.0f}")
    return "\n".join(rows)


def _fmt_closed(closed: list[dict[str, Any]]) -> str:
    if not closed:
        return "  (nothing closed yet)"
    rows = []
    for s in closed[:15]:
        pnl = s.get("realized_pnl")
        pnl_s = f"${pnl:+,.0f}" if pnl is not None else "n/a"
        rows.append(
            f"  {s['underlying']:<6} {s['structure']:<20} {pnl_s:>10}  "
            f"{(s.get('exit_reason') or '')[:60]}")
    return "\n".join(rows)


def _performance_stats(closed: list[dict[str, Any]]) -> str:
    if not closed:
        return "  (no closed trades yet)"
    by_structure: dict[str, list[float]] = {}
    for s in closed:
        pnl = s.get("realized_pnl")
        if pnl is None:
            continue
        by_structure.setdefault(s["structure"], []).append(float(pnl))

    rows = []
    for structure, pnls in sorted(by_structure.items()):
        wins = sum(1 for p in pnls if p > 0)
        rows.append(f"  {structure:<22} n={len(pnls):<3} win={wins}/{len(pnls)}  "
                    f"total=${sum(pnls):+,.0f}  avg=${sum(pnls)/len(pnls):+,.0f}")
    return "\n".join(rows) if rows else "  (no realised P&L yet)"


def _recent_lessons(limit: int = 8) -> str:
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT lessons FROM reflections ORDER BY id DESC LIMIT 5").fetchall()
    lessons: list[str] = []
    for r in rows:
        try:
            lessons.extend(json.loads(r["lessons"] or "[]"))
        except json.JSONDecodeError:
            continue
    if not lessons:
        return "  (no lessons recorded yet)"
    return "\n".join(f"  - {l}" for l in lessons[:limit])


def build_briefing(client: AlpacaClient, slot: str) -> tuple[str, list[Any]]:
    """Assemble the context for one strategist call."""
    directive = load_directive()
    account = client.account()
    open_strategies = db.active_strategies()
    closed = db.closed_strategies(limit=40)
    snapshots = scan_universe(client, tradable_universe(directive), directive.max_dte)

    equity = float(account.get("equity") or 0)
    last_equity = float(account.get("last_equity") or equity)
    day_pnl = (equity - last_equity) / last_equity if last_equity else 0.0
    deployed = sum(float(s.get("max_loss_total") or 0) for s in open_strategies)
    now = datetime.now(ET)

    briefing = f"""{SYSTEM_FRAME}

=== SLOT: {slot} | {now:%A %Y-%m-%d %H:%M} ET ===

ACCOUNT
  equity              ${equity:,.2f}
  day P&L             {day_pnl:+.2%}
  total return        {(equity / 100_000 - 1):+.2%} from a $100,000 start
  options buying power ${float(account.get('options_buying_power') or 0):,.0f}
  deployed risk       ${deployed:,.0f} across {len(open_strategies)} structure(s)

HARD LIMITS (fixed; you cannot change these)
  max loss per position   3.0% of equity (core) / 1.5% (satellite)
  max concurrent          {SETTINGS.limits.max_concurrent_strategies}
  daily loss halt         {SETTINGS.limits.daily_loss_halt_pct:.0%}
  kill switch             {SETTINGS.limits.kill_switch_pct:.0%}

CURRENT EDGE ACROSS UNIVERSE
  vrp_ratio >1 means implied is richer than realised.
  term_slope >1 means the front is backwardated versus the short end.
  edge_score is cross-sectionally normalised; positive = sell premium.
{_fmt_edges(snapshots)}

OPEN POSITIONS
{_fmt_open(open_strategies)}

RECENTLY CLOSED
{_fmt_closed(closed)}

PERFORMANCE BY STRUCTURE
{_performance_stats(closed)}

LESSONS FROM PRIOR REFLECTIONS
{_recent_lessons()}

CURRENT DIRECTIVE ({directive.source})
  regime={directive.regime} bias={directive.directional_bias:+.2f} \
aggression={directive.aggression:.2f} min_edge={directive.min_edge_score:.2f}
  rationale: {directive.rationale}

{OUTPUT_CONTRACT}
"""
    return briefing, snapshots


def run_strategist(client: AlpacaClient, slot: str) -> StrategyDirective:
    """Produce and persist a directive for this slot.

    Falls back to the previous directive on any failure, so a brain outage
    degrades posture-setting rather than stopping the fund.
    """
    if not claude_available():
        log.warning("claude unavailable; keeping previous directive")
        current = load_directive()
        db.log_decision("brain", "unavailable",
                        "claude binary not found; continuing on previous directive")
        return current

    briefing, _ = build_briefing(client, slot)
    payload = ask_for_json(briefing)

    if not payload:
        previous = load_directive()
        previous.source = "previous"
        previous.rationale = (f"[{slot}] brain call failed; "
                              f"continuing on prior directive: {previous.rationale}")
        save_directive(previous)
        log.warning("strategist %s: no usable reply, kept previous directive", slot)
        return previous

    # Whitelist the fields the model is allowed to set. Anything else it
    # invents is ignored rather than trusted, and Pydantic clamps the rest.
    allowed = {
        "regime", "directional_bias", "aggression", "vetoed_underlyings",
        "preferred_structures", "min_edge_score", "max_dte", "rationale",
    }
    filtered = {k: v for k, v in payload.items() if k in allowed}

    try:
        directive = StrategyDirective(
            **filtered,
            allowed_underlyings=list(SETTINGS.universe_daily) + list(SETTINGS.universe_weekly),
            source=f"claude:{slot}",
        )
    except Exception as exc:
        log.warning("strategist %s: directive failed validation (%s)", slot, exc)
        fallback = load_directive()
        fallback.source = "previous"
        save_directive(fallback)
        return fallback

    save_directive(directive)
    log.info("strategist %s: regime=%s aggression=%.2f min_edge=%.2f",
             slot, directive.regime, directive.aggression, directive.min_edge_score)
    return directive
