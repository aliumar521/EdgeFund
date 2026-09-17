"""End-of-day reflection: the loop that lets the agent improve itself.

Pattern borrowed from TradingAgents' reflection mechanism, deliberately kept
cheap and auditable: outcomes are resolved against what was actually believed at
entry, a single model call turns that into short concrete lessons, and those
lessons are injected into tomorrow's strategist prompt.

Two things make this safe to run unattended:

* Every trade stored its *entry features* -- the vrp_ratio, term slope, trend,
  expected value and probability of profit that justified it. Reflection
  compares belief against outcome rather than just tallying P&L, so a lesson can
  say which signal misled it, not merely that it lost.
* Parameter changes are bounded. The model proposes adjustments to a small set
  of tuning knobs and each is clamped to a range it cannot argue its way out of.
  Risk limits are not in that set and are unreachable from here.
"""
from __future__ import annotations

import json
import logging
import statistics
from datetime import datetime, timezone
from typing import Any

from edgefund.brain.claude_client import ask_for_json, claude_available
from edgefund.core import db
from edgefund.data.alpaca import AlpacaClient
from edgefund.watchdog.monitor import ET

log = logging.getLogger("edgefund.reflect")

# The only parameters the reflection loop may move, with hard bounds. Every key
# here MUST be read back through params.get() somewhere in the live path --
# a knob that is tuned but never read is worse than no knob, because the model
# records that it acted on a problem and nothing changes.
#
# `target_short_delta` used to sit here and was exactly that: the live structure
# search in strategy/spreads.py ranks every strike combination by expected value
# and never consults it (only scripts/debug_chain.py does). It absorbed 8 of the
# ~20 adjustments made between 2026-09-03 and 09-16, all of them no-ops. Removed
# rather than wired up -- delta-targeting is the heuristic the EV search replaced.
TUNABLE: dict[str, tuple[float, float]] = {
    "min_edge_score": (0.30, 2.00),
    "profit_target_pct": (0.30, 0.80),
    "stop_loss_mult": (1.30, 3.00),
    "delta_stop": (0.25, 0.50),
}

REFLECTION_PROMPT = """\
You are reviewing EdgeFund's trading now that outcomes are known.

IMPORTANT -- what you are looking at: the trades below are a **rolling window of \
the last {n_window} closed trades**, not just today's. {n_today} of them closed \
today. Older trades reappear here every evening, so do not read a repeated \
pattern as fresh evidence, and do not re-adjust a parameter you have already \
adjusted for the same trades. If the window looks unchanged since your last \
review, say so and propose no changes.

EdgeFund sells defined-risk option spreads when implied volatility is rich \
relative to realised volatility (vrp_ratio > 1), and buys debit spreads when it \
is cheap. Each trade below records what the agent believed at entry -- its \
expected value, probability of profit, and the volatility signals that justified \
it -- alongside what actually happened.

{stats_block}

CLOSED TRADES
{trades_block}

STILL OPEN
{open_block}

Write a reflection that a future version of this agent can act on. Be specific \
and cite numbers. Do not restate the P&L; explain what the signals got right or \
wrong and what should change.

If the honest conclusion is that the fix lies outside the tunable parameters -- \
in how expected value, probability of profit or the exit rules are *computed* -- \
say that plainly in the lessons rather than moving a knob that cannot address it. \
A lesson naming an unreachable defect is more useful than a parameter nudge that \
pretends to fix it.

Reply with ONLY a JSON object:

{{
  "reflection": "3-5 sentences of plain prose on what the day revealed",
  "lessons": ["concrete, actionable lesson", "..."],
  "parameter_adjustments": {{"min_edge_score": 0.9}},
  "confidence": "low" | "medium" | "high"
}}

parameter_adjustments may include any of: {tunables}. \
Omit it entirely, or use an empty object, if the sample is too small to justify \
a change -- with only a handful of trades, restraint is usually correct. \
Never propose a change to risk limits; they are not adjustable.
"""


def _trade_rows(closed: list[dict[str, Any]]) -> str:
    if not closed:
        return "  (no trades closed today)"
    rows = []
    for s in closed:
        f = s.get("entry_features") or {}
        pnl = s.get("realized_pnl")
        pnl_s = f"${pnl:+,.0f}" if pnl is not None else "n/a"
        rows.append(
            f"  {s['underlying']:<6} {s['structure']:<20} qty={s['qty']:<3} "
            f"{pnl_s:>10}\n"
            f"      believed: vrp={f.get('vrp_ratio', 0):.2f} "
            f"term={f.get('term_slope', 0):.2f} trend={f.get('trend', 0):+.2f} "
            f"IV={f.get('atm_iv', 0):.1%} RV={f.get('rv', 0):.1%} "
            f"EV=${f.get('ev', 0):+,.0f} POP={f.get('pop', 0):.0%}\n"
            f"      outcome:  {(s.get('exit_reason') or 'unknown')[:80]}")
    return "\n".join(rows)


def _stats_block(closed: list[dict[str, Any]]) -> str:
    realised = [float(s["realized_pnl"]) for s in closed
                if s.get("realized_pnl") is not None]
    if not realised:
        return "SUMMARY\n  no realised P&L yet"

    wins = [p for p in realised if p > 0]
    losses = [p for p in realised if p <= 0]

    lines = [
        "SUMMARY",
        f"  trades closed     {len(realised)}",
        f"  win rate          {len(wins)}/{len(realised)} "
        f"({len(wins) / len(realised):.0%})",
        f"  total realised    ${sum(realised):+,.2f}",
        f"  average trade     ${statistics.fmean(realised):+,.2f}",
    ]
    if wins:
        lines.append(f"  average win       ${statistics.fmean(wins):+,.2f}")
    if losses:
        lines.append(f"  average loss      ${statistics.fmean(losses):+,.2f}")

    # Belief-versus-outcome is the part worth learning from: if realised win
    # rate sits well below the modelled POP, the vol forecast is too optimistic.
    pops = [float((s.get("entry_features") or {}).get("pop") or 0)
            for s in closed if s.get("realized_pnl") is not None]
    pops = [p for p in pops if p > 0]
    if pops:
        lines.append(f"  modelled POP avg  {statistics.fmean(pops):.0%} "
                     f"(vs realised {len(wins) / len(realised):.0%})")

    by_structure: dict[str, list[float]] = {}
    for s in closed:
        if s.get("realized_pnl") is not None:
            by_structure.setdefault(s["structure"], []).append(float(s["realized_pnl"]))
    if by_structure:
        lines.append("  by structure:")
        for structure, pnls in sorted(by_structure.items()):
            w = sum(1 for p in pnls if p > 0)
            lines.append(f"    {structure:<22} n={len(pnls):<3} win={w}/{len(pnls)} "
                         f"total=${sum(pnls):+,.0f}")
    return "\n".join(lines)


def apply_adjustments(proposed: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Clamp and persist proposed parameter changes.

    Returns `{key: {"from": old, "to": new}}`. Recording the previous value is
    the point: `strategy_params` only ever holds the current one, so without it
    a sequence like delta_stop 0.35 -> 0.40 -> 0.25 is invisible after the fact
    and there is no way to see a knob oscillating on unchanged evidence.

    A proposal that does not move the value is dropped rather than written, so
    the audit trail carries real changes only.
    """
    from edgefund.core import params

    applied: dict[str, dict[str, float]] = {}
    for key, raw in (proposed or {}).items():
        if key not in TUNABLE:
            log.info("ignoring proposed change to unknown/protected param %r", key)
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        lo, hi = TUNABLE[key]
        clamped = max(lo, min(hi, value))
        if clamped != value:
            log.info("clamped %s from %s to %s", key, value, clamped)

        previous = params.get(key)
        try:
            previous = float(previous) if previous is not None else None
        except (TypeError, ValueError):
            previous = None
        if previous is not None and abs(previous - clamped) < 1e-9:
            log.info("%s already at %s; no change written", key, clamped)
            continue

        db.set_param(key, clamped, source="reflection")
        applied[key] = {"from": previous, "to": clamped}
        log.info("param %s: %s -> %s", key, previous, clamped)

    if applied:
        params.invalidate()          # make the new values live before the next pass
    return applied


def _closed_today(closed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Subset of `closed` that actually closed during today's ET session."""
    today = datetime.now(ET).date().isoformat()
    return [s for s in closed if str(s.get("ts_close") or "")[:10] == today]


def run_reflection(client: AlpacaClient) -> dict[str, Any] | None:
    """Resolve the day's outcomes into lessons and bounded parameter nudges."""
    closed = db.closed_strategies(limit=60)
    closed = [s for s in closed if not s.get("dry_run")]
    open_now = db.active_strategies()

    if not closed:
        log.info("reflection skipped: no closed trades to learn from")
        db.log_decision("brain", "reflection_skipped",
                        "no closed live trades yet")
        return None

    # `closed_strategies` is a rolling last-60 window with no date filter, so it
    # keeps returning the same trades on days when nothing closed. Reflecting on
    # it anyway is not free: between 2026-09-04 and 09-16 this ran ten times on a
    # byte-identical sample ({"trades": 60, "total_pnl": -218.0, "win_rate": 0.3})
    # and moved parameters on nine of them -- including delta_stop 0.35 -> 0.40
    # -> 0.25, drift driven entirely by sampling noise in the model's replies.
    # No new outcomes means no new evidence; there is nothing to learn today.
    today = _closed_today(closed)
    if not today:
        log.info("reflection skipped: nothing closed today (window unchanged)")
        db.log_decision("brain", "reflection_skipped",
                        f"no trades closed today; {len(closed)}-trade window "
                        "unchanged since the last review")
        return None

    if not claude_available():
        log.warning("claude unavailable; skipping reflection")
        return None

    prompt = REFLECTION_PROMPT.format(
        n_window=len(closed),
        n_today=len(today),
        stats_block=_stats_block(closed),
        trades_block=_trade_rows(closed[:25]),
        open_block=(
            "\n".join(f"  {s['underlying']:<6} {s['structure']:<20} "
                      f"exp={s['expiry']} risk=${s['max_loss_total']:,.0f}"
                      for s in open_now) or "  (flat)"),
        tunables=", ".join(f"{k} [{lo}-{hi}]" for k, (lo, hi) in TUNABLE.items()),
    )

    payload = ask_for_json(prompt)
    if not payload:
        log.warning("reflection produced no usable reply")
        return None

    text = str(payload.get("reflection") or "").strip()
    lessons = [str(l) for l in (payload.get("lessons") or []) if str(l).strip()]
    applied = apply_adjustments(payload.get("parameter_adjustments") or {})

    realised = [float(s["realized_pnl"]) for s in closed
                if s.get("realized_pnl") is not None]
    stats = {
        "trades": len(realised),
        "closed_today": len(today),
        "total_pnl": round(sum(realised), 2) if realised else 0.0,
        "win_rate": (round(sum(1 for p in realised if p > 0) / len(realised), 3)
                     if realised else None),
        "confidence": payload.get("confidence"),
    }

    with db.connect() as conn:
        conn.execute(
            "INSERT INTO reflections (ts, text, lessons, params, stats) "
            "VALUES (?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), text,
             json.dumps(lessons), json.dumps(applied), json.dumps(stats)),
        )

    db.log_decision(
        "brain", "reflection",
        f"{len(lessons)} lesson(s); adjusted {list(applied) or 'nothing'}",
        payload={"stats": stats, "applied": applied, "lessons": lessons},
    )
    log.info("reflection: %d lesson(s), adjusted %s", len(lessons), applied or "nothing")
    return {"reflection": text, "lessons": lessons, "applied": applied, "stats": stats}
