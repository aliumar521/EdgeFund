# CLAUDE.md — working on EdgeFund

Guidance for Claude Code sessions in this repo. `README.md` explains *what the
strategy is and why*; this file explains *how to interrogate what it actually
did*. Read this first when asked about performance, decisions, or "how is the
agent doing".

---

## Orientation

EdgeFund trades the variance risk premium in US equity options on **Alpaca paper
trading**. One process runs everything: `uvicorn edgefund.dashboard.app:app`
starts the APScheduler inside the FastAPI lifespan, so **the web server is the
fund**. Deployed on Coolify from `docker-compose.yml`.

- **Live dashboard:** <https://edgefund.flyk.pro> — this is the source of truth.
- `ai-hedge-fund/`, `TradingAgents/`, `alpaca-risk-agent/` are **vendored
  reference repos with their own `.git`**. They are not part of the running
  system. Only `edgefund/`, `scripts/`, `tests/` matter.

### Three tiers

| Tier | What | AI? | Cadence |
|---|---|---|---|
| 3 `brain/` | sets posture, never places orders | yes, 3 calls/day | 09:15, 12:30, 16:15 ET |
| 2 `strategy/` | scan → edge → structure search → risk gate → submit | no | :15 and :45, 09:45–15:45 ET |
| 1 `watchdog/` | reconcile, exits, circuit breakers | no | every 60s in session |

Risk limits live in a **frozen dataclass** (`core/config.py::RiskLimits`) with no
path from the AI layer. The brain can only move `aggression`,
`directional_bias`, `min_edge_score`, `max_dte`, vetoes and preferred structures
— every one clamped by a Pydantic validator in `core/models.py`.

---

## Where to get the numbers

**Do not read `data_store/edgefund.db` in the repo — it is schema-only and
empty.** All real history lives in the Coolify Docker volume `edgefund_data` at
`/app/data_store/edgefund.db`. The Alpaca keys in `.env` have been rotated and
return 401; do not assume they work.

Reach the data over HTTP instead. Both endpoints are strictly read-only.

### `/api/state` — the dashboard slice

One call returning current account, stats, tunables, open positions, edges,
scheduler jobs. **Deliberately capped** (60 closed trades, 80 decisions, 10
reflections, 2000 equity points downsampled to 10-minute buckets). Right for a
snapshot, wrong for history — the older rows are simply not in it.

### `/api/history/{table}` — the archive

Use this for anything historical. Tables: `decisions`, `strategies`,
`reflections`, `directives`, `edge_snapshots`, `equity_curve`.

Params: `since`, `until` (bare dates work; `until` is inclusive of its day),
`limit` (max 20000), `order` (`asc`/`desc`), plus `kind`, `underlying`,
`status` where the table has that column.

```bash
curl -s "https://edgefund.flyk.pro/api/history/strategies?status=closed&since=2026-09-01&limit=5000"
curl -s "https://edgefund.flyk.pro/api/history/directives?since=2026-09-01"
curl -s "https://edgefund.flyk.pro/api/history/decisions?kind=brain&since=2026-09-10"
curl -s "https://edgefund.flyk.pro/api/history/equity_curve?since=2026-09-01&limit=20000"
```

---

## What the AI records about itself

Every brain action is persisted. This is the audit trail — nothing the AI
decides is lost.

| Table | Written by | Holds |
|---|---|---|
| `directives` | `brain/strategist.py` (09:15, 12:30) | full `StrategyDirective` JSON in `body`, incl. the model's own `rationale`, and `source` = `claude:<slot>` \| `previous` \| `fallback` |
| `reflections` | `brain/reflect.py` (16:15) | `text` (prose), `lessons` (JSON array), `params` (what it changed, as `{key: {from, to}}`), `stats` |
| `strategy_params` | `apply_adjustments()` | current value of each tunable + `source` (`reflection` \| `config`) |
| `decisions` | everywhere | `kind` ∈ `scan\|entry\|exit\|risk\|brain\|watchdog`, with a `reason` string — **including why a trade did *not* happen** |
| `strategies` | `strategy/`, `execute/` | one row per structure, with `entry_features` = what was believed at entry (`vrp_ratio`, `term_slope`, `trend`, `ev`, `pop`, …) next to `realized_pnl` and `exit_reason` |
| `edge_snapshots` | `edge/score.py` | per-symbol `vrp_ratio`, `term_slope`, `edge_score`, `regime` each scan |
| `equity_curve` | cycle + watchdog | `equity`, `cash`, `options_bp`, `open_pnl`, `day_pnl_pct` |

To answer *"why did it do X on date D"*: pull `directives` for that day (the
`rationale` is the model's stated reasoning), then `decisions?kind=brain`, then
`decisions` generally for the scan/entry/exit trail.

---

## Gotchas that will produce wrong answers

1. **Always filter `dry_run = 0`.** Dry-run rows share the `strategies` table
   with `realized_pnl = 0` and will dilute any win rate you compute.
2. **`realized_pnl` only exists on `status='closed'`.** Statuses are
   `pending | open | closing | closed | failed`.
3. **`net_credit` is per contract**; positive = credit received, negative =
   debit paid. `realized_pnl` is total dollars.
4. **Spread grouping exists only in this DB.** Alpaca returns option positions
   leg by leg with nothing linking them; `strategy_uid` is the only join key.
5. **`/api/state`'s `realized_pnl` is closed trades only**, while its `equity`
   comes from Alpaca and includes open marks. They are different measures — use
   `equity_curve.open_pnl` to bridge them.
6. **The reflection window is rolling, not daily.** `closed_strategies(limit=60)`
   has no date filter. Reflection now skips when nothing closed that day (see
   below), but historical `reflections` rows from before 2026-09-17 were
   generated from an unchanged sample and repeat each other.
7. **Timestamps are UTC ISO8601** and sort lexicographically; the scheduler runs
   on ET wall-clock. Mind the offset when bucketing by trading day.

---

## History worth knowing

- **2026-08-31 → 09-04:** hackathon window. Peaked above +20%.
- **2026-09-04:** the `final_sweep` job flattened the book at 10:15 ET. Any
  equity curve spanning this date has a hard discontinuity here.
- **2026-09-04 → 09-16: the agent traded nothing.** `strategist.py` still
  carried the competition deadline in its prompt, so the model correctly
  concluded there was nothing left to play for and set `max_dte=0` — which
  matches no expiry at all (`edge/score.py` needs `1 <= dte <= max_dte`) and
  halted entries outright. Fixed 2026-09-17: the mandate is now open-ended and
  the validator floors `max_dte` at 1.
- Over those 12 days the 16:15 reflection ran ten times on a byte-identical
  60-trade sample and moved parameters on nine of them. It now skips when
  nothing closed that day.

### Known open defect — POP vs the delta stop

Every reflection in that period independently reached the same correct
conclusion, and it is still unfixed:

> POP is computed as an **expiration-day** probability (`strategy/payoff.py`
> integrates the expiry payoff), but positions exit on an **intraday delta
> barrier** (`watchdog/monitor.py`, `delta_stop`). They model different events.

Result: 30% realised win rate against 71% modelled POP over 60 trades; 42 of 60
exits came via delta stop; iron condors went 0/13. The fix is not in `TUNABLE`,
so the self-tuning loop cannot reach it. Two options: recompute POP as a barrier
probability, or drop the delta stop on defined-risk credit spreads (max loss is
bounded by construction anyway). **Not yet decided — ask before acting.**

---

## Conventions when changing things

- **Every key in `reflect.TUNABLE` must be read back via `params.get()` in live
  code.** `tests/test_selfevolution.py` enforces this. A knob that is tuned but
  never read lets the model record that it fixed something when nothing changed
  — `target_short_delta` did exactly that for two weeks.
- **Risk limits are not negotiable from the AI layer.** Keep it that way.
- Posture values must stay tradable. A setting that makes trading structurally
  impossible is a broken state, not a cautious one — clamp it in the validator.
- Mark-to-market (`unrealised_pnl`) must use the same arithmetic as the realised
  figure in `execute/router.py`, or the two numbers stop being comparable.
- `flatten_all()` has no scheduled caller. It is the manual panic button.

## Commands

```bash
.venv/Scripts/python.exe -m pytest tests/ -q     # 39 tests, all must pass
python scripts/run_once.py                       # one cycle, dry run
python scripts/smoke_edge.py                     # edge engine sanity
uvicorn edgefund.dashboard.app:app --port 8000   # scheduler + dashboard
```

Deploying is a **Coolify redeploy** of this repo — nothing takes effect in
production until then.
