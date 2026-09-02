# EdgeFund

**An autonomous options trading agent that finds its edge by measuring it.**

Built for the [Alpaca AI Trading Agents Hackathon](https://lablab.ai/ai-hackathons/alpaca-ai-trading-agents-hackathon). Trades US equity options on Alpaca paper trading, manages its own positions, and learns from its own trade journal.

---

## The thesis

Option-implied volatility is usually richer than the volatility an underlying subsequently realises. That gap is the **variance risk premium**, and it is the one thing EdgeFund trades.

It is not a vibe. It is two numbers, computed every cycle from a single chain call plus stock bars:

```
vrp_ratio  = ATM implied vol / realised vol      > 1  premium is rich
term_slope = front IV / short-end IV             > 1  backwardation, i.e. stress
```

A rich premium is the opportunity. Backwardation is the warning that the premium is rich *for a reason*, so the score rewards the first and penalises the second. **The sign of the result decides the direction of the trade**, which is what makes one thesis cover both sleeves:

| Signal | Reading | Action |
|---|---|---|
| `edge_score > +0.75` | implied is rich vs realised | **sell** defined-risk credit spreads / iron condors |
| `edge_score < -0.75` | implied is cheap vs realised | **buy** debit spreads for convexity |
| in between | no measurable edge | no position |

Trend never enters the score. It only decides how to *express* it — put-credit in an uptrend, call-credit in a downtrend, iron condor when flat. Mixing a directional view into a volatility signal is how a vol strategy quietly turns into a punt on direction.

### Normalisation is cross-sectional

The obvious approach — z-score each metric against a fixed constant — fails immediately. Measured live across the universe, *every* symbol showed a term slope near 1.4, so a prior centred on 1.0 flagged the whole market as stressed and would have refused to trade at all.

A market-wide level is not an edge. What is tradeable is **relative** richness: which names carry unusually rich premium versus their peers right now. Median and MAD across the universe deliver that on day one with no history, and a time-series z-score blends in as the agent accumulates its own observations. Absolute floors sit on top, so "least cheap in a uniformly cheap market" can never be mistaken for an opportunity.

### Structures are chosen by expected value, not rules of thumb

The first version gated candidates on a credit-to-width ratio. That heuristic cannot tell a good spread from a bad one — live, it rejected every candidate at 18-delta on 1-2 DTE, including clearly profitable ones, purely because short-dated OTM verticals structurally collect a small fraction of their width.

So structures are scored properly. Every viable strike combination is enumerated and its expiry payoff is integrated against a distribution built from **realised** vol rather than implied ([`strategy/payoff.py`](edgefund/strategy/payoff.py)). If the thesis holds, structures the market prices as fair show positive expected value to us — and the size of that gap *is* the variance risk premium. The same conviction that decides whether to trade also decides which contracts to trade.

---

## Architecture

Three tiers, and only the top one costs AI. That split is deliberate: **AI sets policy, deterministic code executes it, and the fund keeps trading safely if the AI is unavailable.**

```
┌─ Tier 3: BRAIN — Claude Code headless, 3 calls/day ──────────────┐
│  09:15 regime read · 12:30 book review · 16:15 reflection        │
│  emits a StrategyDirective (every field hard-clamped)            │
│  NEVER places an order                                           │
└────────────────────────┬─────────────────────────────────────────┘
                         │ posture: bias, aggression, vetoes, min_edge
┌────────────────────────▼─── Tier 2: SCANNER — no AI ─────────────┐
│  chain → realised vol → edge score → structure search by EV      │
│  → risk gate → sizing → mleg submission                          │
└────────────────────────┬─────────────────────────────────────────┘
                         │ orders
┌──────────────────▼─── Tier 1: WATCHDOG — no AI, 60s in session ──┐
│  reconcile fills · profit target · stop · delta stop · expiry     │
│  flatten · limit chase · circuit breakers                        │
└──────────────────────────────────────────────────────────────────┘
                         │
                    SQLite (WAL) ──→ FastAPI dashboard
```

Tiers 1 and 2 are plain Python against Alpaca REST. They cost nothing per run and are the parts that must always be on. Only Tier 3 spends AI, three times a day.

**The AI cannot reach the risk limits.** They live in a frozen dataclass ([`core/config.py`](edgefund/core/config.py)) with no path from the directive. Claude can move `aggression`, `min_edge_score`, `max_dte`, directional bias and veto symbols — each clamped by a Pydantic validator. That is the whole surface.

**Fallback chain:** `claude -p` output → previous stored directive → static defaults. A brain outage degrades posture-setting; it never stops the fund.

### Self-evolution, bounded

Every trade records the features that justified it — `vrp_ratio`, `term_slope`, trend, expected value, modelled probability of profit — alongside its outcome. The nightly reflection compares **belief against result** (if realised win rate sits well below modelled POP, the vol forecast is too optimistic) and writes concrete lessons plus parameter nudges. Each tunable is clamped to a range the model cannot argue its way out of; risk limits are not in the set.

---

## What the API actually allows

Verified live against the paper account rather than assumed from docs — several of these changed the design:

| Finding | Consequence |
|---|---|
| `options_trading_level: 3` | Spreads allowed. **Alpaca permits no naked shorts at any level**, so every structure is defined-risk by necessity as well as by choice. |
| `feed=opra` → *"OPRA agreement is not signed"* | All option data comes from the `indicative` feed. Quotes are model-derived and wide, which is why execution uses a limit chase rather than market orders. |
| `indicative` snapshots **do** carry `greeks` + `impliedVolatility` | The entire edge calculation is viable on the free feed. |
| Recent SIP is not entitled; historical SIP is | SIP for history (accurate realised vol), IEX for anything live, with automatic fallback. |
| `mleg` `limit_price` is **always positive** | True for credits and debits alike — direction is carried by leg side, never the sign. Confirmed by submitting a real test spread; no bundled skill documents this. |
| Positions return **leg by leg**, with nothing linking them | Spread grouping exists only in our database, keyed by `strategy_uid`. |

---

## Risk controls

Immutable, enforced in pure functions ([`risk/limits.py`](edgefund/risk/limits.py)) that nothing is allowed to route around:

```
max loss per position     3.0% of equity (core) / 1.5% (satellite)
max concurrent            16 structures
max risk per underlying   20% of the target deployed book
max buying power          80% deployed
daily loss halt          -10%   no new entries
kill switch              -18%   flatten everything
```

Position size is **derived from each structure's own max loss**, not picked and then checked — a riskier spread automatically produces a smaller contract count.

Exits are deterministic and need no AI: profit target at 55% of credit captured, stop at 2× credit, delta stop at 0.35 on a short leg, and an **unconditional flatten at 15:30 ET on expiry day** — a short option left into expiry risks assignment and pin gamma for no compensating edge.

---

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate      # (or source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env                                 # add your Alpaca paper keys

python scripts/smoke_edge.py                         # is the edge engine sane?
python scripts/run_once.py                           # one full cycle, dry run
python scripts/run_once.py --live --ramp 0.15        # actually trade, small
python -m pytest tests/ -q                           # risk + exit rule tests

uvicorn edgefund.dashboard.app:app --port 8000       # scheduler + dashboard
```

`DRY_RUN` gates every order submission. Set it to `true` to run the full scan/propose/log loop without sending anything to the broker.

### Deploy

```bash
docker compose up -d --build                         # dashboard on :8317
DASHBOARD_PORT=9000 docker compose up -d --build     # ...or any free host port
```

The container always listens on `8000`; only the host side is configurable, so a
host that already has `8000` claimed needs `DASHBOARD_PORT` set rather than an
edit to the image. On Coolify you can drop the published port altogether and let
its proxy route to the `expose`d container port via `SERVICE_FQDN_EDGEFUND_8000`.

Secrets are injected at runtime (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`) and never committed. SQLite sits on a named volume so history survives redeploys.

Coolify builds the same file: one service, build pack `Docker Compose`, the secrets set in the dashboard. `uvicorn edgefund.dashboard.app:app` starts the scheduler inside the FastAPI lifespan, so the web server *is* the fund — there is no second worker to deploy.

---

## Layout

```
edgefund/
  core/       config · pydantic models · sqlite · tunable params
  data/       alpaca REST client · volatility estimators
  edge/       the edge score (VRP, term structure, cross-sectional z)
  strategy/   payoff/EV engine · structure search · trading cycle
  risk/       pure sizing and limit functions
  execute/    mleg submission, limit chase, close, reconciliation
  watchdog/   position monitor: 60s in session, 30min idle off-hours
  brain/      claude headless · directive · strategist · reflection
  dashboard/  FastAPI + single-page UI
  supervisor.py
```

Volatility is estimated with Yang-Zhang on daily bars, and for short-dated contracts with intraday variance **plus the overnight gap** — measuring only the intraday piece drops every opening gap, understates SPY's realised vol by roughly a third, and would inflate every VRP ratio in the book.

## Acknowledgements

Patterns borrowed from [`ai-hedge-fund`](https://github.com/) (typed stage contracts, "conviction requests, risk disposes"), [`TradingAgents`](https://github.com/) (deferred outcome-triggered reflection), and Alpaca's `alpaca-risk-agent` reference (re-derive state every cycle; log *why* it skipped, not just what it did).
