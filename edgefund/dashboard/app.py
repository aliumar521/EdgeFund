"""FastAPI dashboard and process entrypoint.

One process, one container: the scheduler starts inside the app's lifespan, so
`uvicorn edgefund.dashboard.app:app` runs the whole fund and serves the UI.
That keeps the deployment to a single Coolify service with a single volume.

The dashboard is strictly read-only. It renders what the agent recorded and has
no path to placing or closing a position -- there is no trading logic here at
all, and no endpoint that mutates state.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from edgefund.core import db, params
from edgefund.core.config import SETTINGS
from edgefund.data.alpaca import AlpacaClient
from edgefund.supervisor import build_scheduler, current_ramp, scheduler_status

log = logging.getLogger("edgefund.dashboard")

STATIC = Path(__file__).parent / "static"
_scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)

    db.init_db()
    _scheduler = build_scheduler()
    _scheduler.start()
    log.info("EdgeFund up (dry_run=%s ramp=%.2f)", SETTINGS.dry_run, current_ramp())
    try:
        yield
    finally:
        if _scheduler:
            _scheduler.shutdown(wait=False)
        log.info("EdgeFund down")


app = FastAPI(title="EdgeFund", lifespan=lifespan)


def _rows(query: str, args: tuple = (), parse: tuple[str, ...] = ()) -> list[dict]:
    with db.connect() as conn:
        try:
            raw = conn.execute(query, args).fetchall()
        except sqlite3.OperationalError:
            return []
    out = []
    for r in raw:
        d = dict(r)
        for key in parse:
            try:
                d[key] = json.loads(d.get(key) or "{}")
            except (json.JSONDecodeError, TypeError):
                d[key] = {}
        out.append(d)
    return out


# Tables reachable through /api/history, each with the column its rows are dated
# by and the JSON columns to decode. A whitelist rather than a free-form query
# parameter: the table name is interpolated into SQL, so it must never come
# straight from the caller.
HISTORY_TABLES: dict[str, tuple[str, tuple[str, ...]]] = {
    "decisions": ("ts", ("payload",)),
    "strategies": ("ts_open", ("entry_features",)),
    "reflections": ("ts", ("lessons", "params", "stats")),
    "directives": ("ts", ("body",)),
    "edge_snapshots": ("ts", ("detail",)),
    "equity_curve": ("ts", ()),
}


@app.get("/api/history/{table}")
def history(table: str, since: str | None = None, until: str | None = None,
            limit: int = 1000, order: str = "asc",
            kind: str | None = None, underlying: str | None = None,
            status: str | None = None) -> JSONResponse:
    """Full recorded history for one table, unlike /api/state's fixed caps.

    /api/state deliberately returns only a dashboard-sized slice (the last 60
    closed trades, 80 decisions, 10 reflections). That is right for rendering a
    page and wrong for answering "how has the fund done since we started" -- the
    older rows are simply unreachable. This endpoint is the archive: every AI
    directive and its rationale, every reflection, every skip reason, over any
    date range.

    Still strictly read-only, like the rest of this module.

        /api/history/directives?since=2026-09-01
        /api/history/strategies?status=closed&since=2026-09-01&limit=5000
        /api/history/decisions?kind=brain&since=2026-09-10
    """
    if table not in HISTORY_TABLES:
        return JSONResponse(
            {"error": f"unknown table {table!r}",
             "available": sorted(HISTORY_TABLES)}, status_code=404)

    ts_col, json_cols = HISTORY_TABLES[table]
    where: list[str] = []
    args: list[Any] = []

    # Dates compare as strings: every ts is stored as UTC ISO8601, which sorts
    # lexicographically. A bare date like "2026-09-01" therefore works as-is.
    if since:
        where.append(f"{ts_col} >= ?")
        args.append(since)
    if until:
        where.append(f"{ts_col} <= ?")
        args.append(until + "￿")     # make an end date inclusive of its day
    for col, val in (("kind", kind), ("underlying", underlying),
                     ("status", status)):
        if val and col in _columns(table):
            where.append(f"{col} = ?")
            args.append(val)

    clause = (" WHERE " + " AND ".join(where)) if where else ""
    direction = "DESC" if order.lower() == "desc" else "ASC"
    limit = max(1, min(int(limit), 20_000))
    args.append(limit)

    rows = _rows(f"SELECT * FROM {table}{clause} ORDER BY id {direction} LIMIT ?",
                 tuple(args), parse=json_cols)
    return JSONResponse({"table": table, "count": len(rows),
                         "since": since, "until": until, "rows": rows})


def _columns(table: str) -> set[str]:
    """Column names for a whitelisted table, so filters degrade rather than 500."""
    return {r["name"] for r in _rows(f"PRAGMA table_info({table})")}


@app.get("/api/health")
def health() -> JSONResponse:
    beats = {b["component"]: b["ts"] for b in _rows("SELECT * FROM heartbeat")}
    return JSONResponse({
        "ok": True,
        "dry_run": SETTINGS.dry_run,
        "ramp": current_ramp(),
        "heartbeat": beats,
        "now": datetime.now(timezone.utc).isoformat(),
    })


@app.get("/api/state")
def state() -> JSONResponse:
    """Everything the page renders, in one call."""
    account: dict[str, Any] = {}
    positions: list[dict[str, Any]] = []
    try:
        with AlpacaClient() as client:
            account = client.account()
            positions = client.option_positions()
    except Exception as exc:
        log.warning("live account fetch failed: %s", exc)
        account = {"error": str(exc)}

    equity = float(account.get("equity") or 0)
    last_equity = float(account.get("last_equity") or equity or 1)

    open_strategies = _rows(
        "SELECT * FROM strategies WHERE status IN ('pending','open','closing') "
        "ORDER BY id DESC", parse=("entry_features",))
    for s in open_strategies:
        s["legs"] = _rows("SELECT * FROM legs WHERE strategy_uid=? ORDER BY strike",
                          (s["strategy_uid"],))

    closed = _rows(
        "SELECT * FROM strategies WHERE status='closed' ORDER BY id DESC LIMIT 60",
        parse=("entry_features",))

    realised = [float(s["realized_pnl"]) for s in closed
                if s.get("realized_pnl") is not None and not s.get("dry_run")]
    wins = [p for p in realised if p > 0]

    # Return baseline: the first equity ever recorded, not a constant. The old
    # hardcoded $100,000 silently misreports total return the moment the account
    # is funded differently or reset.
    _first = _rows("SELECT equity FROM equity_curve ORDER BY id ASC LIMIT 1")
    baseline = float(_first[0]["equity"]) if _first else 100_000.0

    # Latest edge snapshot per underlying.
    edges = _rows(
        "SELECT e.* FROM edge_snapshots e "
        "JOIN (SELECT underlying, MAX(id) mid FROM edge_snapshots GROUP BY underlying) m "
        "ON e.id = m.mid ORDER BY e.edge_score DESC", parse=("detail",))

    return JSONResponse({
        "account": {
            "equity": equity,
            "cash": float(account.get("cash") or 0),
            "options_bp": float(account.get("options_buying_power") or 0),
            "day_pnl_pct": (equity - last_equity) / last_equity if last_equity else 0,
            "total_return_pct": ((equity / baseline - 1)
                                 if equity and baseline else 0),
            "baseline_equity": baseline,
            "options_level": account.get("options_trading_level"),
            "error": account.get("error"),
        },
        "stats": {
            "closed_trades": len(realised),
            "win_rate": (len(wins) / len(realised)) if realised else None,
            "realized_pnl": round(sum(realised), 2),
            "open_count": len(open_strategies),
            "deployed_risk": round(
                sum(float(s.get("max_loss_total") or 0) for s in open_strategies), 2),
            "leg_positions": len(positions),
        },
        "config": {
            "dry_run": SETTINGS.dry_run,
            "ramp": current_ramp(),
            "daily_halt_pct": SETTINGS.limits.daily_loss_halt_pct,
            "kill_switch_pct": SETTINGS.limits.kill_switch_pct,
            "max_concurrent": SETTINGS.limits.max_concurrent_strategies,
            "max_bp_deployed_pct": SETTINGS.limits.max_bp_deployed_pct,
            "options_feed": SETTINGS.options_feed,
        },
        "params": params.snapshot(),
        # Downsampled to one point per 10 minutes (ts is UTC ISO8601, so
        # substr(...,1,15) buckets on 'YYYY-MM-DDTHH:M'). The watchdog appends a
        # row every minute in session, so a flat LIMIT 500 covered well under a
        # day and made multi-week drawdowns invisible on the chart. Same row
        # budget, roughly a fortnight of history.
        "equity_curve": _rows(
            "SELECT ts, equity, open_pnl, day_pnl_pct FROM equity_curve "
            "WHERE id IN (SELECT MAX(id) FROM equity_curve GROUP BY substr(ts,1,15)) "
            "ORDER BY id DESC LIMIT 2000")[::-1],
        "edges": edges,
        "open_strategies": open_strategies,
        "closed": closed,
        "decisions": _rows(
            "SELECT * FROM decisions ORDER BY id DESC LIMIT 80", parse=("payload",)),
        "reflections": _rows(
            "SELECT * FROM reflections ORDER BY id DESC LIMIT 10",
            parse=("lessons", "params", "stats")),
        "directive": db.latest_directive(),
        "jobs": scheduler_status(_scheduler) if _scheduler else [],
        "heartbeat": {b["component"]: {"ts": b["ts"], "detail": b["detail"]}
                      for b in _rows("SELECT * FROM heartbeat")},
    })


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    page = STATIC / "index.html"
    if not page.exists():
        return HTMLResponse("<h1>EdgeFund</h1><p>dashboard asset missing</p>",
                            status_code=500)
    return HTMLResponse(page.read_text(encoding="utf-8"))
