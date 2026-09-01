"""SQLite persistence. WAL mode so the dashboard can read while the
supervisor writes.

Alpaca reports each option leg as its own position, so it cannot tell us that
two legs are one spread. That grouping only exists here: a `strategies` row
owns its `legs` rows and carries the Alpaca parent order id. Everything the
watchdog and dashboard do depends on that mapping.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from edgefund.core.config import SETTINGS

SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_uid      TEXT UNIQUE NOT NULL,
    ts_open           TEXT NOT NULL,
    ts_close          TEXT,
    underlying        TEXT NOT NULL,
    structure         TEXT NOT NULL,
    sleeve            TEXT NOT NULL,
    status            TEXT NOT NULL,          -- pending|open|closing|closed|failed
    qty               INTEGER NOT NULL,
    expiry            TEXT NOT NULL,
    net_credit        REAL NOT NULL,          -- per contract, + = received
    entry_fill_price  REAL,
    exit_fill_price   REAL,
    max_loss_total    REAL NOT NULL,
    parent_order_id   TEXT,
    close_order_id    TEXT,
    entry_features    TEXT NOT NULL DEFAULT '{}',
    exit_reason       TEXT,
    realized_pnl      REAL,
    dry_run           INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS legs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_uid    TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    ratio_qty       INTEGER NOT NULL,
    side            TEXT NOT NULL,
    position_intent TEXT NOT NULL,
    strike          REAL NOT NULL,
    expiry          TEXT NOT NULL,
    opt_type        TEXT NOT NULL,
    entry_price     REAL,
    entry_delta     REAL,
    entry_iv        REAL,
    FOREIGN KEY (strategy_uid) REFERENCES strategies(strategy_uid)
);
CREATE INDEX IF NOT EXISTS idx_legs_uid ON legs(strategy_uid);
CREATE INDEX IF NOT EXISTS idx_legs_symbol ON legs(symbol);

CREATE TABLE IF NOT EXISTS decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    kind        TEXT NOT NULL,      -- scan|entry|exit|risk|brain|watchdog
    underlying  TEXT,
    action      TEXT NOT NULL,      -- open|skip|close|halt|...
    reason      TEXT NOT NULL,
    payload     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts DESC);

CREATE TABLE IF NOT EXISTS edge_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    underlying  TEXT NOT NULL,
    spot        REAL NOT NULL,
    rv          REAL NOT NULL,
    atm_iv      REAL NOT NULL,
    vrp_ratio   REAL NOT NULL,
    term_slope  REAL NOT NULL,
    trend       REAL NOT NULL,
    edge_score  REAL NOT NULL,
    regime      TEXT NOT NULL,
    detail      TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_edge_ts ON edge_snapshots(ts DESC);
CREATE INDEX IF NOT EXISTS idx_edge_sym ON edge_snapshots(underlying, ts DESC);

CREATE TABLE IF NOT EXISTS directives (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      TEXT NOT NULL,
    source  TEXT NOT NULL,          -- claude|fallback|previous
    body    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reflections (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT NOT NULL,
    text     TEXT NOT NULL,
    lessons  TEXT NOT NULL DEFAULT '[]',
    params   TEXT NOT NULL DEFAULT '{}',
    stats    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS equity_curve (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    equity      REAL NOT NULL,
    cash        REAL NOT NULL,
    options_bp  REAL NOT NULL,
    open_pnl    REAL NOT NULL DEFAULT 0,
    day_pnl_pct REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_curve(ts DESC);

CREATE TABLE IF NOT EXISTS heartbeat (
    component TEXT PRIMARY KEY,
    ts        TEXT NOT NULL,
    detail    TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS strategy_params (
    key        TEXT PRIMARY KEY,
    value      REAL NOT NULL,
    updated_at TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'default'
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def db_path() -> Path:
    p = Path(SETTINGS.db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path(), timeout=15.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------- writes

def log_decision(kind: str, action: str, reason: str,
                 underlying: str = "", payload: dict[str, Any] | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO decisions (ts, kind, underlying, action, reason, payload) "
            "VALUES (?,?,?,?,?,?)",
            (_now(), kind, underlying, action, reason, json.dumps(payload or {}, default=str)),
        )


def record_edge(snap: dict[str, Any]) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO edge_snapshots "
            "(ts, underlying, spot, rv, atm_iv, vrp_ratio, term_slope, trend, edge_score, regime, detail) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                snap.get("ts", _now()), snap["underlying"], snap["spot"], snap["rv"],
                snap["atm_iv"], snap["vrp_ratio"], snap["term_slope"], snap["trend"],
                snap["edge_score"], snap.get("regime", "normal"),
                json.dumps(snap.get("detail", {}), default=str),
            ),
        )


def heartbeat(component: str, detail: str = "") -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO heartbeat (component, ts, detail) VALUES (?,?,?) "
            "ON CONFLICT(component) DO UPDATE SET ts=excluded.ts, detail=excluded.detail",
            (component, _now(), detail),
        )


def record_equity(equity: float, cash: float, options_bp: float,
                  open_pnl: float = 0.0, day_pnl_pct: float = 0.0) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO equity_curve (ts, equity, cash, options_bp, open_pnl, day_pnl_pct) "
            "VALUES (?,?,?,?,?,?)",
            (_now(), equity, cash, options_bp, open_pnl, day_pnl_pct),
        )


def save_directive(body: dict[str, Any], source: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO directives (ts, source, body) VALUES (?,?,?)",
            (_now(), source, json.dumps(body, default=str)),
        )


def latest_directive() -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT body FROM directives ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return json.loads(row["body"]) if row else None


# ---------------------------------------------------------------- strategies

def open_strategy(uid: str, candidate_row: dict[str, Any], legs: list[dict[str, Any]]) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO strategies (strategy_uid, ts_open, underlying, structure, sleeve, "
            "status, qty, expiry, net_credit, max_loss_total, parent_order_id, entry_features, dry_run) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                uid, _now(), candidate_row["underlying"], candidate_row["structure"],
                candidate_row["sleeve"], candidate_row.get("status", "pending"),
                candidate_row["qty"], candidate_row["expiry"], candidate_row["net_credit"],
                candidate_row["max_loss_total"], candidate_row.get("parent_order_id"),
                json.dumps(candidate_row.get("entry_features", {}), default=str),
                1 if candidate_row.get("dry_run", True) else 0,
            ),
        )
        for lg in legs:
            conn.execute(
                "INSERT INTO legs (strategy_uid, symbol, ratio_qty, side, position_intent, "
                "strike, expiry, opt_type, entry_price, entry_delta, entry_iv) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    uid, lg["symbol"], lg["ratio_qty"], lg["side"], lg["position_intent"],
                    lg["strike"], lg["expiry"], lg["opt_type"],
                    lg.get("entry_price"), lg.get("entry_delta"), lg.get("entry_iv"),
                ),
            )


def update_strategy(uid: str, **fields: Any) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    with connect() as conn:
        conn.execute(
            f"UPDATE strategies SET {cols} WHERE strategy_uid=?",
            (*fields.values(), uid),
        )


def active_strategies() -> list[dict[str, Any]]:
    """Everything not yet closed, with its legs attached."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM strategies WHERE status IN ('pending','open','closing') "
            "ORDER BY id DESC"
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["entry_features"] = json.loads(d.get("entry_features") or "{}")
            d["legs"] = [
                dict(x) for x in conn.execute(
                    "SELECT * FROM legs WHERE strategy_uid=?", (d["strategy_uid"],)
                ).fetchall()
            ]
            out.append(d)
    return out


def closed_strategies(limit: int = 100) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM strategies WHERE status='closed' ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["entry_features"] = json.loads(d.get("entry_features") or "{}")
            out.append(d)
    return out


def count_open_by_underlying() -> dict[str, int]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT underlying, COUNT(*) c FROM strategies "
            "WHERE status IN ('pending','open','closing') GROUP BY underlying"
        ).fetchall()
    return {r["underlying"]: r["c"] for r in rows}


def recent_decisions(limit: int = 60) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["payload"] = json.loads(d.get("payload") or "{}")
        out.append(d)
    return out


def get_params() -> dict[str, float]:
    with connect() as conn:
        rows = conn.execute("SELECT key, value FROM strategy_params").fetchall()
    return {r["key"]: r["value"] for r in rows}


def set_param(key: str, value: float, source: str = "reflection") -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO strategy_params (key, value, updated_at, source) VALUES (?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "updated_at=excluded.updated_at, source=excluded.source",
            (key, value, _now(), source),
        )
