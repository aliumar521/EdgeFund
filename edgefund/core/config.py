"""Runtime configuration. Everything comes from the environment so the same
image runs locally and on Coolify with nothing secret in the repo."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]


def _f(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _b(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RiskLimits:
    """Immutable. The AI layer can never widen these -- see risk/limits.py.

    Sized for a paper-trading competition with an explicit mandate to be
    aggressive, but every structure is still defined-risk (Alpaca permits no
    naked shorts at any approval level, so max loss is always bounded).
    """

    max_loss_per_position_core_pct: float = 0.030   # 3.0% of equity
    max_loss_per_position_satellite_pct: float = 0.015
    max_concurrent_strategies: int = 16
    max_risk_per_underlying_pct: float = 0.20       # of total deployed risk
    max_bp_deployed_pct: float = 0.80               # of options buying power
    daily_loss_halt_pct: float = -0.10              # stop new entries
    kill_switch_pct: float = -0.18                  # flatten everything


@dataclass(frozen=True)
class Settings:
    api_key: str = os.environ.get("ALPACA_API_KEY", "")
    secret_key: str = os.environ.get("ALPACA_SECRET_KEY", "")
    trading_base: str = os.environ.get("ALPACA_TRADING_BASE", "https://paper-api.alpaca.markets")
    data_base: str = os.environ.get("ALPACA_DATA_BASE", "https://data.alpaca.markets")

    # OPRA is not signed on this account -- verified live. Indicative is the
    # only feed that returns quotes, and it does carry greeks + IV.
    options_feed: str = os.environ.get("OPTIONS_FEED", "indicative")

    # Recent SIP is not entitled on this account (403: "subscription does not
    # permit querying recent SIP data"), but historical SIP is. So: SIP for
    # history, where the full consolidated tape makes realised vol accurate,
    # and IEX for anything live.
    hist_feed: str = os.environ.get("HIST_FEED", "sip")
    live_feed: str = os.environ.get("LIVE_FEED", "iex")

    dry_run: bool = _b("DRY_RUN", True)
    size_ramp: float = _f("SIZE_RAMP", 0.15)  # Monday ramp: 0.15 -> 0.60 -> 1.0

    db_path: Path = ROOT / os.environ.get("DB_PATH", "data_store/edgefund.db")

    # Universe A trades daily expiries; Universe B is Friday weeklies.
    universe_daily: tuple[str, ...] = ("SPY", "QQQ", "IWM")
    universe_weekly: tuple[str, ...] = (
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD",
    )

    # Edge thresholds
    sell_vol_threshold: float = _f("SELL_VOL_THRESHOLD", 0.75)
    buy_vol_threshold: float = _f("BUY_VOL_THRESHOLD", -0.75)

    # Exit rules (deterministic, no AI)
    profit_target_pct: float = _f("PROFIT_TARGET_PCT", 0.55)   # of credit received
    stop_loss_mult: float = _f("STOP_LOSS_MULT", 2.0)          # x credit received
    delta_stop: float = _f("DELTA_STOP", 0.35)                 # short-leg |delta|

    # Structure construction
    target_short_delta: float = _f("TARGET_SHORT_DELTA", 0.18)
    min_credit_to_width: float = _f("MIN_CREDIT_TO_WIDTH", 0.15)
    max_dte: int = int(_f("MAX_DTE", 4))

    limits: RiskLimits = field(default_factory=RiskLimits)

    @property
    def auth_headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }


SETTINGS = Settings()
