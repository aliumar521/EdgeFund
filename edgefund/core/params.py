"""Runtime-tunable parameters.

Closes the self-evolution loop. The reflection job writes clamped values into
`strategy_params`; this is what reads them back, falling through to the static
config default when the agent has not learned anything about a knob yet.

Values are cached briefly so the 60-second watchdog is not issuing a database
read per position per pass, but the TTL is short enough that a change made by
the evening reflection is live well before the next open.

Only keys in `reflect.TUNABLE` are ever written here. Risk limits live in a
frozen dataclass and deliberately have no path through this module.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from edgefund.core import db
from edgefund.core.config import SETTINGS

log = logging.getLogger("edgefund.params")

_CACHE: dict[str, float] = {}
_CACHE_AT: float = 0.0
_TTL_SECONDS = 30.0


def _refresh(force: bool = False) -> dict[str, float]:
    global _CACHE, _CACHE_AT
    if force or (time.monotonic() - _CACHE_AT) > _TTL_SECONDS:
        try:
            _CACHE = db.get_params()
        except Exception as exc:                 # a params read must never break trading
            log.warning("params read failed (%s); using config defaults", exc)
            _CACHE = {}
        _CACHE_AT = time.monotonic()
    return _CACHE


# A couple of tunables are named differently in the config than in the tuning
# vocabulary the reflection loop uses, so the fallback has to be told where to
# look. Without this the default resolves to None and the dashboard shows a
# tunable with no value.
_CONFIG_ALIAS = {"min_edge_score": "sell_vol_threshold"}


def _config_default(key: str) -> Any:
    return getattr(SETTINGS, _CONFIG_ALIAS.get(key, key), None)


def get(key: str, default: Any = None) -> Any:
    """Learned value for `key`, else the config default, else `default`."""
    learned = _refresh().get(key)
    if learned is not None:
        return learned
    if default is not None:
        return default
    return _config_default(key)


def snapshot() -> dict[str, Any]:
    """Every tunable with its effective value and where it came from."""
    from edgefund.brain.reflect import TUNABLE

    learned = _refresh(force=True)
    out: dict[str, Any] = {}
    for key in TUNABLE:
        config_default = _config_default(key)
        out[key] = {
            "value": learned.get(key, config_default),
            "source": "reflection" if key in learned else "config",
            "default": config_default,
        }
    return out


def invalidate() -> None:
    global _CACHE_AT
    _CACHE_AT = 0.0
