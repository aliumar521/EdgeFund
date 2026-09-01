"""Pydantic contracts between pipeline stages.

Borrowed from ai-hedge-fund's design: every hop between stages is a typed
model, never a loose dict, so a stage can be tested in isolation and the whole
decision is serialisable into the audit log.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Side = Literal["buy", "sell"]
PositionIntent = Literal["buy_to_open", "sell_to_open", "buy_to_close", "sell_to_close"]
OptType = Literal["call", "put"]
Structure = Literal[
    "put_credit_spread", "call_credit_spread", "iron_condor",
    "call_debit_spread", "put_debit_spread",
]
Sleeve = Literal["core", "satellite"]
Regime = Literal["calm", "normal", "stressed", "event"]


class Leg(BaseModel):
    """One option contract inside a structure."""

    symbol: str
    ratio_qty: int = 1
    side: Side
    position_intent: PositionIntent
    strike: float
    expiry: date
    opt_type: OptType
    bid: float = 0.0
    ask: float = 0.0
    delta: float = 0.0
    iv: float = 0.0
    open_interest: int = 0

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return round((self.bid + self.ask) / 2, 4)
        return self.ask or self.bid

    @property
    def spread_width_pct(self) -> float:
        """Bid-ask width as a fraction of mid -- our liquidity proxy, since the
        indicative feed gives no size we can trust."""
        m = self.mid
        if m <= 0:
            return 1.0
        return (self.ask - self.bid) / m

    def to_alpaca_leg(self) -> dict[str, str]:
        return {
            "symbol": self.symbol,
            "ratio_qty": str(self.ratio_qty),
            "side": self.side,
            "position_intent": self.position_intent,
        }


class EdgeSnapshot(BaseModel):
    """The measured volatility edge for one underlying at one point in time.

    This is the whole thesis in one object: vrp_ratio says how rich implied vol
    is versus what the underlying is actually realising, and its sign decides
    whether we sell premium or buy convexity.
    """

    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    underlying: str
    spot: float
    rv: float                 # horizon-matched realised vol, annualised
    atm_iv: float             # ATM implied vol for the target expiry
    vrp_ratio: float          # atm_iv / rv   -- >1 means premium is rich
    term_slope: float         # front IV / ~30d IV -- >1 means backwardation
    trend: float              # -1..+1 normalised trend alignment
    edge_score: float
    regime: Regime = "normal"
    detail: dict[str, Any] = Field(default_factory=dict)


class SpreadCandidate(BaseModel):
    """A proposed structure. Proposals only -- risk/limits.py decides size and
    whether it trades at all ("conviction requests, risk disposes")."""

    underlying: str
    structure: Structure
    sleeve: Sleeve
    legs: list[Leg]
    net_credit: float         # positive = we receive premium, negative = we pay
    width: float              # widest wing, in points
    max_loss_per_contract: float
    max_profit_per_contract: float
    short_delta: float
    dte: int
    expiry: date
    edge_score: float
    features: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_credit(self) -> bool:
        return self.net_credit > 0

    @property
    def credit_to_width(self) -> float:
        return abs(self.net_credit) / self.width if self.width else 0.0

    @property
    def limit_price(self) -> float:
        """Alpaca mleg limit_price is ALWAYS positive regardless of whether the
        structure is a net credit or a net debit -- verified empirically against
        the paper API. Direction is carried by the leg sides, not the sign."""
        return round(abs(self.net_credit), 2)


class SizedOrder(BaseModel):
    candidate: SpreadCandidate
    qty: int
    total_max_loss: float
    total_credit: float
    reason: str


class RiskDecision(BaseModel):
    approved: bool
    qty: int = 0
    reason: str


def _clamp(lo: float, hi: float):
    def _v(v: float) -> float:
        return max(lo, min(hi, v))
    return _v


class StrategyDirective(BaseModel):
    """Policy emitted by the Claude layer. Every field is hard-clamped: the AI
    tunes behaviour inside a box it cannot open. Risk limits are NOT here --
    they live in the frozen RiskLimits dataclass and are unreachable from here.
    """

    regime: Regime = "normal"
    directional_bias: float = 0.0     # -1 bearish .. +1 bullish
    aggression: float = 1.0           # multiplier on position size
    allowed_underlyings: list[str] = Field(default_factory=list)
    vetoed_underlyings: list[str] = Field(default_factory=list)
    preferred_structures: list[Structure] = Field(default_factory=list)
    min_edge_score: float = 0.75
    max_dte: int = 4
    rationale: str = "static default directive"
    lessons: list[str] = Field(default_factory=list)
    source: str = "fallback"

    @field_validator("directional_bias")
    @classmethod
    def _bias(cls, v: float) -> float:
        return _clamp(-1.0, 1.0)(v)

    @field_validator("aggression")
    @classmethod
    def _aggr(cls, v: float) -> float:
        return _clamp(0.3, 1.5)(v)

    @field_validator("min_edge_score")
    @classmethod
    def _mes(cls, v: float) -> float:
        return _clamp(0.3, 2.0)(v)

    @field_validator("max_dte")
    @classmethod
    def _dte(cls, v: int) -> int:
        return int(max(0, min(7, v)))


class Decision(BaseModel):
    """Audit record. Written for skips as well as trades -- the reason a trade
    did NOT happen is as much a part of the story as one that did."""

    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    kind: str
    underlying: str = ""
    action: str
    reason: str
    payload: dict[str, Any] = Field(default_factory=dict)
