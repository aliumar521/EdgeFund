"""Alpaca REST client (Trading + Market Data).

Raw REST on purpose. The `alpaca` CLI is in Alpha Preview and alpaca-py adds a
dependency surface we do not need; every endpoint used here was validated by
hand against the live paper account before this file was written.

Two behaviours worth knowing:

* Throttling is driven off the `X-RateLimit-Remaining` response header rather
  than a hard-coded requests-per-minute figure. The commonly cited 200/min is
  not in Alpaca's current docs, so we react to what the server actually says.
* Options quotes come from the `indicative` feed. This account has no OPRA
  agreement (verified: `{"message":"OPRA agreement is not signed"}`), so
  `feed=opra` is not an option. Indicative quotes are model-derived and can be
  wide, which is why execution uses a limit chase rather than trusting mid.
"""
from __future__ import annotations

import logging
import random
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

import httpx

from edgefund.core.config import SETTINGS

log = logging.getLogger("edgefund.alpaca")

_RETRY_STATUS = {429, 500, 502, 503, 504}


class AlpacaError(RuntimeError):
    def __init__(self, message: str, status: int = 0, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


class AlpacaClient:
    def __init__(self, settings=SETTINGS, timeout: float = 20.0):
        self.s = settings
        self._client = httpx.Client(
            timeout=timeout,
            headers={**settings.auth_headers, "accept": "application/json"},
        )
        self._min_remaining = 15   # start backing off before we hit the wall
        self._pause_until = 0.0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "AlpacaClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------- transport

    def _request(self, method: str, url: str, *, params: dict | None = None,
                 json_body: dict | None = None, attempts: int = 4) -> Any:
        delay = 1.0
        last: Exception | None = None

        for attempt in range(attempts):
            wait = self._pause_until - time.monotonic()
            if wait > 0:
                time.sleep(min(wait, 30.0))

            try:
                resp = self._client.request(method, url, params=params, json=json_body)
            except httpx.HTTPError as exc:            # network-level failure
                last = exc
                time.sleep(delay + random.random() * 0.3)
                delay = min(delay * 2, 8.0)
                continue

            self._note_rate_limit(resp)

            if resp.status_code in _RETRY_STATUS:
                retry_after = resp.headers.get("Retry-After")
                sleep_for = float(retry_after) if retry_after else delay
                log.warning("alpaca %s %s -> %s, retrying in %.1fs",
                            method, url, resp.status_code, sleep_for)
                time.sleep(sleep_for + random.random() * 0.3)
                delay = min(delay * 2, 8.0)
                last = AlpacaError(resp.text, resp.status_code)
                continue

            if resp.status_code >= 400:
                body: Any
                try:
                    body = resp.json()
                except Exception:
                    body = resp.text
                raise AlpacaError(
                    f"{method} {url} -> {resp.status_code}: {body}",
                    resp.status_code, body,
                )

            if not resp.content:
                return None
            return resp.json()

        raise AlpacaError(f"{method} {url} failed after {attempts} attempts: {last}")

    def _note_rate_limit(self, resp: httpx.Response) -> None:
        """Slow down as headroom shrinks instead of waiting to be throttled."""
        raw = resp.headers.get("X-RateLimit-Remaining")
        if raw is None:
            return
        try:
            remaining = int(raw)
        except ValueError:
            return
        if remaining <= self._min_remaining:
            reset = resp.headers.get("X-RateLimit-Reset")
            resume = 1.0
            if reset:
                try:
                    resume = max(0.0, int(reset) - time.time())
                except ValueError:
                    resume = 1.0
            self._pause_until = time.monotonic() + min(resume, 30.0)
            log.info("rate-limit headroom %s, pausing %.1fs", remaining, resume)

    def _trading(self, path: str, **kw: Any) -> Any:
        return self._request("GET", f"{self.s.trading_base}/v2{path}", **kw)

    def _data(self, path: str, **kw: Any) -> Any:
        return self._request("GET", f"{self.s.data_base}{path}", **kw)

    # ------------------------------------------------------------- account

    def account(self) -> dict[str, Any]:
        return self._trading("/account")

    def clock(self) -> dict[str, Any]:
        return self._trading("/clock")

    def calendar(self, start: str, end: str) -> list[dict[str, Any]]:
        return self._trading("/calendar", params={"start": start, "end": end})

    def positions(self) -> list[dict[str, Any]]:
        return self._trading("/positions") or []

    def option_positions(self) -> list[dict[str, Any]]:
        return [p for p in self.positions() if p.get("asset_class") == "us_option"]

    def portfolio_history(self, period: str = "1W", timeframe: str = "1D") -> dict[str, Any]:
        return self._trading("/account/portfolio/history",
                             params={"period": period, "timeframe": timeframe})

    # ------------------------------------------------------------- orders

    def submit_mleg(self, legs: list[dict[str, str]], qty: int, limit_price: float,
                    client_order_id: str, time_in_force: str = "day") -> dict[str, Any]:
        """Submit a multi-leg option order.

        `limit_price` is ALWAYS positive, for credits and debits alike --
        verified against the live paper API. The direction of premium flow is
        carried by each leg's side/position_intent, never by the sign.

        The response carries a parent order whose `legs[]` array holds the
        individual leg orders; that parent id is the only handle Alpaca gives
        us for treating the spread as one thing.
        """
        body = {
            "order_class": "mleg",
            "qty": str(qty),
            "type": "limit",
            "limit_price": f"{abs(limit_price):.2f}",
            "time_in_force": time_in_force,
            "client_order_id": client_order_id,
            "legs": legs,
        }
        return self._request("POST", f"{self.s.trading_base}/v2/orders", json_body=body)

    def get_order(self, order_id: str) -> dict[str, Any]:
        return self._trading(f"/orders/{order_id}")

    def list_orders(self, status: str = "open", limit: int = 100,
                    nested: bool = True) -> list[dict[str, Any]]:
        return self._trading("/orders", params={
            "status": status, "limit": limit, "nested": str(nested).lower(),
        }) or []

    def cancel_order(self, order_id: str) -> None:
        self._request("DELETE", f"{self.s.trading_base}/v2/orders/{order_id}")

    def replace_order(self, order_id: str, limit_price: float) -> dict[str, Any]:
        return self._request(
            "PATCH", f"{self.s.trading_base}/v2/orders/{order_id}",
            json_body={"limit_price": f"{abs(limit_price):.2f}"},
        )

    # ------------------------------------------------------------- contracts

    def option_contracts(self, underlying: str, *, expiration_date: str | None = None,
                         expiration_date_gte: str | None = None,
                         expiration_date_lte: str | None = None,
                         opt_type: str | None = None,
                         strike_gte: float | None = None,
                         strike_lte: float | None = None,
                         limit: int = 500) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"underlying_symbols": underlying, "limit": limit}
        if expiration_date:
            params["expiration_date"] = expiration_date
        if expiration_date_gte:
            params["expiration_date_gte"] = expiration_date_gte
        if expiration_date_lte:
            params["expiration_date_lte"] = expiration_date_lte
        if opt_type:
            params["type"] = opt_type
        if strike_gte is not None:
            params["strike_price_gte"] = strike_gte
        if strike_lte is not None:
            params["strike_price_lte"] = strike_lte

        out: list[dict[str, Any]] = []
        token: str | None = None
        while True:
            if token:
                params["page_token"] = token
            page = self._trading("/options/contracts", params=params)
            out.extend(page.get("option_contracts", []))
            token = page.get("next_page_token")
            if not token or len(out) >= limit:
                break
        return out

    def expirations(self, underlying: str, within_days: int = 10) -> list[str]:
        """Distinct expiry dates available in the near term, ascending."""
        today = datetime.now(timezone.utc).date()
        contracts = self.option_contracts(
            underlying,
            expiration_date_gte=today.isoformat(),
            expiration_date_lte=(today + timedelta(days=within_days)).isoformat(),
            opt_type="call",
            limit=500,
        )
        return sorted({c["expiration_date"] for c in contracts})

    # ------------------------------------------------------------- option data

    def option_chain(self, underlying: str, *, expiration_date: str | None = None,
                     expiration_date_gte: str | None = None,
                     expiration_date_lte: str | None = None,
                     strike_gte: float | None = None,
                     strike_lte: float | None = None,
                     opt_type: str | None = None,
                     limit: int = 500) -> dict[str, Any]:
        """Snapshots for a filtered slice of the chain.

        One call returns quote + greeks + impliedVolatility per contract, which
        is the entire input to the edge calculation.
        """
        params: dict[str, Any] = {"feed": self.s.options_feed, "limit": limit}
        if expiration_date:
            params["expiration_date"] = expiration_date
        if expiration_date_gte:
            params["expiration_date_gte"] = expiration_date_gte
        if expiration_date_lte:
            params["expiration_date_lte"] = expiration_date_lte
        if strike_gte is not None:
            params["strike_price_gte"] = strike_gte
        if strike_lte is not None:
            params["strike_price_lte"] = strike_lte
        if opt_type:
            params["type"] = opt_type

        snapshots: dict[str, Any] = {}
        token: str | None = None
        while True:
            if token:
                params["page_token"] = token
            page = self._data(f"/v1beta1/options/snapshots/{underlying}", params=params)
            snapshots.update(page.get("snapshots", {}))
            token = page.get("next_page_token")
            if not token or len(snapshots) >= limit:
                break
        return snapshots

    def option_snapshots(self, symbols: Iterable[str]) -> dict[str, Any]:
        syms = list(symbols)
        if not syms:
            return {}
        out: dict[str, Any] = {}
        for i in range(0, len(syms), 100):
            chunk = syms[i:i + 100]
            page = self._data("/v1beta1/options/snapshots", params={
                "symbols": ",".join(chunk), "feed": self.s.options_feed,
            })
            out.update(page.get("snapshots", {}))
        return out

    # ------------------------------------------------------------- stock data

    def stock_bars(self, symbols: str | Iterable[str], timeframe: str = "1Day",
                   start: str | None = None, end: str | None = None,
                   limit: int = 1000, feed: str | None = None) -> dict[str, list[dict]]:
        """Historical bars.

        Defaults to SIP (full consolidated volume, which makes for a much better
        realised-vol estimate). This account is not entitled to *recent* SIP, so
        a 403 on the tail of the window falls back to IEX rather than failing --
        verified: historical SIP bars are permitted, live ones are not.
        """
        syms = symbols if isinstance(symbols, str) else ",".join(symbols)

        def _fetch(f: str) -> dict[str, list[dict]]:
            params: dict[str, Any] = {
                "symbols": syms, "timeframe": timeframe, "limit": limit, "feed": f,
            }
            if start:
                params["start"] = start
            if end:
                params["end"] = end

            bars: dict[str, list[dict]] = {}
            token: str | None = None
            while True:
                if token:
                    params["page_token"] = token
                page = self._data("/v2/stocks/bars", params=params)
                for sym, rows in (page.get("bars") or {}).items():
                    bars.setdefault(sym, []).extend(rows)
                token = page.get("next_page_token")
                if not token:
                    break
            return bars

        primary = feed or self.s.hist_feed
        try:
            return _fetch(primary)
        except AlpacaError as exc:
            if exc.status == 403 and primary != self.s.live_feed:
                log.info("%s bars not entitled on %s, falling back to %s",
                         timeframe, primary, self.s.live_feed)
                return _fetch(self.s.live_feed)
            raise

    def latest_stock_price(self, symbols: Iterable[str]) -> dict[str, float]:
        """Spot price per symbol, from the live-entitled feed.

        Prefers the midpoint of the latest quote over the last trade: on a
        single-venue feed the last print can be stale by a few minutes on a
        quiet name, while the quote keeps updating.
        """
        syms = list(symbols)
        if not syms:
            return {}
        page = self._data("/v2/stocks/snapshots", params={
            "symbols": ",".join(syms), "feed": self.s.live_feed,
        })
        out: dict[str, float] = {}
        for sym, snap in (page or {}).items():
            if not isinstance(snap, dict):
                continue
            q = snap.get("latestQuote") or {}
            bid, ask = q.get("bp") or 0.0, q.get("ap") or 0.0
            if bid > 0 and ask > 0 and ask >= bid:
                out[sym] = round((bid + ask) / 2, 4)
                continue
            trade = snap.get("latestTrade") or {}
            if trade.get("p"):
                out[sym] = float(trade["p"])
                continue
            bar = snap.get("minuteBar") or snap.get("dailyBar") or {}
            if bar.get("c"):
                out[sym] = float(bar["c"])
        return out


def parse_occ(symbol: str) -> dict[str, Any]:
    """Decompose an OCC symbol, e.g. SPY260904P00700000.

    Root is variable length, so the fixed 15-character tail is parsed from the
    right rather than assuming a 3-letter root.
    """
    tail = symbol[-15:]
    root = symbol[:-15]
    yy, mm, dd = int(tail[0:2]), int(tail[2:4]), int(tail[4:6])
    return {
        "root": root,
        "expiry": date(2000 + yy, mm, dd),
        "opt_type": "call" if tail[6].upper() == "C" else "put",
        "strike": int(tail[7:]) / 1000.0,
    }


def build_occ(root: str, expiry: date, opt_type: str, strike: float) -> str:
    return (
        f"{root.upper()}{expiry:%y%m%d}"
        f"{'C' if opt_type.startswith('c') else 'P'}"
        f"{int(round(strike * 1000)):08d}"
    )
