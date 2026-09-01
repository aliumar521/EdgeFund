"""Smoke test: does the edge engine produce sane numbers against live chains?

Not a backtest -- this only asserts the plumbing works and the values land in
plausible ranges before any capital is committed.
"""
from __future__ import annotations

import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from edgefund.core.config import SETTINGS
from edgefund.core.db import init_db, record_edge
from edgefund.data.alpaca import AlpacaClient
from edgefund.edge.score import scan_universe

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def main() -> int:
    init_db()
    universe = list(SETTINGS.universe_daily) + list(SETTINGS.universe_weekly)
    bad = 0

    with AlpacaClient() as client:
        acct = client.account()
        clock = client.clock()
        print(f"\nequity=${float(acct['equity']):,.0f}  "
              f"options_bp=${float(acct['options_buying_power']):,.0f}  "
              f"level={acct['options_trading_level']}  "
              f"market_open={clock['is_open']}\n")

        snaps = scan_universe(client, universe, SETTINGS.max_dte)
        if not snaps:
            print("no snapshots produced")
            return 1

        hdr = (f"{'sym':<6}{'spot':>9}{'rv':>7}{'iv':>7}{'vrp':>7}{'term':>7}"
               f"{'xsVRP':>7}{'xsTRM':>7}{'trend':>7}{'edge':>8}  {'regime':<9}{'exp':<12}")
        print(hdr)
        print("-" * len(hdr))

        for s in sorted(snaps, key=lambda x: -x.edge_score):
            d = s.detail
            record_edge(s.model_dump(mode="json"))
            flag = "  <-- " + d["gated"] if d.get("gated") else ""
            print(f"{s.underlying:<6}{s.spot:>9.2f}{s.rv:>7.3f}{s.atm_iv:>7.3f}"
                  f"{s.vrp_ratio:>7.2f}{s.term_slope:>7.2f}"
                  f"{d['xs_vrp']:>7.2f}{d['xs_term']:>7.2f}{s.trend:>7.2f}"
                  f"{s.edge_score:>8.2f}  {s.regime:<9}{d['expiry']:<12}{flag}")

            if not (0.01 < s.rv < 3.0):
                print(f"   !! rv out of range: {s.rv}"); bad += 1
            if not (0.01 < s.atm_iv < 4.0):
                print(f"   !! atm_iv out of range: {s.atm_iv}"); bad += 1
            if not (0.1 < s.vrp_ratio < 10.0):
                print(f"   !! vrp_ratio implausible: {s.vrp_ratio}"); bad += 1

        c = snaps[0].detail["universe_centre"]
        print(f"\nuniverse medians: vrp={c['vrp']:.2f} term={c['term']:.2f}  "
              f"basis={snaps[0].detail['basis']}")
        sells = [s for s in snaps if s.edge_score > 0.75]
        buys = [s for s in snaps if s.edge_score < -0.75]
        print(f"would SELL premium: {[s.underlying for s in sells]}")
        print(f"would BUY convexity: {[s.underlying for s in buys]}")

    print(f"\n{'FAILED' if bad else 'OK'}: {bad} sanity problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
