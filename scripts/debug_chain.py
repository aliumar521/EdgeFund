from __future__ import annotations
import pathlib, sys, logging
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from datetime import date
from edgefund.core.config import SETTINGS
from edgefund.core.db import init_db
from edgefund.data.alpaca import AlpacaClient
from edgefund.edge.score import scan_universe
from edgefund.strategy import spreads as sp
from edgefund.brain.directive import static_default

logging.basicConfig(level=logging.WARNING)
init_db()
syms = sys.argv[1:] or ["IWM", "AAPL", "AMZN"]

with AlpacaClient() as c:
    snaps = {s.underlying: s for s in scan_universe(c, syms, SETTINGS.max_dte)}
    for sym in syms:
        snap = snaps.get(sym)
        if not snap:
            print(f"{sym}: no snapshot"); continue
        exp = snap.detail["expiry"]; dte = snap.detail["dte"]
        band = max(snap.spot * 0.14, 8.0)
        raw = c.option_chain(sym, expiration_date=exp,
                             strike_gte=snap.spot-band, strike_lte=snap.spot+band)
        ch = sp.ChainView(raw, date.fromisoformat(exp))
        print(f"\n=== {sym} spot={snap.spot:.2f} exp={exp} dte={dte} edge={snap.edge_score:.2f}")
        print(f"    raw contracts={len(raw)}  puts kept={len(ch.puts)} calls kept={len(ch.calls)}")
        tradable_puts = [r for r in ch.puts.values() if ch.tradable(r)]
        print(f"    tradable puts={len(tradable_puts)} (of {len(ch.puts)})")
        untr = [r for r in ch.puts.values() if not ch.tradable(r)][:3]
        for r in untr:
            rel = (r['ask']-r['bid'])/r['mid'] if r['mid'] else 9
            print(f"      rejected K={r['strike']:.0f} bid={r['bid']:.2f} ask={r['ask']:.2f} "
                  f"mid={r['mid']:.3f} rel={rel:.2f} (minmid={sp.MIN_LEG_MID} maxrel={sp.MAX_REL_SPREAD})")
        short = ch.by_delta("put", SETTINGS.target_short_delta)
        if not short:
            print("    !! by_delta(put) found nothing"); continue
        print(f"    short leg: K={short['strike']:.0f} delta={short['delta']:.3f} "
              f"bid={short['bid']:.2f} ask={short['ask']:.2f} mid={short['mid']:.3f}")
        width = sp._target_width(snap.spot)
        lr = ch.nearest_strike("put", short['strike']-width)
        if not lr:
            print("    !! no long leg"); continue
        aw = abs(short['strike']-lr['strike'])
        credit = short['mid']-lr['mid']
        print(f"    long leg:  K={lr['strike']:.0f} mid={lr['mid']:.3f}")
        print(f"    target_width={width:.2f} actual_width={aw:.2f} "
              f"(allowed {snap.spot*sp.MIN_WIDTH_PCT:.2f}..{snap.spot*sp.MAX_WIDTH_PCT:.2f})")
        print(f"    credit={credit:.3f} credit/width={credit/aw if aw else 0:.3f} "
              f"(need >= {SETTINGS.min_credit_to_width})")
        avail = sorted(ch.puts)
        print(f"    strike grid near short: {[f'{k:.0f}' for k in avail if abs(k-short['strike'])<=12][:14]}")
