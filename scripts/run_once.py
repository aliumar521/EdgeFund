"""One trading cycle. DRY_RUN=true by default -- set DRY_RUN=false to trade."""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from edgefund.core.config import SETTINGS
from edgefund.core.db import init_db
from edgefund.data.alpaca import AlpacaClient
from edgefund.strategy.cycle import run_cycle

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="actually place orders")
    ap.add_argument("--ramp", type=float, default=None, help="size multiplier 0-1")
    args = ap.parse_args()

    init_db()
    dry_run = not args.live

    with AlpacaClient() as client:
        clock = client.clock()
        print(f"market_open={clock['is_open']}  dry_run={dry_run}  "
              f"ramp={args.ramp if args.ramp is not None else SETTINGS.size_ramp}\n")
        summary = run_cycle(client, dry_run=dry_run, ramp=args.ramp)

    print("\n=== CYCLE SUMMARY ===")
    for k, v in summary.items():
        if k == "skips":
            continue
        print(f"  {k}: {v}")
    if summary.get("skips"):
        print("  skips:")
        for s in summary["skips"]:
            print(f"    - {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
