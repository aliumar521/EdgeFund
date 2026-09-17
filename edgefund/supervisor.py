"""Job scheduling for the whole agent.

One process runs everything: the watchdog (every 60s through the session, every
30 minutes outside it), the periodic scan/trade cycle, and the three daily brain
calls. Cron was the obvious alternative but a single supervisor is a better fit
here -- it works identically on Windows and in the container, and keeps the
scheduler's own state observable on the dashboard.

Jobs are scheduled in UTC against explicit US/Eastern wall-clock times so that a
container running in any timezone behaves the same.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from edgefund.brain.reflect import run_reflection
from edgefund.brain.strategist import run_strategist
from edgefund.core import db
from edgefund.core.config import SETTINGS
from edgefund.data.alpaca import AlpacaClient
from edgefund.strategy.cycle import run_cycle
from edgefund.watchdog.monitor import ET, run_watchdog

log = logging.getLogger("edgefund.supervisor")

# Watchdog runs at full cadence through the session plus the hour after the
# close, where late fills and cancellations still land. Outside that window
# options do not trade at all, so there is nothing to react to -- a slow idle
# pass is enough to keep the equity mark and the dashboard heartbeat honest.
WATCHDOG_ACTIVE_HOURS = (9, 16)     # inclusive, ET


def current_ramp(now: datetime | None = None) -> float:
    """Global size multiplier.

    This was a launch-day ladder (0.15 -> 1.0 over the first session) whose job
    was to prove that mleg orders fill and exits fire before committing real
    size. That is long since observed, and the ladder had been returning a
    constant 1.0 since 2026-09-01 regardless of `SIZE_RAMP`.

    Kept as a function rather than inlined: it is the natural seam for a
    drawdown-responsive throttle, and the dashboard already surfaces it.
    """
    return 1.0


def in_watchdog_window(now: datetime | None = None) -> bool:
    """True when the watchdog should be running its full 60-second cadence."""
    now = now or datetime.now(ET)
    lo, hi = WATCHDOG_ACTIVE_HOURS
    return now.weekday() < 5 and lo <= now.hour <= hi


def _client() -> AlpacaClient:
    return AlpacaClient()


def job_watchdog(idle: bool = False) -> None:
    # The idle cron fires on :00/:30 around the clock, including inside the
    # active window. Bail out there rather than doubling up on the 60s job.
    if idle and in_watchdog_window():
        return
    try:
        with _client() as client:
            summary = run_watchdog(client, idle=idle)
        if summary.get("closed"):
            log.info("watchdog closed %d position(s): %s",
                     summary["closed"], summary.get("actions"))
    except Exception:
        log.exception("watchdog job failed")
        db.heartbeat("watchdog", "error -- see logs")


def job_cycle() -> None:
    try:
        ramp = current_ramp()
        with _client() as client:
            if not client.clock().get("is_open"):
                log.info("cycle skipped: market closed")
                return
            summary = run_cycle(client, ramp=ramp)
        log.info("cycle: scanned=%s proposed=%s opened=%s ramp=%.2f",
                 summary["scanned"], summary["proposed"], summary["opened"], ramp)
    except Exception:
        log.exception("cycle job failed")
        db.log_decision("scan", "error", "cycle raised an exception; see logs")


def job_strategist(slot: str) -> None:
    try:
        with _client() as client:
            run_strategist(client, slot)
    except Exception:
        log.exception("strategist job (%s) failed", slot)


def job_reflection() -> None:
    try:
        with _client() as client:
            run_reflection(client)
    except Exception:
        log.exception("reflection job failed")


def build_scheduler() -> BackgroundScheduler:
    sched = BackgroundScheduler(timezone=ET, job_defaults={
        "coalesce": True,          # a missed run is replaced, never queued up
        "max_instances": 1,        # never let two watchdog passes overlap
        "misfire_grace_time": 45,
    })

    # Watchdog, full cadence: every minute of the session, extended an hour past
    # the close so late fills and cancellations get reconciled.
    sched.add_job(job_watchdog,
                  CronTrigger(day_of_week="mon-fri", hour="9-16", minute="*",
                              second=0, timezone=ET),
                  id="watchdog", name="position watchdog")
    # Watchdog, idle: overnight and weekends nothing can be traded or exited, so
    # a half-hourly pass is enough to keep the equity mark and heartbeat fresh.
    # It self-skips inside the active window (see job_watchdog).
    sched.add_job(job_watchdog,
                  CronTrigger(minute="0,30", timezone=ET), kwargs={"idle": True},
                  id="watchdog_idle", name="watchdog idle heartbeat")

    # Scan/trade cycle: every 30 minutes from 09:45 to 15:45 ET. The first 15
    # minutes after the bell are skipped, where opening rotation makes quotes
    # and greeks unreliable.
    sched.add_job(job_cycle,
                  CronTrigger(day_of_week="mon-fri", hour="9-15", minute="45",
                              timezone=ET),
                  id="cycle", name="scan and trade")
    sched.add_job(job_cycle,
                  CronTrigger(day_of_week="mon-fri", hour="10-15", minute="15",
                              timezone=ET),
                  id="cycle_mid", name="scan and trade (mid-hour)")

    # Brain: three calls a day, the only jobs that spend AI.
    sched.add_job(job_strategist, CronTrigger(day_of_week="mon-fri", hour=9, minute=15,
                                              timezone=ET),
                  args=["premarket"], id="brain_premarket", name="premarket regime read")
    sched.add_job(job_strategist, CronTrigger(day_of_week="mon-fri", hour=12, minute=30,
                                              timezone=ET),
                  args=["midday"], id="brain_midday", name="midday book review")
    sched.add_job(job_reflection, CronTrigger(day_of_week="mon-fri", hour=16, minute=15,
                                              timezone=ET),
                  id="brain_reflection", name="end of day reflection")

    return sched


def scheduler_status(sched: BackgroundScheduler) -> list[dict[str, Any]]:
    return [
        {
            "id": job.id,
            "name": job.name,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
        }
        for job in sched.get_jobs()
    ]


def main() -> None:
    """Run the supervisor standalone, without the dashboard."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)

    db.init_db()
    sched = build_scheduler()
    sched.start()
    log.info("supervisor started (dry_run=%s, ramp=%.2f)",
             SETTINGS.dry_run, current_ramp())
    for job in scheduler_status(sched):
        log.info("  job %-16s next: %s", job["id"], job["next_run"])

    try:
        import time as _time
        while True:
            _time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        sched.shutdown()
        log.info("supervisor stopped")


if __name__ == "__main__":
    main()
