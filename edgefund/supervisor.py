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
from datetime import date, datetime, time
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from edgefund.brain.reflect import run_reflection
from edgefund.brain.strategist import run_strategist
from edgefund.core import db
from edgefund.core.config import SETTINGS
from edgefund.data.alpaca import AlpacaClient
from edgefund.strategy.cycle import run_cycle
from edgefund.watchdog.monitor import ET, flatten_all, run_watchdog

log = logging.getLogger("edgefund.supervisor")

# Monday's ramp. The code will be hours old at the first bell, so the opening
# trades are deliberately small: their job is to prove that mleg orders fill and
# that exits fire, not to make money. Size scales up once that is observed.
RAMP_SCHEDULE: list[tuple[time, float]] = [
    (time(9, 30), 0.15),
    (time(11, 0), 0.35),
    (time(13, 0), 0.60),
    (time(15, 0), 1.00),
]

# Competition deadline is Fri 2026-09-04 at 11:00 ET. Positions are flattened
# before it so the submitted equity is realised P&L rather than a mark against
# wide indicative quotes.
FINAL_SWEEP_DATE = date(2026, 9, 4)
FINAL_SWEEP_TIME = time(10, 15)

# Watchdog runs at full cadence through the session plus the hour after the
# close, where late fills and cancellations still land. Outside that window
# options do not trade at all, so there is nothing to react to -- a slow idle
# pass is enough to keep the equity mark and the dashboard heartbeat honest.
WATCHDOG_ACTIVE_HOURS = (9, 16)     # inclusive, ET


def current_ramp(now: datetime | None = None) -> float:
    """Size multiplier for right now."""
    now = now or datetime.now(ET)
    if now.date() > date(2026, 8, 31):
        return 1.0                      # ramp only applies to launch day
    ramp = RAMP_SCHEDULE[0][1]
    for start, value in RAMP_SCHEDULE:
        if now.time() >= start:
            ramp = value
    return ramp


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


def job_final_sweep() -> None:
    """Realise everything before the submission deadline."""
    if datetime.now(ET).date() != FINAL_SWEEP_DATE:
        return
    try:
        with _client() as client:
            closed = flatten_all(client, "final sweep before competition deadline")
        log.info("final sweep closed %d structure(s)", closed)
    except Exception:
        log.exception("final sweep failed")


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

    sched.add_job(job_final_sweep,
                  CronTrigger(year=FINAL_SWEEP_DATE.year, month=FINAL_SWEEP_DATE.month,
                              day=FINAL_SWEEP_DATE.day, hour=FINAL_SWEEP_TIME.hour,
                              minute=FINAL_SWEEP_TIME.minute, timezone=ET),
                  id="final_sweep", name="final sweep before deadline")

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
