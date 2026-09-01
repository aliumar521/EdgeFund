"""Headless Claude calls via the Claude Code CLI.

Uses the existing Claude Code subscription (`claude -p`) rather than an API key,
so there is no second billing relationship to manage for a five-day competition.

Everything here is written on the assumption that this call *will* fail at some
point -- the binary missing in the container, a rate limit, a timeout, a
non-JSON reply. None of those may stop the agent trading, so every failure path
returns None and the caller falls back to the previous directive. The brain is
an advisor the system can lose without stopping.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from typing import Any

log = logging.getLogger("edgefund.claude")

DEFAULT_TIMEOUT = 240


def claude_available() -> bool:
    return _binary() is not None


def _binary() -> str | None:
    explicit = os.environ.get("CLAUDE_BIN")
    if explicit and os.path.exists(explicit):
        return explicit
    for name in ("claude", "claude.cmd", "claude.exe"):
        found = shutil.which(name)
        if found:
            return found
    return None


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a model reply.

    Models wrap JSON in prose or fences however they like, so this tries the
    whole string, then a fenced block, then the outermost balanced braces.
    """
    if not text:
        return None

    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def _json_candidates(text: str):
    stripped = text.strip()
    yield stripped

    fence = re.search(r"```(?:json)?\s*(.+?)```", stripped, re.DOTALL)
    if fence:
        yield fence.group(1).strip()

    start = stripped.find("{")
    if start == -1:
        return
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                yield stripped[start:i + 1]
                return


def ask_claude(prompt: str, timeout: int = DEFAULT_TIMEOUT) -> str | None:
    """Run one headless prompt. Returns the reply text, or None on any failure."""
    binary = _binary()
    if not binary:
        log.warning("claude binary not found on PATH; brain unavailable")
        return None

    cmd = [binary, "-p", prompt, "--output-format", "json"]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        log.warning("claude call timed out after %ss", timeout)
        return None
    except Exception as exc:
        log.warning("claude call failed to start: %s", exc)
        return None

    if proc.returncode != 0:
        log.warning("claude exited %s: %s", proc.returncode,
                    (proc.stderr or "")[:400])
        return None

    # --output-format json wraps the reply; fall back to raw stdout if the
    # wrapper shape ever changes.
    try:
        envelope = json.loads(proc.stdout)
        if isinstance(envelope, dict):
            if envelope.get("is_error"):
                log.warning("claude reported an error: %s",
                            str(envelope.get("result"))[:300])
                return None
            result = envelope.get("result")
            if isinstance(result, str):
                return result
    except json.JSONDecodeError:
        pass

    return proc.stdout or None


def ask_for_json(prompt: str, timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any] | None:
    """Ask for a JSON object and parse it, with one retry on a parse failure."""
    for attempt in (1, 2):
        reply = ask_claude(prompt, timeout=timeout)
        if reply is None:
            return None
        parsed = extract_json(reply)
        if parsed is not None:
            return parsed
        log.warning("claude reply was not parseable as JSON (attempt %d): %s",
                    attempt, reply[:300])
        prompt = (prompt + "\n\nYour previous reply could not be parsed. "
                  "Reply with ONLY a valid JSON object and nothing else.")
    return None
