# agent/evolution.py
"""
Evolution engine — tracks all self-modifications with versioning and rollback.

Every mutation the agent makes to its own prompts, LLM configs, or other
settings is recorded here with a reason, enabling auditing and rollback.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agent.logging_config import get_logger

log = get_logger("evolution")

JOURNAL_FILE = Path("evolution_journal.jsonl")


@dataclass
class EvolutionEntry:
    timestamp: str
    category: str        # "prompt" | "llm_config"
    key: str             # e.g. "mind_generate", "brain_llm"
    old_value: Any
    new_value: Any
    reason: str
    version: int         # per-key version counter
    success: bool = True


class EvolutionJournal:
    """Append-only, file-backed journal of every self-modification."""

    def __init__(self, path: Path = JOURNAL_FILE):
        self.path = path
        self._entries: list[EvolutionEntry] = []
        self._versions: dict[str, int] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self):
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = EvolutionEntry(**json.loads(line))
                self._entries.append(entry)
                self._versions[entry.key] = max(
                    self._versions.get(entry.key, 0), entry.version
                )
            except Exception as exc:
                log.warning("Skipping corrupt journal line: %s", exc)

    def _persist(self, entry: EvolutionEntry):
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    # ── Recording ────────────────────────────────────────────────────────

    def record(
        self,
        category: str,
        key: str,
        old_value: Any,
        new_value: Any,
        reason: str,
        success: bool = True,
    ) -> EvolutionEntry:
        ver = self._versions.get(key, 0) + 1
        self._versions[key] = ver
        entry = EvolutionEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            category=category,
            key=key,
            old_value=_safe(old_value),
            new_value=_safe(new_value),
            reason=reason,
            version=ver,
            success=success,
        )
        self._entries.append(entry)
        self._persist(entry)
        log.info(
            "Evolution [%s] %s v%d — %s",
            category, key, ver, reason[:120],
        )
        return entry

    # ── Querying ─────────────────────────────────────────────────────────

    def history(
        self,
        key: str | None = None,
        category: str | None = None,
        limit: int = 10,
    ) -> list[EvolutionEntry]:
        out = self._entries
        if key:
            out = [e for e in out if e.key == key]
        if category:
            out = [e for e in out if e.category == category]
        return out[-limit:]

    def last(self, key: str) -> EvolutionEntry | None:
        for e in reversed(self._entries):
            if e.key == key:
                return e
        return None

    def summary(self, limit: int = 20) -> str:
        """Human-readable summary of recent evolutions."""
        recent = self._entries[-limit:]
        if not recent:
            return "No evolutions recorded yet."
        lines = [
            f"[{e.timestamp[:19]}] {e.category}/{e.key} v{e.version} "
            f"{'OK' if e.success else 'FAIL'} — {e.reason[:80]}"
            for e in recent
        ]
        return "\n".join(lines)

    @property
    def total(self) -> int:
        return len(self._entries)


def _safe(value: Any) -> Any:
    """Ensure a value is JSON-serializable."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


# Singleton
journal = EvolutionJournal()
