"""Support helpers for the Flask runtime such as result persistence."""
from __future__ import annotations

import base64
import json
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from werkzeug.utils import secure_filename


def _now() -> datetime:
    """Return an aware :class:`datetime` in UTC for timestamps."""

    return datetime.now(UTC)


@dataclass(slots=True)
class ResultRecord:
    """Persisted metadata for a processed output."""

    identifier: str
    original_name: str
    result_name: str
    path: Path
    mime_type: str
    created_at: datetime
    options: dict[str, Any] = field(default_factory=dict)
    size_bytes: int = 0

    def as_dict(self, *, include_preview: bool = False) -> dict[str, Any]:
        """Return a serialisable dictionary representation of the record."""

        payload: dict[str, Any] = {
            "id": self.identifier,
            "original_name": self.original_name,
            "result_name": self.result_name,
            "mime_type": self.mime_type,
            "created_at": self.created_at.isoformat(),
            "size_bytes": self.size_bytes,
            "options": self.options,
        }
        if include_preview and self.mime_type.startswith("image/") and self.path.exists():
            payload["preview_data_uri"] = build_data_uri(self.path, self.mime_type)
        return payload


def build_data_uri(path: Path, mime_type: str) -> str:
    """Return a data URI for ``path`` using ``mime_type``."""

    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


class ResultStore:
    """Disk-backed store for processed results and user history."""

    def __init__(self, base_dir: Path, *, history_limit: int = 50) -> None:
        self.base_dir = base_dir
        self.history_limit = max(1, history_limit)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._history_path = self.base_dir / "history.json"
        self._lock = threading.Lock()
        self._records = self._load_history()

    # ------------------------------------------------------------------
    # Record management
    # ------------------------------------------------------------------
    def add_record(self, record: ResultRecord) -> ResultRecord:
        """Persist ``record`` and return it for convenience."""

        with self._lock:
            self._records.append(record)
            self._records = self._records[-self.history_limit :]
            self._persist()
        return record

    def get(self, identifier: str) -> ResultRecord | None:
        """Return a record by ``identifier`` or ``None``."""

        with self._lock:
            for record in reversed(self._records):
                if record.identifier == identifier:
                    return record
        return None

    def list_records(self) -> list[ResultRecord]:
        """Return the stored records newest first."""

        with self._lock:
            return list(sorted(self._records, key=lambda item: item.created_at, reverse=True))

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def _load_history(self) -> list[ResultRecord]:
        if not self._history_path.exists():
            return []
        try:
            data = json.loads(self._history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

        records: list[ResultRecord] = []
        for payload in data:
            try:
                created_at = datetime.fromisoformat(payload["created_at"])
            except (KeyError, ValueError):
                created_at = _now()
            record = ResultRecord(
                identifier=payload.get("id", ""),
                original_name=payload.get("original_name", ""),
                result_name=payload.get("result_name", ""),
                path=Path(payload.get("path", "")),
                mime_type=payload.get("mime_type", "application/octet-stream"),
                created_at=created_at,
                options=payload.get("options", {}),
                size_bytes=int(payload.get("size_bytes", 0)),
            )
            records.append(record)
        return records

    def _persist(self) -> None:
        payload = [
            {
                "id": record.identifier,
                "original_name": record.original_name,
                "result_name": record.result_name,
                "path": str(record.path),
                "mime_type": record.mime_type,
                "created_at": record.created_at.isoformat(),
                "options": record.options,
                "size_bytes": record.size_bytes,
            }
            for record in self._records
        ]
        self._history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_filename(name: str, default: str = "output.png") -> str:
    """Return a safe filename, falling back to ``default`` if required."""

    candidate = secure_filename(name)
    return candidate or default


def total_size(paths: Iterable[Path]) -> int:
    """Return the combined file size of ``paths`` in bytes."""

    total = 0
    for path in paths:
        if path.exists():
            total += path.stat().st_size
    return total


__all__ = [
    "ResultRecord",
    "ResultStore",
    "build_data_uri",
    "ensure_filename",
    "total_size",
]
