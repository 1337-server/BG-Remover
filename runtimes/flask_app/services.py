"""Support helpers for the Flask runtime such as result persistence."""
from __future__ import annotations

import base64
import json
import queue
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

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


@dataclass(slots=True)
class BatchEvent:
    """Server-sent event payload emitted during batch processing."""

    event: Literal["started", "item_success", "item_error", "finished"]
    data: dict[str, Any]


@dataclass(slots=True)
class BatchJobSummary:
    """Immutable snapshot describing the final state of a batch job."""

    job_id: str
    output_dir: Path
    started_at: datetime
    finished_at: datetime | None
    total: int
    successes: int
    failures: int
    success_items: list[dict[str, Any]]
    failure_items: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        """Return a serialisable representation of the batch summary."""

        return {
            "job_id": self.job_id,
            "output_dir": str(self.output_dir),
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "counts": {
                "total": self.total,
                "success": self.successes,
                "failed": self.failures,
            },
            "successes": list(self.success_items),
            "failures": list(self.failure_items),
        }


class BatchJob:
    """Background job coordinating folder processing and SSE updates."""

    def __init__(self, job_id: str, output_dir: Path) -> None:
        self.job_id = job_id
        self.output_dir = output_dir
        self.started_at = _now()
        self.finished_at: datetime | None = None
        self._events: queue.Queue[BatchEvent | None] = queue.Queue()
        self._successes: list[dict[str, Any]] = []
        self._failures: list[dict[str, Any]] = []
        self._total = 0
        self._lock = threading.Lock()
        self._status: Literal["pending", "running", "finished", "failed"] = "pending"
        self._error_message: str | None = None

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def publish(self, event: BatchEvent) -> None:
        """Queue ``event`` for streaming to connected clients."""

        self._events.put(event)

    def event_stream(self):
        """Yield events until the job signals completion."""

        while True:
            item = self._events.get()
            if item is None:
                break
            yield item

    def close_stream(self) -> None:
        """Signal any listeners that the stream has finished."""

        self._events.put(None)

    # ------------------------------------------------------------------
    # State tracking
    # ------------------------------------------------------------------
    @property
    def status(self) -> Literal["pending", "running", "finished", "failed"]:
        with self._lock:
            return self._status

    def mark_started(self, *, total: int | None = None) -> None:
        """Mark the job as running and optionally record ``total`` items."""

        with self._lock:
            self._status = "running"
            if total is not None:
                self._total = int(total)

    def record_success(self, payload: dict[str, Any]) -> None:
        """Record a successful item event."""

        with self._lock:
            self._successes.append(payload)

    def record_failure(self, payload: dict[str, Any]) -> None:
        """Record a failed item event."""

        with self._lock:
            self._failures.append(payload)

    def mark_finished(self, *, error: str | None = None) -> None:
        """Transition the job into its terminal state."""

        with self._lock:
            self.finished_at = _now()
            self._status = "failed" if error else "finished"
            self._error_message = error

    def build_summary(self) -> BatchJobSummary:
        """Return a :class:`BatchJobSummary` representing the current state."""

        with self._lock:
            summary = BatchJobSummary(
                job_id=self.job_id,
                output_dir=self.output_dir,
                started_at=self.started_at,
                finished_at=self.finished_at,
                total=self._total or (len(self._successes) + len(self._failures)),
                successes=len(self._successes),
                failures=len(self._failures),
                success_items=list(self._successes),
                failure_items=list(self._failures),
            )
        return summary

    @property
    def error_message(self) -> str | None:
        with self._lock:
            return self._error_message


class BatchJobManager:
    """Registry for active and historical batch jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, BatchJob] = {}
        self._lock = threading.Lock()

    def register(self, job: BatchJob) -> BatchJob:
        """Store ``job`` for later lookup and return it."""

        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> BatchJob | None:
        """Return the job matching ``job_id`` if it exists."""

        with self._lock:
            return self._jobs.get(job_id)

    def summary(self, job_id: str) -> BatchJobSummary | None:
        """Return the summary for ``job_id`` when available."""

        job = self.get(job_id)
        if job is None:
            return None
        return job.build_summary()


__all__ = [
    "ResultRecord",
    "ResultStore",
    "build_data_uri",
    "ensure_filename",
    "BatchEvent",
    "BatchJob",
    "BatchJobManager",
    "BatchJobSummary",
    "total_size",
]
