"""Mixins and utilities for drag-and-drop behaviour in the GUI."""
from __future__ import annotations

import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import ttkbootstrap as tb

try:
    from bgremover_core.paths import INPUT_DIR
except ImportError:  # pragma: no cover - allow running from source without package install
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from bgremover_core.paths import INPUT_DIR

LOGGER = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Drag-and-drop helpers
# ------------------------------------------------------------------
SUPPORTED_IMAGE_SUFFIXES: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
)


def _resolve_style_color(colors: Any, key: str, default: str) -> str:
    """Return ``key`` from a ttkbootstrap ``colors`` mapping with fallback."""

    getter = getattr(colors, "get", None)
    if callable(getter):
        try:
            value = getter(key)
        except TypeError:
            try:
                value = getter(key, default)
            except TypeError:  # pragma: no cover - defensive for exotic signatures
                value = None
        if value not in (None, ""):
            return str(value)
    if isinstance(colors, dict):
        value = colors.get(key, default)
        if value not in (None, ""):
            return str(value)
    return default


class DragAndDropMixin:
    """Mixin providing drag-and-drop behaviours for the GUI."""

    def _init_styles(self) -> None:
        """Initialise custom styles for drag-and-drop affordances."""

        style: tb.Style = self.style
        colors = getattr(style, "colors", {})
        border_color = _resolve_style_color(colors, "info", "#38bdf8")
        active_color = _resolve_style_color(colors, "primary", "#2563eb")
        background = _resolve_style_color(colors, "bg", "#ffffff")
        foreground = _resolve_style_color(colors, "body", "#111827")

        style.configure(
            "DropZone.TFrame",
            bordercolor=border_color,
            borderwidth=2,
            relief="ridge",
            background=background,
        )
        style.configure(
            "DropZoneActive.TFrame",
            bordercolor=active_color,
            borderwidth=3,
            relief="ridge",
            background=background,
        )
        style.configure(
            "DropZone.TLabel",
            foreground=foreground,
            font=("Helvetica", 12, "bold"),
            background=background,
        )
        style.configure(
            "DropZoneActive.TLabel",
            foreground=active_color,
            font=("Helvetica", 12, "bold"),
            background=background,
        )

    def _register_root_drop_target(self) -> None:
        """Initialise drop zone messaging for the active root window."""

        self._update_drop_zone_badge()

    def _set_drop_zone_active(self, active: bool) -> None:
        """Toggle the drop zone highlight to reflect drag state."""

        frame = getattr(self, "drop_zone_frame", None)
        label = getattr(self, "drop_zone_label", None)
        style_name = "DropZoneActive.TFrame" if active else "DropZone.TFrame"
        label_style = "DropZoneActive.TLabel" if active else "DropZone.TLabel"
        if frame is not None:
            try:
                frame.configure(style=style_name)
            except Exception:  # pragma: no cover - visual hint only
                pass
        if label is not None:
            try:
                label.configure(style=label_style)
            except Exception:  # pragma: no cover - visual hint only
                pass

    def _update_drop_zone_badge(self) -> None:
        """Refresh the instructional text shown inside the drop zone."""

        base = "Drop images here (PNG/JPG/WebP/BMP/TIFF) or click ‘Browse’"
        if not getattr(self, "_dnd_available", False):
            base += "\n(Install `tkinterdnd2` to enable drag-and-drop)"
        queued = len(self._drop_queue)
        if self._drop_active and self._current_drop_path is not None:
            queued += 1
        if queued:
            suffix = "s" if queued != 1 else ""
            base += f"\n{queued} file{suffix} queued"
        self.drop_zone_message_var.set(base)

    def _set_batch_drop_zone_active(self, active: bool) -> None:
        """Toggle the batch drop zone highlight based on drag state."""

        frame = getattr(self, "batch_drop_frame", None)
        label = getattr(self, "batch_drop_label", None)
        style_name = "DropZoneActive.TFrame" if active else "DropZone.TFrame"
        label_style = "DropZoneActive.TLabel" if active else "DropZone.TLabel"
        if frame is not None:
            try:
                frame.configure(style=style_name)
            except Exception:  # pragma: no cover - visual hint only
                pass
        if label is not None:
            try:
                label.configure(style=label_style)
            except Exception:  # pragma: no cover - visual hint only
                pass

    def _update_batch_drop_message(self) -> None:
        """Refresh instructional text displayed in the batch drop zone."""

        base = "Drop images or folders here to process (supports recursion)."
        if not getattr(self, "_dnd_available", False):
            base += "\n(Install `tkinterdnd2` to enable drag-and-drop)"
        if self._batch_staged_count:
            folder_name = self._batch_staged_dir.name if self._batch_staged_dir else "staging folder"
            plural = "s" if self._batch_staged_count != 1 else ""
            base += f"\n{self._batch_staged_count} file{plural} staged in {folder_name}"
        self.batch_drop_message_var.set(base)

    def _on_drop_enter(self, _event: Any) -> None:
        """Highlight the drop zone when files enter its bounds."""

        if not getattr(self, "_dnd_available", False):
            return
        self._drop_highlight_depth += 1
        self._set_drop_zone_active(True)

    def _on_drop_leave(self, _event: Any) -> None:
        """Remove drop zone highlight when drag leaves the widget."""

        if not getattr(self, "_dnd_available", False):
            return
        self._drop_highlight_depth = max(0, self._drop_highlight_depth - 1)
        if self._drop_highlight_depth == 0:
            self._set_drop_zone_active(False)

    def _on_batch_drop_enter(self, _event: Any) -> None:
        """Highlight the batch drop zone when dragged items enter."""

        if not getattr(self, "_dnd_available", False):
            return
        self._batch_drop_highlight_depth += 1
        self._set_batch_drop_zone_active(True)

    def _on_batch_drop_leave(self, _event: Any) -> None:
        """Remove the batch drop zone highlight after drag leave."""

        if not getattr(self, "_dnd_available", False):
            return
        self._batch_drop_highlight_depth = max(0, self._batch_drop_highlight_depth - 1)
        if self._batch_drop_highlight_depth == 0:
            self._set_batch_drop_zone_active(False)

    def _on_drop_files(self, event: Any) -> None:
        """Handle TkinterDnD drop events by queueing supported images."""

        self._drop_highlight_depth = 0
        self._set_drop_zone_active(False)
        data = getattr(event, "data", "")
        if not data:
            self._log("Drop ignored — no data received.")
            return

        try:
            paths = self._parse_dropped_files(str(data))
        except Exception as error:  # pragma: no cover - defensive parsing
            LOGGER.warning("Failed to parse dropped data: %s", error)
            self._log("Drop failed ✗ — Reason: unable to parse file list.", error=True)
            return

        valid_paths: list[Path] = []
        ignored = 0
        for candidate in paths:
            if not candidate.exists() or not candidate.is_file():
                ignored += 1
                continue
            if not self._is_supported_image(candidate):
                ignored += 1
                continue
            try:
                staged = self._ensure_in_input_dir(candidate)
            except Exception as staging_error:
                LOGGER.warning("Unable to stage dropped file %s: %s", candidate, staging_error)
                self._log(
                    f"Failed to queue {candidate.name} ✗ — Reason: {staging_error}",
                    error=True,
                )
                continue
            valid_paths.append(staged)

        if not valid_paths:
            self._log("No supported images found in drop.")
            self._update_drop_zone_badge()
            return

        if ignored:
            plural = "s" if ignored != 1 else ""
            self._log(f"Ignored {ignored} non-image file{plural} in drop.")

        self._enqueue_dropped_files(valid_paths)

    def _on_drop_batch(self, event: Any) -> None:
        """Handle drag-and-drop operations targeting the batch tab."""

        self._batch_drop_highlight_depth = 0
        self._set_batch_drop_zone_active(False)
        data = getattr(event, "data", "")
        if not data:
            self._log("Batch drop ignored — no data received.")
            self._update_batch_drop_message()
            return

        try:
            raw_paths = self._parse_dropped_files(str(data))
        except Exception as error:  # pragma: no cover - defensive parsing
            LOGGER.warning("Failed to parse batch drop data: %s", error)
            self._log("Batch drop failed ✗ — Reason: unable to parse file list.", error=True)
            self._update_batch_drop_message()
            return

        if not raw_paths:
            self._log("Batch drop ignored — no paths resolved.")
            self._update_batch_drop_message()
            return

        recursive = False
        recursive_var = getattr(self, "recursive_var", None)
        if recursive_var is not None:
            try:
                recursive = bool(recursive_var.get())
            except Exception:
                recursive = bool(self.settings.get("recursive", False))
        else:
            recursive = bool(self.settings.get("recursive", False))

        self._log("Scanning dropped items for batch processing…")
        files, unsupported, missing, duplicates = self._collect_batch_sources(raw_paths, recursive)

        if not files:
            reasons: list[str] = []
            if unsupported:
                suffix = "s" if unsupported != 1 else ""
                reasons.append(f"{unsupported} unsupported file{suffix}")
            if duplicates:
                suffix = "s" if duplicates != 1 else ""
                reasons.append(f"{duplicates} duplicate file{suffix}")
            if missing:
                suffix = "s" if missing != 1 else ""
                reasons.append(f"{missing} missing item{suffix}")
            detail = "; ".join(reasons) if reasons else "no supported images found"
            self._log(f"No supported images found in batch drop ({detail}).")
            self._update_batch_drop_message()
            return

        try:
            staging_dir, staged_paths = self._stage_batch_drop_files(files)
        except Exception as error:  # pragma: no cover - defensive filesystem handling
            LOGGER.warning("Unable to stage batch drop files: %s", error)
            self._log(f"Unable to prepare dropped files ✗ — Reason: {error}", error=True)
            self._update_batch_drop_message()
            return

        self._batch_staged_dir = staging_dir
        self._batch_staged_count = len(staged_paths)
        self.batch_input_var.set(str(staging_dir))
        self._update_batch_drop_message()
        self._reset_batch_progress()

        plural = "s" if len(staged_paths) != 1 else ""
        self._log(f"Staged {len(staged_paths)} image file{plural} for batch processing via drag-and-drop.")

        if unsupported:
            suffix = "s" if unsupported != 1 else ""
            self._log(f"Skipped {unsupported} unsupported file{suffix} during batch drop.")
        if duplicates:
            suffix = "s" if duplicates != 1 else ""
            self._log(f"Skipped {duplicates} duplicate file{suffix} during batch drop.")
        if missing:
            suffix = "s" if missing != 1 else ""
            self._log(f"Skipped {missing} missing item{suffix} during batch drop.")

    def _collect_batch_sources(
        self,
        items: list[Path],
        recursive: bool,
    ) -> tuple[list[Path], int, int, int]:
        """Return supported image files discovered from ``items``."""

        collected: list[Path] = []
        unsupported = 0
        missing = 0
        duplicates = 0
        seen: set[Path] = set()

        for item in items:
            if not item.exists():
                missing += 1
                continue
            if item.is_file():
                file_path = item
                if not self._is_supported_image(file_path):
                    unsupported += 1
                    continue
                try:
                    resolved = file_path.resolve(strict=False)
                except Exception:
                    resolved = file_path
                if resolved in seen:
                    duplicates += 1
                    continue
                seen.add(resolved)
                collected.append(resolved)
                continue

            if item.is_dir():
                for root, _, files in os.walk(item):
                    root_path = Path(root)
                    for name in files:
                        candidate = root_path / name
                        if not candidate.exists():
                            missing += 1
                            continue
                        if not self._is_supported_image(candidate):
                            unsupported += 1
                            continue
                        try:
                            resolved = candidate.resolve(strict=False)
                        except Exception:
                            resolved = candidate
                        if resolved in seen:
                            duplicates += 1
                            continue
                        seen.add(resolved)
                        collected.append(resolved)
                    if not recursive:
                        break
                    if hasattr(self, "update_idletasks"):
                        try:
                            self.update_idletasks()
                        except Exception:  # pragma: no cover - defensive UI pump
                            pass
                continue

            missing += 1

        return collected, unsupported, missing, duplicates

    def _stage_batch_drop_files(self, sources: list[Path]) -> tuple[Path, list[Path]]:
        """Copy ``sources`` into a dedicated batch staging directory."""

        input_root = INPUT_DIR.resolve()
        staging_root = input_root / "batch_drop"
        staging_root.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target_dir = staging_root / f"drop_{timestamp}"
        counter = 1
        while target_dir.exists():
            target_dir = staging_root / f"drop_{timestamp}_{counter}"
            counter += 1
        target_dir.mkdir(parents=True, exist_ok=False)

        staged: list[Path] = []
        for index, source in enumerate(sources, start=1):
            staged_path = self._copy_file_to_directory(source, target_dir)
            staged.append(staged_path)
            if index % 50 == 0 and hasattr(self, "update_idletasks"):
                try:
                    self.update_idletasks()
                except Exception:  # pragma: no cover - defensive UI pump
                    pass

        return target_dir, staged

    def _copy_file_to_directory(self, source: Path, destination_dir: Path) -> Path:
        """Copy ``source`` into ``destination_dir`` avoiding name collisions."""

        destination_dir.mkdir(parents=True, exist_ok=True)
        source_resolved = source.expanduser()
        try:
            source_resolved = source_resolved.resolve(strict=True)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Dropped file not found: {source}") from error

        destination = destination_dir / source_resolved.name
        stem = destination.stem
        suffix = destination.suffix
        counter = 1
        while destination.exists():
            try:
                if destination.samefile(source_resolved):
                    return destination.resolve(strict=True)
            except Exception:
                pass
            destination = destination_dir / f"{stem}-{counter}{suffix}"
            counter += 1

        shutil.copy2(source_resolved, destination)
        return destination.resolve(strict=True)

    def _enqueue_dropped_files(self, files: list[Path]) -> None:
        """Add ``files`` to the background processing queue."""

        newly_added: list[Path] = []
        for file_path in files:
            try:
                resolved = file_path.resolve(strict=False)
            except Exception:
                resolved = file_path
            if resolved in self._drop_seen:
                continue
            self._drop_queue.append(resolved)
            self._drop_seen.add(resolved)
            newly_added.append(resolved)

        if not newly_added:
            self._log("Dropped images were already queued.")
            self._update_drop_zone_badge()
            return

        if len(newly_added) == 1:
            self._log(f"Queued {newly_added[0].name} for processing via drag-and-drop.")
        else:
            self._log(f"Queued {len(newly_added)} images for processing via drag-and-drop.")

        self._update_drop_zone_badge()
        self.after(150, self._process_next_dropped_file)

    def _process_next_dropped_file(self) -> None:
        """Start processing the next queued file when the UI is idle."""

        if self._drop_active or not self._drop_queue:
            return
        if self._processing_context is not None:
            self.after(300, self._process_next_dropped_file)
            return

        next_path = self._drop_queue.pop(0)
        self._drop_active = True
        self._current_drop_path = next_path
        self.single_input_var.set(str(next_path))
        self._update_drop_zone_badge()
        try:
            self._process_single()
        except Exception as error:  # pragma: no cover - defensive
            LOGGER.exception("Failed to start processing for %s", next_path)
            self._log(f"Failed to start processing {next_path.name} ✗ — Reason: {error}", error=True)
            self._drop_active = False
            self._current_drop_path = None
            self._update_drop_zone_badge()
            self.after(300, self._process_next_dropped_file)

    def _parse_dropped_files(self, data: str) -> list[Path]:
        """Return file system paths parsed from a TkinterDnD payload."""

        if not data:
            return []
        try:
            items = self.tk.splitlist(data)
        except Exception:
            items = data.split()

        paths: list[Path] = []
        for item in items:
            text = item.strip()
            if not text:
                continue
            candidate: Path
            if text.startswith("file://"):
                parsed = urlparse(text)
                path_part = unquote(parsed.path or "")
                if parsed.netloc:
                    path_part = f"//{parsed.netloc}{path_part}"
                candidate = Path(url2pathname(path_part))
            else:
                candidate = Path(unquote(text))
            candidate = candidate.expanduser()
            try:
                candidate = candidate.resolve(strict=False)
            except Exception:
                pass
            paths.append(candidate)
        return paths

    def _is_supported_image(self, path: Path) -> bool:
        """Return ``True`` when ``path`` has a supported image suffix."""

        return path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES

    def _ensure_in_input_dir(self, path: Path) -> Path:
        """Stage ``path`` inside the project input directory if required."""

        source = path.expanduser()
        try:
            source_resolved = source.resolve(strict=True)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"Dropped file not found: {source}") from error

        input_root = INPUT_DIR
        input_root.mkdir(parents=True, exist_ok=True)
        input_root_resolved = input_root.resolve()
        try:
            source_resolved.relative_to(input_root_resolved)
            return source_resolved
        except ValueError:
            pass

        destination = input_root_resolved / source_resolved.name
        stem = destination.stem
        suffix = destination.suffix
        counter = 1
        while destination.exists():
            try:
                if destination.samefile(source_resolved):
                    return destination.resolve(strict=True)
            except Exception:
                pass
            destination = input_root_resolved / f"{stem}-{counter}{suffix}"
            counter += 1

        shutil.copy2(source_resolved, destination)
        return destination.resolve(strict=True)

    def _on_single_run_complete(self, _input_path: Path) -> None:
        """Clear drag-and-drop state after a single image run finishes."""

        if self._drop_active:
            self._drop_active = False
            self._current_drop_path = None
            self._update_drop_zone_badge()
            if self._drop_queue:
                self.after(250, self._process_next_dropped_file)
        else:
            self._update_drop_zone_badge()
