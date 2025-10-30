"""Tkinter GUI for the background remover runtimes."""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import ttkbootstrap as tb
from PIL import Image, ImageDraw, ImageTk
from ttkbootstrap.constants import BOTH, END, LEFT, RIGHT, W
from ttkbootstrap.scrolled import ScrolledText
from ttkbootstrap.tooltip import ToolTip

try:
    import tkinter as tk  # noqa: I001
    from tkinterdnd2 import DND_FILES, TkinterDnD  # noqa: I001

    _DND_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    import tkinter as tk

    _DND_AVAILABLE = False
    DND_FILES = None
    TkinterDnD = None

from tkinter import Canvas, filedialog, messagebox

try:
    from bgremover_core import Config, init_logging, load_config, persist_config, process_folder
    from bgremover_core.background_remover import process_image
    from bgremover_core.io.image_io import image_to_numpy, save_image_to_path
    from bgremover_core.models.loader import detect_providers
    from bgremover_core.models.specs import MODEL_SPECS
    from bgremover_core.paths import CONFIG_FILE, INPUT_DIR, MODELS_DIR, OUTPUT_DIR
    from bgremover_core.processing.pipeline import ProcessingResult, ReportEntry
except ImportError:  # pragma: no cover - allow running from source without package install
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from bgremover_core import Config, init_logging, load_config, persist_config, process_folder
    from bgremover_core.background_remover import process_image
    from bgremover_core.io.image_io import image_to_numpy, save_image_to_path
    from bgremover_core.models.loader import detect_providers
    from bgremover_core.models.specs import MODEL_SPECS
    from bgremover_core.paths import CONFIG_FILE, INPUT_DIR, MODELS_DIR, OUTPUT_DIR
    from bgremover_core.processing.pipeline import ProcessingResult, ReportEntry

LOGGER = logging.getLogger(__name__)

GUI_SETTINGS_FILE = CONFIG_FILE

DEFAULT_SETTINGS: dict[str, Any] = {
    "model_key": "isnet-general-use",
    "input_resize": "stretch",
    "alpha_matting": False,
    "alpha_foreground_threshold": 240,
    "alpha_background_threshold": 10,
    "alpha_erode_size": 10,
    "smoothing": 0.0,
    "edge_refinement": False,
    "feather_radius": 3,
    "output_format": "PNG",
    "preserve_names": False,
    "output_directory": "",
    "device": "Auto",
    "recursive": False,
    "parallel_threads": 4,
    "model_dir": str(MODELS_DIR),
    "theme": "flatly",
}


VALID_RESIZE_MODES: tuple[str, ...] = ("auto", "keep-aspect", "crop", "stretch")

PREVIEW_ZOOM_MIN = 25.0
PREVIEW_ZOOM_MAX = 400.0
PREVIEW_ZOOM_DEFAULT = 100.0
PREVIEW_ZOOM_WHEEL_STEP = 10.0

# ``PreviewAnchor`` tracks a canvas coordinate and the corresponding widget
# pointer location to keep stable while zooming.
type PreviewAnchor = tuple[float, float, float, float]


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
def _format_meta(format_name: str) -> tuple[str, str]:
    """Return the Pillow format hint and file suffix for ``format_name``."""

    match format_name.upper():
        case "JPEG" | "JPG":
            return "JPEG", "jpg"
        case "WEBP":
            return "WEBP", "webp"
        case _:
            return "PNG", "png"


def _hex_to_rgb(value: str | None) -> tuple[int, int, int] | None:
    """Return ``value`` interpreted as a RGB tuple when valid."""

    if not value:
        return None
    text = value.strip().lstrip("#")
    if len(text) != 6:
        return None
    try:
        return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))
    except ValueError:  # pragma: no cover - defensive
        return None


def _resolve_resize_mode(value: str | None) -> str:
    """Return a supported resize mode string defaulting to ``"stretch"``."""

    if not value:
        return "stretch"
    lowered = value.strip().lower()
    for mode in VALID_RESIZE_MODES:
        if lowered == mode:
            return mode
    LOGGER.warning("Unknown resize mode %s; falling back to 'stretch'", value)
    return "stretch"


def _coerce_smoothing(value: Any) -> float:
    """Return a clamped smoothing ratio compatible with the processing pipeline."""

    try:
        smoothing = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, smoothing))


def run_gui_pipeline_for_parity(
    image: Image.Image,
    *,
    config: Config,
    model_key: str,
    feather_radius: int = 3,
    **advanced_options: object,
) -> ProcessingResult:
    """Return the GUI pipeline result for diagnostic parity checks."""

    array = image_to_numpy(image)
    return process_image(
        array,
        model_key=model_key,
        config=config,
        feather_radius=feather_radius,
        **advanced_options,
    )


class CollapsibleSection(tb.Frame):
    """A reusable frame with a toggleable content area."""

    def __init__(
        self,
        parent: Any,
        *,
        title: str = "",
        start_open: bool = True,
    ) -> None:
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self._title = title
        self.content_visible = start_open

        self.header = tb.Frame(self)
        self.header.grid(row=0, column=0, sticky="ew")

        arrow = "▼" if start_open else "►"
        self.toggle_button = tb.Button(
            self.header,
            text=f"{arrow} {title}",
            command=self.toggle,
        )

        self.toggle_button.configure(style="TButton", padding=(5, 2))
        self.toggle_button.pack(fill="x", anchor="w")
        self.content = tb.Frame(self)
        if start_open:
            self.content.grid(row=1, column=0, sticky="ew")

    def toggle(self) -> None:
        """Collapse or expand the content frame."""

        if self.content_visible:
            self.content.grid_remove()
            self.toggle_button.configure(text=f"► {self._title}")
        else:
            self.content.grid(row=1, column=0, sticky="ew")
            self.toggle_button.configure(text=f"▼ {self._title}")
        self.content_visible = not self.content_visible


if _DND_AVAILABLE:

    class _TkRoot(TkinterDnD.Tk):
        """Tk root window with TkinterDnD2 drag-and-drop support."""


else:

    class _TkRoot(tk.Tk):
        """Standard Tk root window when TkinterDnD2 is unavailable."""


def _initialize_background_remover_app(app: BackgroundRemoverApp) -> None:
    """Configure the background remover GUI on an initialised Tk root."""

    theme = DEFAULT_SETTINGS.get("theme", "flatly")
    try:
        if GUI_SETTINGS_FILE.exists():
            user_settings = json.loads(GUI_SETTINGS_FILE.read_text(encoding="utf-8"))
            theme = user_settings.get("theme", theme)
    except Exception:  # pragma: no cover - defensive fallback
        pass

    app.style = tb.Style(theme=theme)
    app.themename = theme
    app.title("Background Remover - PRO")
    app.geometry("1920x1080")
    app.resizable(True, True)

    app.drop_zone_message_var = tb.StringVar()
    app.batch_drop_message_var = tb.StringVar()
    app._drop_queue: list[Path] = []
    app._drop_seen: set[Path] = set()
    app._drop_active = False
    app._current_drop_path: Path | None = None
    app._drop_highlight_depth = 0
    app._batch_drop_highlight_depth = 0
    app._processing_context: str | None = None
    app.drop_zone_frame: tb.Frame | None = None
    app.drop_zone_label: tb.Label | None = None
    app.batch_drop_frame: tb.Frame | None = None
    app.batch_drop_label: tb.Label | None = None
    app._batch_staged_dir: Path | None = None
    app._batch_staged_count = 0

    app._init_styles()

    app.config = load_config()
    init_logging(app.config.log_level)
    app.settings = app._load_settings(app.config)
    app.settings.setdefault("theme", app.themename)
    app._tooltips: dict[object, ToolTip] = {}

    icon_path = Path(os.path.dirname(__file__)) / "bg_icon.ico"
    logging.info("Attempting to load window icon from: %s", icon_path)

    try:
        if icon_path.exists():
            app.iconbitmap(icon_path)
            logging.info("Successfully applied .ico icon to GUI window.")
        else:
            logging.warning("Icon file not found: %s", icon_path)
    except Exception as error:  # pragma: no cover - icon best-effort
        logging.exception("Failed to set .ico icon: %s", error)
        try:
            from tkinter import PhotoImage

            png_icon = icon_path.with_suffix(".png")
            if png_icon.exists():
                app.iconphoto(False, PhotoImage(file=str(png_icon)))
                logging.info("Fallback: applied .png icon successfully.")
            else:
                logging.warning("No fallback PNG found at %s", png_icon)
        except Exception as png_error:  # pragma: no cover - optional path
            logging.exception("Failed to set .png fallback icon: %s", png_error)

    app.providers = detect_providers(app._provider_hints())
    app._preview_image: Image.Image | None = None
    app._preview_photo: ImageTk.PhotoImage | None = None
    app._preview_output_path: Path | None = None
    app._preview_format_hint: str | None = None
    app._preview_original_name: str | None = None
    app._preview_saved_path: Path | None = None
    app._preview_canvas_image: int | None = None
    app._preview_display_override: Image.Image | None = None
    app._preview_last_fill_color: tuple[int, int, int] | None = None
    app._preview_fill_color: tuple[int, int, int] | None = None
    app.preview_zoom_var = tb.DoubleVar(value=PREVIEW_ZOOM_DEFAULT)
    app._preview_zoom_manual_override = False
    app._preview_zoom_updating = False
    app._preview_render_size: tuple[int, int] = (0, 0)
    app._preview_canvas_size: tuple[int, int] = (0, 0)
    app._preview_canvas_hover = False

    app._build_ui()
    app._register_root_drop_target()
    app._clear_preview_state()
    app._refresh_badge()


class BackgroundRemoverApp(_TkRoot):
    """Main application window for background removal."""

    def __init__(self) -> None:
        super().__init__()
        _initialize_background_remover_app(self)

    # ------------------------------------------------------------------
    # Drag-and-drop helpers
    # ------------------------------------------------------------------
    def _init_styles(self) -> None:
        """Initialise custom styles for drag-and-drop affordances."""

        style = self.style
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
        if not _DND_AVAILABLE:
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
        if not _DND_AVAILABLE:
            base += "\n(Install `tkinterdnd2` to enable drag-and-drop)"
        if self._batch_staged_count:
            folder_name = self._batch_staged_dir.name if self._batch_staged_dir else "staging folder"
            plural = "s" if self._batch_staged_count != 1 else ""
            base += f"\n{self._batch_staged_count} file{plural} staged in {folder_name}"
        self.batch_drop_message_var.set(base)

    def _on_drop_enter(self, _event: Any) -> None:
        """Highlight the drop zone when files enter its bounds."""

        if not _DND_AVAILABLE:
            return
        self._drop_highlight_depth += 1
        self._set_drop_zone_active(True)

    def _on_drop_leave(self, _event: Any) -> None:
        """Remove drop zone highlight when drag leaves the widget."""

        if not _DND_AVAILABLE:
            return
        self._drop_highlight_depth = max(0, self._drop_highlight_depth - 1)
        if self._drop_highlight_depth == 0:
            self._set_drop_zone_active(False)

    def _on_batch_drop_enter(self, _event: Any) -> None:
        """Highlight the batch drop zone when dragged items enter."""

        if not _DND_AVAILABLE:
            return
        self._batch_drop_highlight_depth += 1
        self._set_batch_drop_zone_active(True)

    def _on_batch_drop_leave(self, _event: Any) -> None:
        """Remove the batch drop zone highlight after drag leave."""

        if not _DND_AVAILABLE:
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
        """Return supported image files discovered from ``items``.

        The returned tuple contains ``(files, unsupported_count, missing_count,
        duplicate_count)``.
        """

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
            # Re-check shortly once the current task finishes.
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
                # If the file exists but cannot be compared, keep searching.
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

    # Settings helpers
    # ------------------------------------------------------------------
    def _load_settings(self, base_config: Config) -> dict[str, Any]:
        """Load persisted GUI settings merged with ``base_config`` defaults."""

        settings = DEFAULT_SETTINGS.copy()
        try:
            if GUI_SETTINGS_FILE.exists():
                raw = json.loads(GUI_SETTINGS_FILE.read_text(encoding="utf-8"))
                settings.update(raw)
        except Exception as error:  # pragma: no cover - defensive
            LOGGER.warning("Unable to read GUI settings: %s", error)
        settings.setdefault("model_key", base_config.default_model)
        settings.setdefault("model_dir", str(base_config.model_dir))
        settings["input_resize"] = _resolve_resize_mode(settings.get("input_resize"))
        settings["smoothing"] = _coerce_smoothing(settings.get("smoothing", 0.0))
        return settings

    def _save_settings(self) -> None:
        """Persist current GUI settings to disk."""

        try:
            GUI_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, Any] = {}
            if GUI_SETTINGS_FILE.exists():
                try:
                    payload = json.loads(GUI_SETTINGS_FILE.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    payload = {}
            payload.update(self.settings)
            GUI_SETTINGS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as error:  # pragma: no cover - defensive
            LOGGER.warning("Failed to persist GUI settings: %s", error)

    def _update_setting(self, key: str, value: Any, *, persist: bool = True) -> None:
        """Update ``key`` inside :attr:`settings` and optionally persist."""

        self.settings[key] = value
        if persist:
            self._save_settings()

    def _provider_hints(self) -> tuple[str, ...]:
        """Return provider hints derived from current settings."""

        settings = getattr(self, "__dict__", {}).get("settings") or {}
        device = (settings.get("device") or "Auto").lower()
        if device == "gpu":
            return ("CUDAExecutionProvider", "CPUExecutionProvider")
        if device == "cpu":
            return ("CPUExecutionProvider",)
        return self.config.provider_hints

    def _active_config(self) -> Config:
        """Return a :class:`Config` reflecting interactive selections."""

        updates: dict[str, Any] = {}
        settings = getattr(self, "__dict__", {}).get("settings") or {}
        model_dir_text = (settings.get("model_dir") or "").strip()
        if not model_dir_text:
            model_dir_var = getattr(self, "__dict__", {}).get("model_dir_var")
            if model_dir_var is not None:
                try:
                    model_dir_text = (model_dir_var.get() or "").strip()
                except Exception:  # pragma: no cover - safeguard for mocked widgets
                    model_dir_text = ""
        if model_dir_text:
            updates["model_dir"] = Path(model_dir_text)
        provider_hints = self._provider_hints()
        if provider_hints != self.config.provider_hints:
            updates["provider_hints"] = provider_hints
        return self.config.with_updates(**updates) if updates else self.config

    def _processing_kwargs(self) -> dict[str, Any]:
        """Return advanced processing keyword arguments."""

        return {
            "resize_mode": _resolve_resize_mode(self.settings.get("input_resize")),
            "alpha_matting": bool(self.settings.get("alpha_matting", False)),
            "alpha_foreground_threshold": self.settings.get("alpha_foreground_threshold", 240),
            "alpha_background_threshold": self.settings.get("alpha_background_threshold", 10),
            "alpha_erode_size": self.settings.get("alpha_erode_size", 10),
            "smoothing": _coerce_smoothing(self.settings.get("smoothing", 0.0)),
            "edge_refinement": bool(self.settings.get("edge_refinement", False)),
            "output_format": self.settings.get("output_format", "PNG"),
            "preserve_names": bool(self.settings.get("preserve_names", False)),
            "parallel_threads": int(self.settings.get("parallel_threads", 1)),
        }

    # ------------------------------------------------------------------
    # UI builders
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        """Construct the main UI layout."""

        container = tb.Frame(self, padding=20)
        container.pack(fill=BOTH, expand=True)

        top_frame = tb.Frame(container)
        top_frame.pack(fill=BOTH, expand=True)

        control_frame = tb.Frame(top_frame)
        control_frame.pack(side=LEFT, fill=BOTH, expand=True)

        header = tb.Frame(control_frame)
        header.pack(fill=BOTH, expand=False)

        tb.Label(header, text="Background Remover", font=("Helvetica", 20, "bold")).pack(side=LEFT)

        self.badge = tb.Label(header, padding=(10, 4))
        self.badge.pack(side=RIGHT)
        self.theme_toggle_btn = tb.Button(
            header,
            text="🌙" if self.themename == "flatly" else "☀️",
            command=self._toggle_theme,
            width=3,
        )
        self.theme_toggle_btn.pack(side=RIGHT, padx=(0, 10))
        self._add_tooltip(self.theme_toggle_btn, "Toggle dark/light mode")
        self._add_tooltip(self.badge, "Providers: detecting…")

        notebook = tb.Notebook(control_frame, bootstyle="tabs")
        notebook.pack(fill=BOTH, expand=True, pady=(20, 10))

        self.single_tab = tb.Frame(notebook, padding=10)
        notebook.add(self.single_tab, text="Single Image")
        self._build_single_tab(self.single_tab)

        self.batch_tab = tb.Frame(notebook, padding=10)
        notebook.add(self.batch_tab, text="Batch Folder")
        self._build_batch_tab(self.batch_tab)

        advanced_frame = tb.Frame(control_frame)
        advanced_frame.pack(fill=BOTH, expand=False, pady=(0, 10))
        self._build_advanced_panel(advanced_frame)

        preview_frame = tb.Labelframe(top_frame, text="Preview", padding=10)
        preview_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=(12, 0))
        preview_frame.rowconfigure(2, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        toolbar = tb.Frame(preview_frame)
        toolbar.grid(row=0, column=0, columnspan=3, sticky="we", pady=(0, 4))
        self.preview_fill_button = tb.Button(
            toolbar,
            text="Background Fill",
            command=self._preview_fill_bg,
            state="disabled",
        )
        self.preview_fill_button.pack(side=LEFT, padx=(0, 6))

        self.preview_clear_button = tb.Button(
            toolbar,
            text="Background Clear",
            command=self._preview_clear_bg,
            state="disabled",
        )
        self.preview_clear_button.pack(side=LEFT)

        self.preview_info = tb.Label(preview_frame, text="No preview available yet.", anchor="w")
        self.preview_info.grid(row=1, column=0, columnspan=3, sticky="we")

        canvas_container = tb.Frame(preview_frame)
        canvas_container.grid(row=2, column=0, columnspan=3, sticky="nsew", pady=(8, 8))
        canvas_container.rowconfigure(0, weight=1)
        canvas_container.columnconfigure(0, weight=1)

        self.preview_canvas = Canvas(canvas_container, highlightthickness=0, background="#111827")
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        self.preview_canvas.bind("<Configure>", self._on_preview_canvas_resize)
        self.preview_canvas.bind("<Enter>", self._on_preview_canvas_enter)
        self.preview_canvas.bind("<Leave>", self._on_preview_canvas_leave)
        self.preview_canvas.bind("<MouseWheel>", self._on_preview_mouse_wheel)
        self.preview_canvas.bind("<Button-4>", self._on_preview_mouse_wheel)
        self.preview_canvas.bind("<Button-5>", self._on_preview_mouse_wheel)
        self.preview_canvas.bind("<ButtonPress-1>", self._on_preview_drag_start)
        self.preview_canvas.bind("<B1-Motion>", self._on_preview_drag_motion)

        self.preview_overlay_frame = tb.Frame(canvas_container, bootstyle="dark")
        self.preview_overlay_label = tb.Label(
            self.preview_overlay_frame,
            text="Processing…",
            anchor="center",
            font=("Helvetica", 14, "bold"),
            bootstyle="light",
            padding=20,
        )
        self.preview_overlay_label.pack(expand=True, fill=BOTH)
        self.preview_overlay_frame.place_forget()

        self.preview_scroll_y = tb.Scrollbar(
            canvas_container,
            orient="vertical",
            command=self.preview_canvas.yview,
        )
        self.preview_scroll_y.grid(row=0, column=1, sticky="ns")

        self.preview_scroll_x = tb.Scrollbar(
            preview_frame,
            orient="horizontal",
            command=self.preview_canvas.xview,
        )
        self.preview_scroll_x.grid(row=3, column=0, columnspan=3, sticky="we")

        self.preview_canvas.configure(
            xscrollcommand=self.preview_scroll_x.set,
            yscrollcommand=self.preview_scroll_y.set,
        )

        zoom_controls = tb.Frame(preview_frame)
        zoom_controls.grid(row=4, column=0, columnspan=3, sticky="we")
        tb.Label(zoom_controls, text="Zoom").pack(side=LEFT)
        self.preview_zoom_slider = tb.Scale(
            zoom_controls,
            from_=int(PREVIEW_ZOOM_MIN),
            to=int(PREVIEW_ZOOM_MAX),
            orient="horizontal",
            variable=self.preview_zoom_var,
            command=lambda _: self._on_preview_zoom(),
        )
        self.preview_zoom_slider.pack(side=LEFT, fill=BOTH, expand=True, padx=6)
        self.preview_zoom_value = tb.Label(
            zoom_controls, text=f"{int(PREVIEW_ZOOM_DEFAULT)}%", width=6
        )
        self.preview_zoom_value.pack(side=LEFT)

        action_frame = tb.Frame(preview_frame)
        action_frame.grid(row=5, column=0, columnspan=3, sticky="we", pady=(8, 0))
        self.save_button = tb.Button(
            action_frame,
            text="Save",
            bootstyle="success",
            command=self._on_preview_save,
            state="disabled",
        )
        self.save_button.pack(side=LEFT, padx=(0, 6))
        self.discard_button = tb.Button(
            action_frame,
            text="Discard",
            bootstyle="secondary",
            command=self._on_preview_discard,
            state="disabled",
        )
        self.discard_button.pack(side=LEFT, padx=(0, 6))
        self.view_full_button = tb.Button(
            action_frame,
            text="View Full",
            bootstyle="info",
            command=self._view_saved_preview,
            state="disabled",
        )
        self.view_full_button.pack(side=LEFT)

        log_frame = tb.Labelframe(container, text="Activity Log", padding=10)
        log_frame.pack(fill=BOTH, expand=True, pady=(12, 0))
        self.log_widget = ScrolledText(log_frame, height=10)
        self.log_widget.pack(fill=BOTH, expand=True)
        self.log_widget.tag_config("error", foreground="#b91c1c")

    def _build_single_tab(self, parent: tb.Frame) -> None:
        """Create widgets for single image processing."""

        input_frame = tb.Frame(parent)
        input_frame.pack(fill=BOTH, expand=False, pady=5)

        tb.Label(input_frame, text="Input image").pack(anchor="w")
        control = tb.Frame(input_frame)
        control.pack(fill=BOTH, expand=False)
        self.single_input_var = tb.StringVar(value="")
        tb.Entry(control, textvariable=self.single_input_var, width=60).pack(side=LEFT, padx=(0, 8))
        tb.Button(control, text="Browse", command=self._choose_single_file).pack(side=LEFT)

        output_frame = tb.Frame(parent)
        output_frame.pack(fill=BOTH, expand=False, pady=5)
        tb.Label(output_frame, text="Output file (optional)").pack(anchor="w")
        self.single_output_var = tb.StringVar(value="")
        tb.Entry(output_frame, textvariable=self.single_output_var, width=60).pack(side=LEFT, padx=(0, 8))
        tb.Button(output_frame, text="Browse", command=self._choose_single_output).pack(side=LEFT)

        self.single_process_button = tb.Button(
            parent,
            text="Process Image",
            bootstyle="primary",
            command=self._process_single,
        )
        self.single_process_button.pack(pady=(10, 0))

        self.single_spinner = tb.Progressbar(parent, mode="indeterminate", length=220)
        self.single_spinner.pack(fill="x", pady=(6, 0))
        self.single_spinner.stop()
        self.single_spinner.pack_forget()

        drop_zone = tb.Frame(parent, padding=16, style="DropZone.TFrame")
        drop_zone.pack(fill=BOTH, expand=True, pady=(15, 5))
        drop_zone.columnconfigure(0, weight=1)
        drop_zone.rowconfigure(0, weight=1)

        drop_label = tb.Label(
            drop_zone,
            textvariable=self.drop_zone_message_var,
            style="DropZone.TLabel",
            anchor="center",
            justify="center",
            wraplength=480,
        )
        drop_label.grid(row=0, column=0, sticky="nsew")

        self.drop_zone_frame = drop_zone
        self.drop_zone_label = drop_label
        self._set_drop_zone_active(False)

        if _DND_AVAILABLE:
            try:
                drop_zone.drop_target_register(DND_FILES)
                drop_zone.dnd_bind("<<Drop>>", self._on_drop_files)
                drop_zone.dnd_bind("<<DragEnter>>", self._on_drop_enter)
                drop_zone.dnd_bind("<<DragLeave>>", self._on_drop_leave)
            except Exception as error:  # pragma: no cover - optional path
                LOGGER.warning("Unable to enable drag-and-drop on drop zone: %s", error)

        self._update_drop_zone_badge()

    def _build_batch_tab(self, parent: tb.Frame) -> None:
        """Create widgets for batch folder processing."""

        input_frame = tb.Frame(parent)
        input_frame.pack(fill=BOTH, expand=False, pady=5)
        tb.Label(input_frame, text="Input folder").pack(anchor="w")
        control = tb.Frame(input_frame)
        control.pack(fill=BOTH, expand=False)
        self.batch_input_var = tb.StringVar(value="")
        tb.Entry(control, textvariable=self.batch_input_var, width=60).pack(side=LEFT, padx=(0, 8))
        tb.Button(control, text="Browse", command=self._choose_batch_folder).pack(side=LEFT)

        output_frame = tb.Frame(parent)
        output_frame.pack(fill=BOTH, expand=False, pady=5)
        tb.Label(output_frame, text="Output folder (optional)").pack(anchor="w")
        self.batch_output_var = tb.StringVar(value="")
        tb.Entry(output_frame, textvariable=self.batch_output_var, width=60).pack(side=LEFT, padx=(0, 8))
        tb.Button(output_frame, text="Browse", command=self._choose_batch_output).pack(side=LEFT)

        drop_zone = tb.Frame(parent, padding=16, style="DropZone.TFrame")
        drop_zone.pack(fill=BOTH, expand=True, pady=(15, 5))
        drop_zone.columnconfigure(0, weight=1)
        drop_zone.rowconfigure(0, weight=1)

        drop_label = tb.Label(
            drop_zone,
            textvariable=self.batch_drop_message_var,
            style="DropZone.TLabel",
            anchor="center",
            justify="center",
            wraplength=480,
        )
        drop_label.grid(row=0, column=0, sticky="nsew")

        self.batch_drop_frame = drop_zone
        self.batch_drop_label = drop_label
        self._set_batch_drop_zone_active(False)

        if _DND_AVAILABLE:
            try:
                drop_zone.drop_target_register(DND_FILES)
                drop_zone.dnd_bind("<<Drop>>", self._on_drop_batch)
                drop_zone.dnd_bind("<<DragEnter>>", self._on_batch_drop_enter)
                drop_zone.dnd_bind("<<DragLeave>>", self._on_batch_drop_leave)
            except Exception as error:  # pragma: no cover - optional path
                LOGGER.warning("Unable to enable drag-and-drop on batch drop zone: %s", error)

        self._update_batch_drop_message()

        self.batch_process_button = tb.Button(
            parent,
            text="Process Folder",
            bootstyle="primary",
            command=self._process_batch,
        )
        self.batch_process_button.pack(pady=(10, 10))

        self.batch_spinner = tb.Progressbar(parent, mode="indeterminate", length=220)
        self.batch_spinner.pack(fill="x", pady=(0, 10))
        self.batch_spinner.stop()
        self.batch_spinner.pack_forget()

        progress_frame = tb.Labelframe(parent, text="Batch Progress", padding=6)
        progress_frame.pack(fill=BOTH, expand=True)
        columns = ("file", "status", "details")
        self.batch_tree = tb.Treeview(progress_frame, columns=columns, show="headings", height=6)
        self.batch_tree.heading("file", text="File")
        self.batch_tree.heading("status", text="Status")
        self.batch_tree.heading("details", text="Details")
        self.batch_tree.column("file", width=160, anchor=W)
        self.batch_tree.column("status", width=70, anchor=W)
        self.batch_tree.column("details", anchor=W)
        self.batch_tree.pack(fill=BOTH, expand=True)
        self._add_tooltip(
            self.batch_tree,
            "Shows progress and results for each processed file.",
        )

    def _build_advanced_panel(self, parent: tb.Frame) -> None:
        """Create the collapsible advanced settings panel."""

        header = tb.Frame(parent)
        header.pack(fill=BOTH, expand=False)
        tb.Label(header, text="Advanced Settings", font=("Helvetica", 16, "bold")).pack(side=LEFT)
        toggle_all = tb.Button(
            header,
            text="Expand All",
            command=self._toggle_all_sections,
        )
        toggle_all.pack(side=RIGHT, padx=(0, 8))
        self.toggle_all_button = toggle_all
        self._sections_expanded = False

        self.advanced_visible = tb.BooleanVar(value=True)
        self.toggle_button = tb.Button(header, text="Hide", command=self._toggle_advanced)
        self.toggle_button.pack(side=RIGHT)

        self.advanced_body = tb.Frame(parent, padding=10)
        self.advanced_body.pack(fill=BOTH, expand=True)

        self._build_general_section(self.advanced_body)

        alpha_section = CollapsibleSection(
            self.advanced_body,
            title="Alpha Matting Refinement",
            start_open=False,
        )
        alpha_section.pack(fill="x", pady=(0, 8))
        self._build_alpha_section(alpha_section.content)

        mask_section = CollapsibleSection(
            self.advanced_body,
            title="Mask Refinement",
            start_open=False,
        )
        mask_section.pack(fill="x", pady=(0, 8))
        self._build_mask_section(mask_section.content)

        output_section = CollapsibleSection(
            self.advanced_body,
            title="Output",
            start_open=False,
        )
        output_section.pack(fill="x", pady=(0, 8))
        self._build_output_section(output_section.content)

        batch_section = CollapsibleSection(
            self.advanced_body,
            title="Batch Processing",
            start_open=False,
        )
        batch_section.pack(fill="x")
        self._build_batch_section(batch_section.content)
        self._toggle_alpha_controls()

    def _toggle_all_sections(self) -> None:
        """Expand or collapse every collapsible advanced settings section."""

        expand = not getattr(self, "_sections_expanded", False)
        for child in self.advanced_body.winfo_children():
            if isinstance(child, CollapsibleSection):
                if expand and not child.content_visible:
                    child.toggle()
                elif not expand and child.content_visible:
                    child.toggle()
        self._sections_expanded = expand
        if self.toggle_all_button:
            self.toggle_all_button.configure(
                text="Collapse All" if self._sections_expanded else "Expand All"
            )

    def _build_general_section(self, parent: tb.Frame) -> None:
        """Create general processing preference controls."""

        general = tb.Frame(parent)
        general.pack(fill="x", pady=(0, 10))
        general.columnconfigure(1, weight=1)

        tb.Label(general, text="Model").grid(row=0, column=0, sticky=W)
        self.model_var = tb.StringVar(value=self.settings.get("model_key", self.config.default_model))
        model_combo = tb.Combobox(
            general,
            values=sorted(MODEL_SPECS.keys()),
            textvariable=self.model_var,
            width=40,
            state="readonly",
        )
        model_combo.grid(row=0, column=1, sticky="we", padx=8)
        self._add_tooltip(
            model_combo,
            "Select the model used for background removal. Larger models are slower but more accurate.",
        )
        self.model_var.trace_add("write", lambda *_: self._on_model_change())

        tb.Label(general, text="Input resize").grid(row=1, column=0, sticky=W)
        self.resize_var = tb.StringVar(value=self.settings.get("input_resize", "stretch"))
        resize_combo = tb.Combobox(
            general,
            values=list(VALID_RESIZE_MODES),
            textvariable=self.resize_var,
            width=40,
            state="readonly",
        )
        resize_combo.grid(row=1, column=1, sticky="we", padx=8)
        self.resize_var.trace_add(
            "write",
            lambda *_: self._update_setting("input_resize", self.resize_var.get()),
        )
        self._add_tooltip(
            resize_combo,
            (
                "Controls how input images are resized before inference. 'Stretch' matches "
                "CLI and Flask results; other modes preserve composition differently."
            ),
        )

        tb.Label(general, text="Device").grid(row=2, column=0, sticky=W)
        self.device_var = tb.StringVar(value=self.settings.get("device", "Auto"))
        device_combo = tb.Combobox(
            general,
            values=["Auto", "GPU", "CPU"],
            textvariable=self.device_var,
            width=40,
            state="readonly",
        )
        device_combo.grid(row=2, column=1, sticky="we", padx=8)
        self.device_var.trace_add("write", lambda *_: self._on_device_change())
        self._add_tooltip(
            device_combo,
            "Preferred hardware backend. Auto selects GPU if available, otherwise CPU.",
        )

        tb.Label(general, text="Model directory").grid(row=3, column=0, sticky=W)
        self.model_dir_var = tb.StringVar(value=self.settings.get("model_dir", str(self.config.model_dir)))
        model_dir_frame = tb.Frame(general)
        model_dir_frame.grid(row=3, column=1, sticky="we", padx=8)
        model_dir_frame.grid_columnconfigure(0, weight=1)
        model_entry = tb.Entry(model_dir_frame, textvariable=self.model_dir_var)
        model_entry.grid(row=0, column=0, sticky="we", padx=(0, 6))
        model_entry.bind("<FocusOut>", lambda *_: self._on_model_dir_change())
        tb.Button(
            model_dir_frame,
            text="Browse",
            command=self._choose_model_dir,
        ).grid(row=0, column=1, padx=(0, 6))
        tb.Button(
            model_dir_frame,
            text="Save",
            command=self._persist_model_dir,
        ).grid(row=0, column=2)
        self._add_tooltip(
            model_dir_frame,
            "Choose where models are downloaded or loaded from.",
        )

    def _build_alpha_section(self, parent: tb.Frame) -> None:
        """Create the alpha matting refinement group."""

        frame = tb.Frame(parent, padding=10)
        frame.pack(fill="x", pady=4)
        frame.columnconfigure(1, weight=1)

        self.alpha_enabled_var = tb.BooleanVar(value=bool(self.settings.get("alpha_matting", False)))
        enable_check = tb.Checkbutton(
            frame,
            text="Enable alpha matting",
            variable=self.alpha_enabled_var,
            command=self._on_alpha_toggle,
        )
        enable_check.grid(row=0, column=0, columnspan=2, sticky=W)
        self._add_tooltip(enable_check, "Enable refined matting for detailed edges.")

        self.alpha_fg_var = tb.IntVar(value=int(self.settings.get("alpha_foreground_threshold", 240)))
        tb.Label(frame, text="Foreground threshold").grid(row=1, column=0, sticky=W)
        fg_scale = tb.Scale(
            frame,
            from_=0,
            to=255,
            orient="horizontal",
            variable=self.alpha_fg_var,
            command=lambda *_: self._on_alpha_change(),
        )
        fg_scale.grid(row=1, column=1, sticky="we", padx=8)
        fg_value = tb.Label(frame, text=str(self.alpha_fg_var.get()))
        fg_value.grid(row=1, column=2, sticky=W)
        self.alpha_fg_var.trace_add(
            "write",
            lambda *_: self._update_alpha_value(
                "alpha_foreground_threshold",
                self.alpha_fg_var,
                fg_value,
            ),
        )
        self._add_tooltip(fg_scale, "Minimum intensity considered foreground. Range: 0–255. Default: 240")

        self.alpha_bg_var = tb.IntVar(value=int(self.settings.get("alpha_background_threshold", 10)))
        tb.Label(frame, text="Background threshold").grid(row=2, column=0, sticky=W)
        bg_scale = tb.Scale(
            frame,
            from_=0,
            to=255,
            orient="horizontal",
            variable=self.alpha_bg_var,
            command=lambda *_: self._on_alpha_change(),
        )
        bg_scale.grid(row=2, column=1, sticky="we", padx=8)
        bg_value = tb.Label(frame, text=str(self.alpha_bg_var.get()))
        bg_value.grid(row=2, column=2, sticky=W)
        self.alpha_bg_var.trace_add(
            "write",
            lambda *_: self._update_alpha_value(
                "alpha_background_threshold",
                self.alpha_bg_var,
                bg_value,
            ),
        )
        self._add_tooltip(bg_scale, "Maximum intensity considered background. Range: 0–255. Default: 10")

        self.alpha_erode_var = tb.IntVar(value=int(self.settings.get("alpha_erode_size", 10)))
        tb.Label(frame, text="Erode size").grid(row=3, column=0, sticky=W)
        erode_scale = tb.Scale(
            frame,
            from_=0,
            to=30,
            orient="horizontal",
            variable=self.alpha_erode_var,
            command=lambda *_: self._on_alpha_change(),
        )
        erode_scale.grid(row=3, column=1, sticky="we", padx=8)
        erode_value = tb.Label(frame, text=str(self.alpha_erode_var.get()))
        erode_value.grid(row=3, column=2, sticky=W)
        self.alpha_erode_var.trace_add(
            "write",
            lambda *_: self._update_alpha_value(
                "alpha_erode_size",
                self.alpha_erode_var,
                erode_value,
            ),
        )
        self._add_tooltip(erode_scale, "Number of pixels to erode the mask. Range: 0–30. Default: 10")

        self.alpha_controls = [fg_scale, bg_scale, erode_scale]

    def _build_mask_section(self, parent: tb.Frame) -> None:
        """Create mask refinement controls."""

        frame = tb.Frame(parent, padding=10)
        frame.pack(fill="x", pady=4)
        frame.columnconfigure(1, weight=1)

        self.smoothing_var = tb.DoubleVar(value=float(self.settings.get("smoothing", 0.0)))
        tb.Label(frame, text="Smoothing").grid(row=0, column=0, sticky=W)
        smoothing_scale = tb.Scale(
            frame,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            variable=self.smoothing_var,
            command=lambda *_: self._on_mask_change(),
        )
        smoothing_scale.grid(row=0, column=1, sticky="we", padx=8)
        smoothing_value = tb.Label(frame, text=f"{self.smoothing_var.get():.2f}")
        smoothing_value.grid(row=0, column=2, sticky=W)
        self.smoothing_var.trace_add("write", lambda *_: self._update_smoothing(smoothing_value))
        self._add_tooltip(
            smoothing_scale,
            "Applies smoothing to soften mask edges. Range: 0–1. Start at 0.0 and increase only if needed.",
        )

        self.edge_var = tb.BooleanVar(value=bool(self.settings.get("edge_refinement", False)))
        edge_check = tb.Checkbutton(
            frame,
            text="Edge refinement",
            variable=self.edge_var,
            command=lambda: self._update_setting("edge_refinement", bool(self.edge_var.get())),
        )
        edge_check.grid(row=1, column=0, columnspan=2, sticky=W)
        self._add_tooltip(edge_check, "If enabled, runs an additional pass to sharpen boundaries.")

        self.feather_var = tb.IntVar(value=int(self.settings.get("feather_radius", 3)))
        tb.Label(frame, text="Feather radius").grid(row=2, column=0, sticky=W)
        feather_spin = tb.Spinbox(frame, from_=0, to=50, increment=1, textvariable=self.feather_var, width=10)
        feather_spin.grid(row=2, column=1, sticky=W, padx=8)
        self.feather_var.trace_add("write", lambda *_: self._on_feather_change())
        self._add_tooltip(feather_spin, "Feather the mask edges for smoother blending. Range: 0–50")

    def _build_output_section(self, parent: tb.Frame) -> None:
        """Create output configuration controls."""

        frame = tb.Frame(parent, padding=10)
        frame.pack(fill="x", pady=4)
        frame.columnconfigure(1, weight=1)

        tb.Label(frame, text="Format").grid(row=0, column=0, sticky=W)
        self.format_var = tb.StringVar(value=self.settings.get("output_format", "PNG"))
        format_combo = tb.Combobox(
            frame,
            values=["PNG", "JPEG", "WEBP"],
            textvariable=self.format_var,
            state="readonly",
        )
        format_combo.grid(row=0, column=1, sticky=W, padx=8)
        self.format_var.trace_add(
            "write",
            lambda *_: self._update_setting("output_format", self.format_var.get()),
        )
        self._add_tooltip(format_combo, "Select output image format.")

        self.preserve_var = tb.BooleanVar(value=bool(self.settings.get("preserve_names", False)))
        preserve_check = tb.Checkbutton(
            frame,
            text="Use original filename",
            variable=self.preserve_var,
            command=lambda: self._update_setting("preserve_names", bool(self.preserve_var.get())),
        )
        preserve_check.grid(row=1, column=0, columnspan=2, sticky=W)
        self._add_tooltip(
            preserve_check,
            "If enabled, processed files will overwrite original filenames (in output folder).",
        )

        tb.Label(frame, text="Output directory").grid(row=2, column=0, sticky=W)
        self.output_dir_var = tb.StringVar(value=self.settings.get("output_directory", ""))
        output_dir_frame = tb.Frame(frame)
        output_dir_frame.grid(row=2, column=1, sticky="we", padx=8)
        output_dir_frame.grid_columnconfigure(0, weight=1)
        output_entry = tb.Entry(output_dir_frame, textvariable=self.output_dir_var)
        output_entry.grid(row=0, column=0, sticky="we", padx=(0, 6))
        output_entry.bind(
            "<FocusOut>",
            lambda *_: self._update_setting("output_directory", self.output_dir_var.get()),
        )
        tb.Button(output_dir_frame, text="Browse", command=self._choose_output_dir).grid(row=0, column=1)
        self._add_tooltip(
            output_dir_frame,
            "Select where processed images will be saved. Defaults to ./output next to input folder.",
        )

    def _build_batch_section(self, parent: tb.Frame) -> None:
        """Create batch processing configuration controls."""

        frame = tb.Frame(parent, padding=10)
        frame.pack(fill="x", pady=4)
        frame.columnconfigure(1, weight=1)

        self.recursive_var = tb.BooleanVar(value=bool(self.settings.get("recursive", False)))
        recursive_check = tb.Checkbutton(
            frame,
            text="Include subfolders",
            variable=self.recursive_var,
            command=lambda: self._update_setting("recursive", bool(self.recursive_var.get())),
        )
        recursive_check.grid(row=0, column=0, columnspan=2, sticky=W)
        self._add_tooltip(recursive_check, "If checked, scans all subfolders for images.")

        self.threads_var = tb.IntVar(value=int(self.settings.get("parallel_threads", 4)))
        tb.Label(frame, text="Parallel threads").grid(row=1, column=0, sticky=W)
        threads_spin = tb.Spinbox(frame, from_=1, to=16, increment=1, textvariable=self.threads_var, width=10)
        threads_spin.grid(row=1, column=1, sticky=W, padx=8)
        self.threads_var.trace_add("write", lambda *_: self._on_threads_change())
        self._add_tooltip(
            threads_spin,
            "Number of threads to use for batch processing. Recommended: 4–8.",
        )

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _add_tooltip(self, widget: Any, text: str) -> None:
        """Attach or update a tooltip with ``text`` on ``widget``."""

        tooltip = self._tooltips.get(widget)
        if tooltip is None:
            tooltip = ToolTip(widget, text=text)
            self._tooltips[widget] = tooltip
            return
        try:
            tooltip.configure(text=text)
        except AttributeError:  # pragma: no cover - fallback
            tooltip.text = text

    def _toggle_theme(self) -> None:
        """Toggle between light and dark themes and persist selection."""

        new_theme = "darkly" if self.themename == "flatly" else "flatly"
        self.style.theme_use(new_theme)
        self.themename = new_theme
        self._update_setting("theme", new_theme)
        icon = "☀️" if new_theme == "darkly" else "🌙"
        self.theme_toggle_btn.configure(text=icon)

    def _toggle_advanced(self) -> None:
        """Toggle visibility of the advanced settings frame."""

        if self.advanced_visible.get():
            self.advanced_body.pack_forget()
            self.advanced_visible.set(False)
            self.toggle_button.configure(text="Show")
        else:
            self.advanced_body.pack(fill=BOTH, expand=True)
            self.advanced_visible.set(True)
            self.toggle_button.configure(text="Hide")

    def _toggle_alpha_controls(self) -> None:
        """Enable or disable alpha control widgets."""

        state = "normal" if self.alpha_enabled_var.get() else "disabled"
        for widget in self.alpha_controls:
            widget.configure(state=state)

    def _update_preview_controls(self) -> None:
        """Synchronize preview-related button states with available data."""

        has_preview = self._preview_image is not None
        has_saved = self._preview_saved_path is not None and self._preview_saved_path.exists()
        self.save_button.configure(state="normal" if has_preview else "disabled")
        self.discard_button.configure(state="normal" if has_preview else "disabled")
        self.view_full_button.configure(state="normal" if has_saved else "disabled")
        fill_state = "normal" if has_preview else "disabled"
        clear_state = "normal" if has_preview and self._preview_display_override else "disabled"
        self.preview_fill_button.configure(state=fill_state)
        self.preview_clear_button.configure(state=clear_state)

    def _refresh_badge(self) -> None:
        """Update the provider badge and tooltip."""

        badge_style = (
            "success"
            if self.providers and self.providers[0].lower().startswith("cuda")
            else "secondary"
        )
        badge_text = "GPU" if badge_style == "success" else "CPU"
        self.badge.configure(text=badge_text, bootstyle=f"{badge_style}-inverse")
        tooltip_text = "Providers: " + ", ".join(self.providers or ["CPUExecutionProvider"])
        self._add_tooltip(self.badge, tooltip_text)

    def _preview_fill_bg(self) -> None:
        """Prompt for a fill colour, store it, and apply it to the preview."""

        from tkinter import colorchooser

        chosen = colorchooser.askcolor(title="Select background colour")
        if not chosen or not chosen[0]:
            return

        fill = tuple(map(int, chosen[0]))
        self._preview_fill_color = fill
        self._preview_last_fill_color = None

        if not self._preview_image:
            self._log(
                f"Background colour set to {self._preview_fill_color} for future previews."
            )
            return

        if self._apply_background_fill():
            self._render_preview_image()
            self._log(
                f"Background colour set to {self._preview_fill_color} and applied to the preview."
            )
        else:
            self._log(f"Background colour set to {self._preview_fill_color}.")
        self._update_preview_controls()

    def _preview_clear_bg(self) -> None:
        """Clear the stored fill colour and visualise transparency in the preview."""

        self._preview_fill_color = None
        self._preview_last_fill_color = None

        if not self._preview_image:
            self._preview_display_override = None
            self._update_preview_controls()
            return

        transparent = self._compose_transparency_preview()
        if transparent is None:
            return

        self._preview_display_override = transparent
        self._render_preview_image()
        self._update_preview_controls()
        self._log("Background cleared — transparency restored.")

    def _compose_transparency_preview(self) -> Image.Image | None:
        """Return the preview composited over a checkerboard background."""

        if not self._preview_image:
            return None

        rgba = self._preview_image.convert("RGBA")
        width, height = rgba.size
        checker_size = 16
        checker = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(checker)
        for y in range(0, height, checker_size):
            for x in range(0, width, checker_size):
                if (x // checker_size + y // checker_size) % 2 == 0:
                    draw.rectangle([x, y, x + checker_size, y + checker_size], fill=(200, 200, 200))

        return Image.alpha_composite(checker.convert("RGBA"), rgba)
    def _on_model_change(self) -> None:
        """Handle updates to the selected model."""

        model_key = self.model_var.get()
        self._update_setting("model_key", model_key)

    def _on_device_change(self) -> None:
        """Handle device preference updates."""

        self._update_setting("device", self.device_var.get())
        self.providers = detect_providers(self._provider_hints())
        self._refresh_badge()
        self._update_preview_controls()

    def _on_model_dir_change(self) -> None:
        """Persist model directory text edits into settings."""

        self._update_setting("model_dir", self.model_dir_var.get())

    def _persist_model_dir(self) -> None:
        """Persist the active configuration including the model directory."""

        model_dir_text = (self.model_dir_var.get() or "").strip()
        if not model_dir_text:
            messagebox.showerror("Error", "Please choose a directory before saving.")
            return
        updated = self.config.with_updates(model_dir=Path(model_dir_text))
        try:
            persist_config(updated)
            self.config = updated
            self._log("Model directory saved.")
            self._update_setting("model_dir", model_dir_text)
        except Exception as error:  # pragma: no cover - GUI feedback only
            messagebox.showerror("Error", f"Unable to save configuration: {error}")

    def _choose_model_dir(self) -> None:
        """Display a directory chooser for the model directory."""

        directory = filedialog.askdirectory(initialdir=str(MODELS_DIR))
        if directory:
            self.model_dir_var.set(directory)
            self._on_model_dir_change()

    def _choose_output_dir(self) -> None:
        """Display a directory chooser for the output directory."""

        directory = filedialog.askdirectory(initialdir=str(OUTPUT_DIR))
        if directory:
            self.output_dir_var.set(directory)
            self._update_setting("output_directory", directory)

    def _on_alpha_toggle(self) -> None:
        """Handle toggling the alpha matting checkbox."""

        enabled = bool(self.alpha_enabled_var.get())
        self._update_setting("alpha_matting", enabled)
        self._toggle_alpha_controls()

    def _on_alpha_change(self) -> None:
        """Persist alpha slider edits."""

        self._update_setting("alpha_foreground_threshold", int(self.alpha_fg_var.get()))
        self._update_setting("alpha_background_threshold", int(self.alpha_bg_var.get()))
        self._update_setting("alpha_erode_size", int(self.alpha_erode_var.get()))

    def _update_alpha_value(self, key: str, variable: tb.Variable, label: tb.Label) -> None:
        """Update alpha-related setting and UI label."""

        value = int(float(variable.get()))
        label.configure(text=str(value))
        self._update_setting(key, value, persist=False)
        self._save_settings()

    def _on_mask_change(self) -> None:
        """Persist smoothing changes."""

        self._update_setting("smoothing", _coerce_smoothing(self.smoothing_var.get()))

    def _update_smoothing(self, label: tb.Label) -> None:
        """Refresh smoothing label and persist value."""

        value = _coerce_smoothing(self.smoothing_var.get())
        label.configure(text=f"{value:.2f}")
        self._update_setting("smoothing", value)

    def _on_feather_change(self) -> None:
        """Persist feather radius changes."""

        try:
            value = int(self.feather_var.get())
        except (TypeError, ValueError):
            value = 3
        self._update_setting("feather_radius", max(0, min(50, value)))

    def _on_threads_change(self) -> None:
        """Persist thread count changes."""

        try:
            value = int(self.threads_var.get())
        except (TypeError, ValueError):
            value = 1
        value = max(1, min(16, value))
        self.threads_var.set(value)
        self._update_setting("parallel_threads", value)

    # ------------------------------------------------------------------
    # File chooser helpers
    # ------------------------------------------------------------------
    def _choose_single_file(self) -> None:
        """Prompt the user for a single image file."""

        filename = filedialog.askopenfilename(
            initialdir=str(INPUT_DIR),
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tiff")]
        )
        if filename:
            self.single_input_var.set(filename)

    def _choose_single_output(self) -> None:
        """Prompt the user for an optional single-image output path."""

        filename = filedialog.asksaveasfilename(
            initialdir=str(OUTPUT_DIR),
            defaultextension=".png",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")],
        )
        if filename:
            self.single_output_var.set(filename)

    def _choose_batch_folder(self) -> None:
        """Prompt the user for a batch input directory."""

        directory = filedialog.askdirectory(initialdir=str(INPUT_DIR))
        if directory:
            self.batch_input_var.set(directory)

    def _choose_batch_output(self) -> None:
        """Prompt the user for an optional batch output directory."""

        directory = filedialog.askdirectory(initialdir=str(OUTPUT_DIR))
        if directory:
            self.batch_output_var.set(directory)

    # ------------------------------------------------------------------
    # Processing helpers
    # ------------------------------------------------------------------
    def _log(self, message: str, *, error: bool = False) -> None:
        """Append ``message`` to the activity log."""

        tag = "error" if error else None
        self.log_widget.insert(END, message + "\n", tag)
        self.log_widget.see(END)

    def _determine_single_output(self, input_path: Path) -> Path:
        """Return the output path for single image processing."""

        explicit = (self.single_output_var.get() or "").strip()
        if explicit:
            return Path(explicit)
        output_dir = (self.output_dir_var.get() or "").strip()
        if output_dir:
            base_dir = Path(output_dir)
        else:
            base_dir = OUTPUT_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        format_hint, suffix = _format_meta(self.settings.get("output_format", "PNG"))
        if self.settings.get("preserve_names"):
            return base_dir / f"{input_path.stem}.{suffix}"
        return base_dir / f"{input_path.stem}_no_bg.{suffix}"

    def _process_single(self) -> None:
        """Validate inputs and start single image processing."""

        path = Path(self.single_input_var.get())
        if not path.exists():
            messagebox.showerror("Error", "Please choose a valid input image.")
            return
        output_path = self._determine_single_output(path)
        self._log(f"Starting processing for {path.name}…")
        self._set_processing_state(True, "single")
        threading.Thread(target=self._run_single, args=(path, output_path), daemon=True).start()

    def _run_single(self, input_path: Path, output_path: Path) -> None:
        """Worker that performs single image processing."""

        try:
            source_image = self._load_source_image(input_path)
            array = image_to_numpy(source_image)
            config = self._active_config()
            kwargs = self._processing_kwargs()
            kwargs["feather_radius"] = int(self.settings.get("feather_radius", 3))
            result = process_image(
                array,
                model_key=self.model_var.get(),
                config=config,
                **kwargs,
            )
            result_image = result.image
            format_hint, _ = _format_meta(self.settings.get("output_format", "PNG"))
            self.after(
                0,
                lambda: self._show_preview(
                    result_image,
                    output_path,
                    format_hint,
                    input_path.name,
                ),
            )
            self.after(0, lambda: self._set_processing_state(False, "single"))
            self.after(0, lambda: self._on_single_run_complete(input_path))
        except Exception as error:
            LOGGER.exception("Single image processing failed")
            message = str(error)
            self.after(0, lambda: self._log(f"{input_path.name} failed ✗ — Reason: {message}", error=True))
            self.after(0, lambda: messagebox.showerror("Processing failed", message))
            self.after(0, lambda: self._set_processing_state(False, "single"))
            self.after(0, lambda: self._on_single_run_complete(input_path))

    def _load_source_image(self, path: Path) -> Image.Image:
        """Return a freshly loaded RGBA image from ``path``."""

        with Image.open(path) as source:
            return source.convert("RGBA")

    def _save_processed_image(
        self,
        pil_image: Image.Image,
        output_path: Path,
        format_hint: str,
    ) -> None:
        """Persist ``pil_image`` to ``output_path`` respecting the configured format."""

        image_to_save = pil_image
        if format_hint != "PNG" and pil_image.mode != "RGB":
            image_to_save = pil_image.convert("RGB")
        save_image_to_path(image_to_save, output_path, format_hint=format_hint)

    def _clear_preview_state(self) -> None:
        """Reset preview data structures and disable preview controls."""

        self._preview_image = None
        self._preview_photo = None
        self._preview_output_path = None
        self._preview_format_hint = None
        self._preview_original_name = None
        self._preview_saved_path = None
        self._preview_canvas_image = None
        self._preview_display_override = None
        self._preview_last_fill_color = None
        self.preview_canvas.delete("all")
        self.preview_canvas.configure(scrollregion=(0, 0, 0, 0))
        self._preview_zoom_manual_override = False
        self._preview_render_size = (0, 0)
        self._preview_canvas_size = (
            max(1, self.preview_canvas.winfo_width()),
            max(1, self.preview_canvas.winfo_height()),
        )
        self._preview_zoom_updating = True
        try:
            self.preview_zoom_var.set(PREVIEW_ZOOM_DEFAULT)
        finally:
            self._preview_zoom_updating = False
        self.preview_zoom_value.configure(text=f"{int(PREVIEW_ZOOM_DEFAULT)}%")
        self.preview_info.configure(text="No preview available yet.")
        self._update_preview_controls()

    def _apply_background_fill(self) -> bool:
        """Apply the stored fill colour beneath the preview image when available."""

        if not self._preview_image or self._preview_fill_color is None:
            return False

        fill = self._preview_fill_color
        rgba = self._preview_image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, fill + (255,))
        composed = Image.alpha_composite(background, rgba).convert("RGB")
        self._preview_display_override = composed
        self._preview_last_fill_color = fill
        return True

    def _render_preview_image(self, anchor: PreviewAnchor | None = None) -> None:
        """Render the in-memory preview image respecting the zoom slider."""

        if not self._preview_image:
            self.preview_canvas.delete("all")
            self.preview_canvas.configure(scrollregion=(0, 0, 0, 0))
            self._preview_render_size = (0, 0)
            return

        zoom_value = max(PREVIEW_ZOOM_MIN, min(PREVIEW_ZOOM_MAX, float(self.preview_zoom_var.get())))
        if zoom_value != float(self.preview_zoom_var.get()):
            self._preview_zoom_updating = True
            try:
                self.preview_zoom_var.set(zoom_value)
            finally:
                self._preview_zoom_updating = False
        scale = zoom_value / 100.0
        source_image = self._preview_display_override or self._preview_image
        width = max(1, int(round(source_image.width * scale)))
        height = max(1, int(round(source_image.height * scale)))
        previous_size = self._preview_render_size
        self._preview_render_size = (width, height)
        resized = source_image.resize((width, height), Image.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(resized)
        self.preview_canvas.delete("all")
        self._preview_canvas_image = self.preview_canvas.create_image(
            0,
            0,
            anchor="nw",
            image=self._preview_photo,
        )
        self.preview_canvas.configure(scrollregion=(0, 0, width, height))
        self.preview_zoom_value.configure(text=f"{int(round(zoom_value))}%")
        self._update_preview_view(anchor, previous_size)

    def _update_preview_view(
        self,
        anchor: PreviewAnchor | None,
        previous_size: tuple[int, int],
    ) -> None:
        """Adjust the canvas viewport to keep the ``anchor`` position stable."""

        if self._preview_canvas_image is None:
            return

        widget_width = max(1, self.preview_canvas.winfo_width())
        widget_height = max(1, self.preview_canvas.winfo_height())
        new_width, new_height = self._preview_render_size
        if new_width <= 0 or new_height <= 0:
            return

        if anchor is None:
            anchor = self._current_view_anchor()
            if anchor is None:
                return

        canvas_x, canvas_y, pointer_x, pointer_y = anchor
        pointer_x = float(min(max(pointer_x, 0.0), widget_width))
        pointer_y = float(min(max(pointer_y, 0.0), widget_height))

        old_width, old_height = previous_size
        if old_width <= 0 or old_height <= 0:
            anchor_ratio_x = pointer_x / widget_width
            anchor_ratio_y = pointer_y / widget_height
        else:
            anchor_ratio_x = float(canvas_x) / float(old_width)
            anchor_ratio_y = float(canvas_y) / float(old_height)

        anchor_ratio_x = min(max(anchor_ratio_x, 0.0), 1.0)
        anchor_ratio_y = min(max(anchor_ratio_y, 0.0), 1.0)

        new_anchor_x = anchor_ratio_x * new_width
        new_anchor_y = anchor_ratio_y * new_height

        max_x = max(0.0, new_width - widget_width)
        max_y = max(0.0, new_height - widget_height)
        target_x = min(max(new_anchor_x - pointer_x, 0.0), max_x)
        target_y = min(max(new_anchor_y - pointer_y, 0.0), max_y)

        if new_width > 0:
            self.preview_canvas.xview_moveto(target_x / new_width)
        if new_height > 0:
            self.preview_canvas.yview_moveto(target_y / new_height)

    def _current_view_anchor(self) -> PreviewAnchor | None:
        """Return the anchor representing the viewport centre."""

        width, height = self._preview_render_size
        if width <= 0 or height <= 0:
            return None

        widget_width = max(1, self.preview_canvas.winfo_width())
        widget_height = max(1, self.preview_canvas.winfo_height())
        start_x, _ = self.preview_canvas.xview()
        start_y, _ = self.preview_canvas.yview()
        canvas_x = start_x * width + widget_width / 2.0
        canvas_y = start_y * height + widget_height / 2.0
        return (canvas_x, canvas_y, widget_width / 2.0, widget_height / 2.0)

    def _auto_fit_preview(self) -> None:
        """Scale the preview to fit inside the canvas when auto-fit is active."""

        if not self._preview_image:
            return

        canvas_width, canvas_height = self._preview_canvas_size
        if canvas_width <= 0 or canvas_height <= 0:
            canvas_width = max(1, self.preview_canvas.winfo_width())
            canvas_height = max(1, self.preview_canvas.winfo_height())

        source_image = self._preview_display_override or self._preview_image
        if source_image.width <= 0 or source_image.height <= 0:
            return

        scale = min(canvas_width / source_image.width, canvas_height / source_image.height)
        if scale <= 0:
            return

        zoom_value = max(PREVIEW_ZOOM_MIN, min(PREVIEW_ZOOM_MAX, scale * 100.0))
        if abs(zoom_value - float(self.preview_zoom_var.get())) < 0.1 and self._preview_canvas_image:
            return

        self._preview_zoom_manual_override = False
        self._apply_preview_zoom(zoom_value)

    def _apply_preview_zoom(
        self,
        zoom_value: float,
        *,
        anchor: PreviewAnchor | None = None,
        manual_override: bool = False,
    ) -> None:
        """Clamp and apply ``zoom_value`` while optionally preserving ``anchor``."""

        clamped = max(PREVIEW_ZOOM_MIN, min(PREVIEW_ZOOM_MAX, float(zoom_value)))
        if manual_override:
            self._preview_zoom_manual_override = True
        self._preview_zoom_updating = True
        try:
            self.preview_zoom_var.set(clamped)
        finally:
            self._preview_zoom_updating = False
        self._render_preview_image(anchor)

    def _build_anchor_from_event(self, event: Any) -> PreviewAnchor | None:
        """Return an anchor derived from ``event`` coordinates if possible."""

        if self._preview_canvas_image is None:
            return None

        pointer_x = float(getattr(event, "x", 0.0))
        pointer_y = float(getattr(event, "y", 0.0))
        canvas_x = self.preview_canvas.canvasx(pointer_x)
        canvas_y = self.preview_canvas.canvasy(pointer_y)
        return (canvas_x, canvas_y, pointer_x, pointer_y)

    def _on_preview_canvas_enter(self, _event: Any) -> None:
        """Focus the preview canvas when the cursor enters its bounds."""

        self._preview_canvas_hover = True
        try:
            self.preview_canvas.focus_set()
        except Exception:  # pragma: no cover - focus best effort
            pass

    def _on_preview_canvas_leave(self, _event: Any) -> None:
        """Track when the pointer leaves the preview canvas."""

        self._preview_canvas_hover = False

    def _on_preview_mouse_wheel(self, event: Any) -> None:
        """Adjust zoom in response to mouse wheel events."""

        if not self._preview_image:
            return

        if not self._preview_canvas_hover and self.focus_get() is not self.preview_canvas:
            return

        delta = 0
        if hasattr(event, "delta") and event.delta:
            delta = 1 if event.delta > 0 else -1
        elif getattr(event, "num", None) in (4, 5):
            delta = 1 if event.num == 4 else -1
        if delta == 0:
            return

        anchor = self._build_anchor_from_event(event)
        new_zoom = float(self.preview_zoom_var.get()) + delta * PREVIEW_ZOOM_WHEEL_STEP
        self._apply_preview_zoom(new_zoom, anchor=anchor, manual_override=True)

    def _on_preview_drag_start(self, event: Any) -> None:
        """Record the initial pointer position for preview panning."""

        self.preview_canvas.scan_mark(event.x, event.y)

    def _on_preview_drag_motion(self, event: Any) -> None:
        """Pan the image preview while the left mouse button is held."""

        self.preview_canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_preview_zoom(self) -> None:
        """Handle zoom slider changes by re-rendering the preview image."""

        if not self._preview_image:
            self._preview_zoom_updating = True
            try:
                self.preview_zoom_var.set(PREVIEW_ZOOM_DEFAULT)
            finally:
                self._preview_zoom_updating = False
            self.preview_zoom_value.configure(text=f"{int(PREVIEW_ZOOM_DEFAULT)}%")
            return

        anchor = None
        if not self._preview_zoom_updating:
            self._preview_zoom_manual_override = True
            anchor = self._current_view_anchor()
        self._render_preview_image(anchor)

    def _on_preview_canvas_resize(self, event: Any) -> None:
        """Update the canvas scroll region after a resize event."""

        previous_size = self._preview_canvas_size
        width = max(1, int(getattr(event, "width", self.preview_canvas.winfo_width())))
        height = max(1, int(getattr(event, "height", self.preview_canvas.winfo_height())))
        self._preview_canvas_size = (width, height)

        if self._preview_canvas_image is not None:
            bbox = self.preview_canvas.bbox(self._preview_canvas_image)
            if bbox:
                self.preview_canvas.configure(scrollregion=bbox)

        if (
            self._preview_image
            and not self._preview_zoom_manual_override
            and previous_size != self._preview_canvas_size
        ):
            self._auto_fit_preview()

    def _on_preview_save(self) -> None:
        """Persist the preview image using the configured output path."""

        if (
            not self._preview_image
            or not self._preview_output_path
            or not self._preview_format_hint
        ):
            return
        try:
            self._save_processed_image(
                self._preview_image,
                self._preview_output_path,
                self._preview_format_hint,
            )
        except Exception as error:  # pragma: no cover - GUI feedback
            error_message = str(error)
            self._log(f"Failed to save image ✗ — Reason: {error_message}", error=True)
            messagebox.showerror("Save failed", error_message)
            return

        self._preview_saved_path = self._preview_output_path
        original_name = self._preview_original_name or "Image"
        self._log(f"{original_name} processed successfully ✓")
        self.preview_info.configure(
            text=f"Saved to {self._preview_output_path}",
        )
        self._update_preview_controls()

    def _on_preview_discard(self) -> None:
        """Discard the current preview image and reset controls."""

        if not self._preview_image:
            return
        self._log("❌ Preview discarded without saving.")
        self._clear_preview_state()

    def _view_saved_preview(self) -> None:
        """Open the saved preview image in the default system viewer."""

        if not self._preview_saved_path or not self._preview_saved_path.exists():
            messagebox.showerror(
                "Image not saved",
                "Save the preview before opening it in an external viewer.",
            )
            return
        try:
            webbrowser.open(self._preview_saved_path.as_uri())
        except Exception as error:  # pragma: no cover - GUI feedback
            messagebox.showerror("Unable to open image", str(error))

    def _show_preview(
        self,
        pil_image: Image.Image,
        output_path: Path,
        format_hint: str,
        original_name: str,
    ) -> None:
        """Render ``pil_image`` inside the inline preview panel."""

        self._preview_image = pil_image
        self._preview_output_path = output_path
        self._preview_format_hint = format_hint
        self._preview_original_name = original_name
        self._preview_saved_path = None
        self._preview_display_override = None
        self._preview_last_fill_color = None
        self._preview_zoom_manual_override = False
        self._preview_zoom_updating = True
        try:
            self.preview_zoom_var.set(PREVIEW_ZOOM_DEFAULT)
        finally:
            self._preview_zoom_updating = False
        self._preview_canvas_size = (
            max(1, self.preview_canvas.winfo_width()),
            max(1, self.preview_canvas.winfo_height()),
        )
        self.preview_info.configure(text=f"Preview ready: {original_name}")
        applied_fill = self._apply_background_fill()
        self._render_preview_image()
        self._auto_fit_preview()
        if applied_fill and self._preview_fill_color is not None:
            self._log(
                "Preview generated successfully ✔ — "
                f"background colour {self._preview_fill_color} applied."
            )
        else:
            self._log("Preview generated successfully ✔")
        self._update_preview_controls()

    def _process_batch(self) -> None:
        """Validate inputs and start batch processing."""

        path = Path(self.batch_input_var.get())
        if not path.exists() or not path.is_dir():
            messagebox.showerror("Error", "Please choose a valid input folder.")
            return
        if self.batch_output_var.get():
            output_dir = Path(self.batch_output_var.get())
        elif self.output_dir_var.get():
            output_dir = Path(self.output_dir_var.get())
        else:
            output_dir = None
        self._reset_batch_progress()
        self._log(f"Starting processing for {path.name}…")
        self._set_processing_state(True, "batch")
        threading.Thread(target=self._run_batch, args=(path, output_dir), daemon=True).start()

    def _reset_batch_progress(self) -> None:
        """Clear progress indicators for a new batch run."""

        for item in self.batch_tree.get_children():
            self.batch_tree.delete(item)

    def _run_batch(self, input_dir: Path, output_dir: Path | None) -> None:
        """Worker that performs batch processing."""

        try:
            config = self._active_config()
            kwargs = self._processing_kwargs()
            kwargs.update(
                {
                    "feather_radius": int(self.settings.get("feather_radius", 3)),
                    "recursive": bool(self.settings.get("recursive", False)),
                    "output_format": self.settings.get("output_format", "PNG"),
                    "preserve_names": bool(self.settings.get("preserve_names", False)),
                }
            )
            batch_output_dir = output_dir or OUTPUT_DIR
            batch_output_dir.mkdir(parents=True, exist_ok=True)
            report = process_folder(
                input_dir,
                batch_output_dir,
                "*",
                model_key=self.model_var.get(),
                config=config,
                progress_callback=self._on_batch_entry,
                **kwargs,
            )
            summary = f"Batch complete: {report.successes}/{report.total} succeeded"
            self.after(0, lambda: self._log(summary))
            if report.failures:
                self.after(0, lambda: messagebox.showerror("Batch finished with errors", summary))
            self.after(0, lambda: self._set_processing_state(False, "batch"))
        except Exception as error:
            LOGGER.exception("Batch processing failed")
            message = str(error)
            self.after(0, lambda: self._log(f"Batch failed ✗ — Reason: {message}", error=True))
            self.after(0, lambda: messagebox.showerror("Processing failed", message))
            self.after(0, lambda: self._set_processing_state(False, "batch"))

    def _on_batch_entry(self, entry: ReportEntry) -> None:
        """Schedule UI updates for batch progress entries."""

        self.after(0, lambda: self._record_batch_entry(entry))

    def _record_batch_entry(self, entry: ReportEntry) -> None:
        """Display a :class:`ReportEntry` inside the progress table."""

        status = "✓" if entry.success else "✗"
        details = entry.error or "Completed"
        self.batch_tree.insert("", END, values=(entry.path_in.name, status, details))
        if entry.success:
            log_message = f"{entry.path_in.name} processed successfully ✓"
        else:
            log_message = f"{entry.path_in.name} failed ✗ — Reason: {details}"
        self._log(log_message, error=not entry.success)

    def _set_processing_state(self, active: bool, context: str) -> None:
        """Toggle interactive widgets and visual indicators for processing state."""

        spinners = {
            "single": getattr(self, "single_spinner", None),
            "batch": getattr(self, "batch_spinner", None),
        }
        pack_options = {
            "single": {"fill": "x", "pady": (6, 0)},
            "batch": {"fill": "x", "pady": (0, 10)},
        }
        target_spinner = spinners.get(context)
        if active:
            self._processing_context = context
            if target_spinner is not None:
                target_spinner.pack(**pack_options.get(context, {"fill": "x"}))
                target_spinner.start(10)
            self.preview_overlay_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.preview_overlay_frame.lift()
            self.single_process_button.configure(state="disabled")
            self.batch_process_button.configure(state="disabled")
            self.save_button.configure(state="disabled")
            self.discard_button.configure(state="disabled")
            self.view_full_button.configure(state="disabled")
            self.preview_fill_button.configure(state="disabled")
            self.preview_clear_button.configure(state="disabled")
        else:
            if self._processing_context == context:
                self._processing_context = None
            for spinner in spinners.values():
                if spinner is not None:
                    spinner.stop()
                    if spinner.winfo_manager():
                        spinner.pack_forget()
            self.preview_overlay_frame.place_forget()
            self.single_process_button.configure(state="normal")
            self.batch_process_button.configure(state="normal")
            self._update_preview_controls()


def main() -> None:
    """Entry point used by ``python -m`` execution."""

    app = BackgroundRemoverApp()
    app.mainloop()


if __name__ == "__main__":
    main()
