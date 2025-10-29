"""Tkinter front-end for background removal workflows.

This module provides a ttkbootstrap-powered desktop interface that mirrors
core options available in the Flask web UI. It supports both single-image
processing and batch folder processing with live progress reporting.
"""
from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
import traceback
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from tkinter import filedialog
from typing import TYPE_CHECKING, Any

import ttkbootstrap as tb
from PIL import Image, ImageTk
from ttkbootstrap.constants import BOTH, END, LEFT, RIGHT, W
from ttkbootstrap.dialogs import Messagebox
from ttkbootstrap.scrolled import ScrolledText
from ttkbootstrap.tooltip import ToolTip

from bg_removal import (
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_FORMAT,
    OUTPUT_FORMATS,
    SUPPORTED_EXTENSIONS,
    ensure_global_session,
    ensure_models_downloaded,
    get_models_directory,
    get_output_format_spec,
    remove_bg_file,
    set_models_directory,
)

# Model options mirror the Flask UI so users can switch between
# performance profiles without leaving the desktop client.
REMOVAL_MODEL_OPTIONS: tuple[dict[str, str], ...] = (
    {
        "key": "general",
        "label": "General Model (isnet-general-use)",
        "model_name": "isnet-general-use",
    },
    {
        "key": "general_high_quality",
        "label": "BRIA RMBG v2.0 (High-Quality General)",
        "model_name": "briaai/RMBG-2.0",
    },
    {
        "key": "human",
        "label": "Human Model (u2net_human_seg)",
        "model_name": "u2net_human_seg",
    },
    {
        "key": "human_matting",
        "label": "BRIA RMBG v1.4 (Portrait Matting)",
        "model_name": "matting-by-generation",
    },
    {
        "key": "object",
        "label": "Object Model (u2net)",
        "model_name": "u2net",
    },
    {
        "key": "complex_scene",
        "label": "BRIA RMBG v2.0 (Complex Scenes)",
        "model_name": "sam_segmentation_model",
    },
    {
        "key": "anime",
        "label": "Anime / Illustration Model (isnet-anime)",
        "model_name": "isnet-anime",
    },
)

_REMOVAL_MODEL_LOOKUP: dict[str, str] = {
    option["key"]: option["model_name"] for option in REMOVAL_MODEL_OPTIONS
}

DEFAULT_ALPHA_MATTING_VALUES = {
    "alpha_matting": False,
    "am_foreground": 240,
    "am_background": 10,
    "am_erode": 10,
}

DEFAULT_TUNE_VALUES = {
    "feather_radius": 3,
    "colorkey_tolerance": 14,
    "use_colorkey_fallback": True,
}


def resource_path(rel_path: str) -> str:
    """Return the absolute path for ``rel_path`` when bundled by PyInstaller."""

    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, rel_path)


def _ensure_pyinstaller_hidden_imports() -> None:
    """Import modules that PyInstaller struggles to detect automatically."""

    if TYPE_CHECKING:
        return

    try:
        import importlib

        for module_name in (
            "ttkbootstrap.dialogs",
            "ttkbootstrap.tooltip",
            "ttkbootstrap.scrolled",
            "ttkbootstrap.constants",
            "PIL.Image",
            "PIL.ImageTk",
            "watchdog.events",
            "watchdog.observers",
        ):
            importlib.import_module(module_name)
    except Exception:
        # Import failures are non-fatal during development but will be logged.
        logging.getLogger(__name__).debug(
            "Optional PyInstaller imports are unavailable in this environment.",
            exc_info=True,
        )


_ensure_pyinstaller_hidden_imports()

def _runtime_directory() -> Path:
    """Return the directory that should contain runtime artefacts."""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


LOG_FILE = _runtime_directory() / "error.log"


def _write_traceback_to_log(traceback_text: str) -> Path:
    """Persist ``traceback_text`` to :data:`LOG_FILE` and return the path."""

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as log_file:
            if LOG_FILE.exists() and LOG_FILE.stat().st_size > 0:
                log_file.write("\n")
            log_file.write("=" * 80 + "\n")
            log_file.write(time.strftime("%Y-%m-%d %H:%M:%S"))
            log_file.write("\n")
            log_file.write(traceback_text)
    except Exception:  # pragma: no cover - best-effort logging fallback
        print("Failed to write error log:")
        print(traceback_text)
    return LOG_FILE


def create_tooltip(widget: Any, text: str, *, wraplength: int = 280) -> ToolTip:
    """Attach a tooltip with consistent styling to ``widget`` and return it."""

    return ToolTip(widget, text, wraplength=wraplength)


def _configure_logging() -> None:
    """Initialise a file-based logger for capturing runtime issues."""

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(LOG_FILE):
            return

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


LOGGER = logging.getLogger(__name__)


def _show_fatal_error_dialog(title: str, log_path: Path) -> None:
    """Display a blocking error dialog referencing ``log_path``."""

    message = f"An unexpected error occurred. See {log_path} for details."
    try:
        Messagebox.show_error(message, title)
    except Exception:
        # ``Messagebox`` relies on an existing Tk interpreter. Create a minimal
        # fallback root to surface the error when initialisation fails early.
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()


def safe_callback(fn):
    """Return a wrapper that logs callback errors without crashing the UI."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        owner = args[0] if args else None

        def handle_exception(exc: Exception) -> None:
            """Log ``exc`` to the GUI and :data:`LOG_FILE`."""

            tb_text = traceback.format_exc()
            message = f"❌ {type(exc).__name__}: {exc}"
            if owner is not None and hasattr(owner, "log_status"):
                try:
                    owner.log_status(message, color="red")
                except Exception:  # pragma: no cover - defensive logging
                    LOGGER.error(
                        "Failed to log status message for callback %s.",
                        fn.__name__,
                        exc_info=True,
                    )
            else:
                LOGGER.error(message)

            log_path = _write_traceback_to_log(tb_text)
            LOGGER.error(
                "Unhandled exception in Tkinter callback %s. Traceback stored at %s.",
                fn.__name__,
                log_path,
                exc_info=True,
            )
            print(tb_text, file=sys.stderr, flush=True)

        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - runtime safety net
            handle_exception(exc)
            return None

        if isinstance(result, Iterator):

            def generator_wrapper() -> Iterator[Any]:
                try:
                    yield from result
                except Exception as exc:  # pragma: no cover - runtime safety net
                    handle_exception(exc)
                    return

            return generator_wrapper()

        return result

    return wrapper


def human_readable_duration(seconds: float) -> str:
    """Return a friendly textual representation for ``seconds``."""

    if seconds <= 0:
        return "Calculating…"
    minutes, secs = divmod(int(seconds), 60)
    if minutes == 0:
        return f"{secs}s remaining"
    return f"{minutes}m {secs}s remaining"


@dataclass(slots=True)
class FolderTaskOptions:
    """Container describing configuration shared with the worker thread."""

    folder: Path
    output_root: Path
    output_format: str
    model_key: str
    recursive: bool
    skip_existing: bool
    alpha_matting: bool
    am_foreground: int
    am_background: int
    am_erode: int
    feather_radius: int
    use_colorkey_fallback: bool
    colorkey_tolerance: int

    def as_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for :func:`remove_bg_file`."""

        model_name = _REMOVAL_MODEL_LOOKUP.get(self.model_key, DEFAULT_MODEL_NAME)
        return {
            "output_format": self.output_format,
            "model_name": model_name,
            "alpha_matting": self.alpha_matting,
            "am_foreground": self.am_foreground,
            "am_background": self.am_background,
            "am_erode": self.am_erode,
            "feather_radius": self.feather_radius,
            "use_colorkey_fallback": self.use_colorkey_fallback,
            "colorkey_tolerance": self.colorkey_tolerance,
        }


class FolderProcessor(threading.Thread):
    """Process folders of images in a background thread."""

    def __init__(
        self,
        options: FolderTaskOptions,
        events: queue.Queue[tuple[str, dict[str, Any]]],
        cancel_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.options = options
        self.events = events
        self.cancel_event = cancel_event

    def _iter_sources(self) -> Iterable[Path]:
        """Yield supported source images."""

        folder = self.options.folder
        iterator: Iterable[Path]
        if self.options.recursive:
            iterator = folder.rglob("*")
        else:
            iterator = folder.iterdir()
        for item in iterator:
            if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield item

    def run(self) -> None:  # noqa: D401 - inherited behaviour documented above.
        # Saving occurs inside a dedicated ``output_root`` folder to keep
        # original images untouched. The location is user-configurable via
        # :class:`FolderTaskOptions`.
        output_root = self.options.output_root
        output_root.mkdir(parents=True, exist_ok=True)

        sources = list(self._iter_sources())
        total = len(sources)
        if total == 0:
            self.events.put(
                (
                    "batch_empty",
                    {
                        "folder": str(self.options.folder),
                        "output": str(output_root),
                    },
                )
            )
            return

        self.events.put(
            (
                "batch_start",
                {
                    "total": total,
                    "folder": str(self.options.folder),
                    "output": str(output_root),
                },
            )
        )

        try:
            session = ensure_global_session(
                _REMOVAL_MODEL_LOOKUP.get(self.options.model_key, DEFAULT_MODEL_NAME)
            )
        except Exception as exc:  # pragma: no cover - defensive guard for UI errors.
            LOGGER.exception("Unable to prepare the selected model for batch processing.")
            self.events.put(
                (
                    "batch_error",
                    {
                        "message": "Unable to prepare the selected model.",
                        "exception": exc,
                    },
                )
            )
            return

        processed = 0
        durations: list[float] = []
        format_spec = get_output_format_spec(self.options.output_format)

        for index, source in enumerate(sources, start=1):
            if self.cancel_event.is_set():
                self.events.put(
                    (
                        "batch_cancelled",
                        {
                            "processed": processed,
                            "total": total,
                        },
                    )
                )
                return

            relative = source.relative_to(self.options.folder)
            destination = format_spec.normalise_filename(output_root / relative)
            destination.parent.mkdir(parents=True, exist_ok=True)

            if self.options.skip_existing and destination.exists():
                processed += 1
                durations.append(0.0)
                self.events.put(
                    (
                        "image_skipped",
                        {
                            "source": str(source),
                            "destination": str(destination),
                            "index": index,
                            "total": total,
                        },
                    )
                )
                self._update_progress(processed, total, durations)
                continue

            self.events.put(
                (
                    "image_start",
                    {
                        "source": str(source),
                        "index": index,
                        "total": total,
                    },
                )
            )

            start_time = time.perf_counter()
            try:
                result = remove_bg_file(
                    source,
                    destination,
                    session=session,
                    retain_image=False,
                    save_to_disk=True,
                    **self.options.as_kwargs(),
                )
            except Exception as exc:  # pragma: no cover - best effort logging.
                LOGGER.exception("Error while processing %s during batch run", source)
                durations.append(time.perf_counter() - start_time)
                processed += 1
                self.events.put(
                    (
                        "image_error",
                        {
                            "source": str(source),
                            "error": str(exc),
                            "index": index,
                            "total": total,
                        },
                    )
                )
                self._update_progress(processed, total, durations)
                continue

            durations.append(time.perf_counter() - start_time)
            processed += 1
            self.events.put(
                (
                    "image_complete",
                    {
                        "source": str(source),
                        "destination": str(result.path_out) if result.path_out else str(destination),
                        "index": index,
                        "total": total,
                        "success": result.success,
                    },
                )
            )
            self._update_progress(processed, total, durations)

        self.events.put(
            (
                "batch_complete",
                {
                    "processed": processed,
                    "total": total,
                    "output": str(output_root),
                },
            )
        )

    def _update_progress(
        self,
        processed: int,
        total: int,
        durations: list[float],
    ) -> None:
        """Send progress information to the UI event queue."""

        remaining = max(total - processed, 0)
        average = sum(durations) / len(durations) if durations else 0.0
        eta = average * remaining
        self.events.put(
            (
                "progress",
                {
                    "processed": processed,
                    "total": total,
                    "eta": eta,
                },
            )
        )


class BackgroundRemoverApp(tb.Window):
    """Desktop interface for running background removal locally."""

    def __init__(self) -> None:
        super().__init__(title="Background Remover", themename="flatly")
        self.app_style = tb.Style("flatly")
        self.geometry("1100x720")
        self.minsize(960, 640)
        self._theme_dark = False
        self._folder_worker: FolderProcessor | None = None
        self._folder_cancel_event = threading.Event()
        self._single_worker: threading.Thread | None = None
        self.event_queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        self._output_folder_override: Path | None = None
        self._image_preview: ImageTk.PhotoImage | None = None
        self._result_preview: ImageTk.PhotoImage | None = None
        self.alpha_spinboxes: list[tb.Spinbox] = []
        self.provider_badge: tb.Label | None = None
        self.provider_tooltip: ToolTip | None = None
        self._provider_status = tb.StringVar(value="[ CPU ]")
        self._provider_refresh_thread: threading.Thread | None = None
        self._pending_provider_key: str | None = None

        self._create_variables()
        self._build_ui()
        self.after(100, self._process_event_queue)
        self._refresh_provider_badge()
        self._prepare_model_runtime(initial=True)

    def _create_variables(self) -> None:
        """Initialise Tkinter variables used across the UI."""

        self.mode_var = tb.StringVar(value="single")
        self.selected_file = tb.StringVar(value="No image selected")
        self.selected_folder = tb.StringVar(value="No folder selected")
        self.selected_output_folder = tb.StringVar(
            value="Select a folder to choose the output location."
        )
        self.folder_summary = tb.StringVar(value="")
        self.progress_var = tb.DoubleVar(value=0.0)
        self.progress_text = tb.StringVar(value="Waiting to start")
        self.eta_text = tb.StringVar(value="")
        self.current_image_name = tb.StringVar(value="")
        self.output_message = tb.StringVar(value="")
        self.output_folder_var = tb.StringVar(value="./output")
        self.alpha_matting_var = tb.BooleanVar(value=DEFAULT_ALPHA_MATTING_VALUES["alpha_matting"])
        self.am_foreground_var = tb.IntVar(value=DEFAULT_ALPHA_MATTING_VALUES["am_foreground"])
        self.am_background_var = tb.IntVar(value=DEFAULT_ALPHA_MATTING_VALUES["am_background"])
        self.am_erode_var = tb.IntVar(value=DEFAULT_ALPHA_MATTING_VALUES["am_erode"])
        self.feather_var = tb.IntVar(value=DEFAULT_TUNE_VALUES["feather_radius"])
        self.colorkey_var = tb.IntVar(value=DEFAULT_TUNE_VALUES["colorkey_tolerance"])
        self.use_colorkey_var = tb.BooleanVar(value=DEFAULT_TUNE_VALUES["use_colorkey_fallback"])
        self.output_format_var = tb.StringVar(value=DEFAULT_OUTPUT_FORMAT)
        default_model_dir = get_models_directory()
        default_display = str(default_model_dir)
        if default_model_dir == Path("./models"):
            default_display = "./models"
        self.model_folder_var = tb.StringVar(value=default_display)
        self.model_var = tb.StringVar(value="general")
        self.recursive_var = tb.BooleanVar(value=False)
        self.skip_existing_var = tb.BooleanVar(value=True)

    def _build_ui(self) -> None:
        """Construct the window layout and widgets."""

        container = tb.Frame(self, padding=20)
        container.pack(fill=BOTH, expand=True)

        header = tb.Frame(container)
        header.pack(fill=BOTH, expand=False)

        self.app_style.configure(
            "StatusBadge.TLabel",
            padding=(10, 4),
            font=("Segoe UI", 9, "bold"),
        )

        title = tb.Label(
            header,
            text="Background Remover",
            font=("Segoe UI", 20, "bold"),
        )
        title.pack(side=LEFT)

        subtitle = tb.Label(
            header,
            text="Process single images or entire folders with live previews.",
            bootstyle="secondary",
        )
        subtitle.pack(side=LEFT, padx=10, pady=6)

        self.provider_badge = tb.Label(
            header,
            textvariable=self._provider_status,
            style="StatusBadge.TLabel",
            bootstyle="secondary",
        )
        self.provider_badge.pack(side=RIGHT, padx=(0, 12))
        self.provider_tooltip = create_tooltip(
            self.provider_badge,
            "Active providers: CPUExecutionProvider",
        )

        theme_button = tb.Button(
            header,
            text="Toggle Theme",
            command=self._toggle_theme,
            bootstyle="secondary-outline",
        )
        theme_button.pack(side=RIGHT)
        create_tooltip(theme_button, "Switch between light and dark themes.")

        mode_frame = tb.Frame(container, padding=(0, 20, 0, 10))
        mode_frame.pack(fill=BOTH, expand=False)

        tb.Label(mode_frame, text="Processing mode", font=("Segoe UI", 12, "bold")).pack(
            anchor=W
        )

        choices = tb.Frame(mode_frame)
        choices.pack(anchor=W, pady=8)
        single_radio = tb.Radiobutton(
            choices,
            text="Single Image",
            value="single",
            variable=self.mode_var,
            command=self._on_mode_changed,
            bootstyle="success-toolbutton",
        )
        single_radio.pack(side=LEFT, padx=(0, 8))
        create_tooltip(single_radio, "Process a single image file.")

        folder_radio = tb.Radiobutton(
            choices,
            text="Folder Batch",
            value="folder",
            variable=self.mode_var,
            command=self._on_mode_changed,
            bootstyle="info-toolbutton",
        )
        folder_radio.pack(side=LEFT, padx=(0, 8))
        create_tooltip(folder_radio, "Process all supported images inside a folder.")

        body = tb.PanedWindow(container, orient="horizontal")
        body.pack(fill=BOTH, expand=True)

        left_panel = tb.Frame(body, padding=10)
        right_panel = tb.Frame(body, padding=10)
        body.add(left_panel, weight=2)
        body.add(right_panel, weight=1)

        # Shared options reside in the right panel.
        self._build_options_panel(right_panel)

        # Mode specific controls.
        self.single_frame = tb.Labelframe(left_panel, text="Single Image Workflow", padding=15)
        self.single_frame.pack(fill=BOTH, expand=True)
        self._build_single_controls(self.single_frame)

        self.folder_frame = tb.Labelframe(left_panel, text="Folder Workflow", padding=15)
        self.folder_frame.pack(fill=BOTH, expand=True)
        self._build_folder_controls(self.folder_frame)
        self.folder_frame.forget()

    def _build_options_panel(self, parent: tb.Frame) -> None:
        """Create the panel that exposes shared processing options."""

        info_label = tb.Label(
            parent,
            text=(
                "Tune how backgrounds are removed. These settings match the "
                "advanced options in the web interface."
            ),
            wraplength=280,
            bootstyle="secondary",
            justify=LEFT,
        )
        info_label.pack(anchor=W, pady=(0, 10))

        model_frame = tb.Labelframe(parent, text="Model Settings", padding=10)
        model_frame.pack(fill=BOTH, pady=(0, 10))

        folder_row = tb.Frame(model_frame)
        folder_row.pack(fill=BOTH, pady=(0, 5))

        tb.Label(folder_row, text="Model folder:").pack(side=LEFT, padx=(0, 6))
        folder_entry = tb.Entry(folder_row, textvariable=self.model_folder_var, width=34)
        folder_entry.pack(side=LEFT, fill=BOTH, expand=True)
        folder_entry.bind("<FocusOut>", lambda _event: self._prepare_model_runtime())
        folder_entry.bind("<Return>", lambda _event: self._prepare_model_runtime(announce_ready=True))
        browse_button = tb.Button(folder_row, text="Browse…", command=self._choose_model_folder)
        browse_button.pack(side=LEFT, padx=(6, 0))

        ToolTip(
            model_frame,
            (
                "Choose where ONNX models are stored or downloaded. Useful if you want "
                "to keep models on a separate drive or shared folder."
            ),
        )

        format_label = tb.Label(parent, text="Output format", font=("Segoe UI", 10, "bold"))
        format_label.pack(anchor=W)
        format_tooltip = (
            "Choose how processed images are saved. PNG keeps transparency (default), JPG fills the "
            "background, and WEBP balances quality with smaller files."
        )
        create_tooltip(format_label, format_tooltip)

        formats = [f"{spec.label}" for spec in OUTPUT_FORMATS]
        keys = [spec.key for spec in OUTPUT_FORMATS]
        try:
            initial_index = keys.index(self.output_format_var.get())
        except ValueError:
            initial_index = 0
            self.output_format_var.set(keys[0])
        self.output_format_display = tb.StringVar(value=formats[initial_index])
        self.format_combo = tb.Combobox(
            parent,
            values=formats,
            state="readonly",
            textvariable=self.output_format_display,
        )
        self.format_combo.current(initial_index)
        self.format_combo.pack(fill=BOTH, pady=5)
        self.format_combo.bind("<<ComboboxSelected>>", self._on_format_selected)
        create_tooltip(self.format_combo, format_tooltip)

        model_label = tb.Label(parent, text="Removal model", font=("Segoe UI", 10, "bold"))
        model_label.pack(anchor=W, pady=(10, 0))
        model_tooltip = (
            "Choose which ONNX model handles background removal. Larger models capture more detail "
            "but take longer to run; lighter models are faster with slightly softer edges."
        )
        create_tooltip(model_label, model_tooltip)

        model_values = [option["label"] for option in REMOVAL_MODEL_OPTIONS]
        model_keys = [option["key"] for option in REMOVAL_MODEL_OPTIONS]
        self.model_combo = tb.Combobox(
            parent,
            values=model_values,
            state="readonly",
        )
        try:
            idx = model_keys.index(self.model_var.get())
        except ValueError:
            idx = 0
            self.model_var.set(model_keys[0])
        self.model_display = tb.StringVar(value=model_values[idx])
        self.model_combo.configure(textvariable=self.model_display)
        self.model_combo.current(idx)
        self.model_combo.pack(fill=BOTH, pady=5)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)
        create_tooltip(self.model_combo, model_tooltip)

        alpha_check = tb.Checkbutton(
            parent,
            text="Enable alpha matting",
            variable=self.alpha_matting_var,
            command=self._sync_alpha_controls,
        )
        alpha_check.pack(anchor=W, pady=(15, 5))
        create_tooltip(
            alpha_check,
            (
                "Enables an extra refinement pass that preserves fine details like hair or fur. "
                "Disable for faster processing."
            ),
        )

        alpha_frame = tb.Frame(parent)
        alpha_frame.pack(fill=BOTH, pady=5)

        self._add_labeled_spinbox(
            alpha_frame,
            label="Foreground threshold",
            variable=self.am_foreground_var,
            from_=0,
            to=255,
            tooltip=(
                "Sets how strongly the foreground is preserved when alpha matting runs (0–255; "
                "default 240)."
            ),
            collector=self.alpha_spinboxes,
        )
        self._add_labeled_spinbox(
            alpha_frame,
            label="Background threshold",
            variable=self.am_background_var,
            from_=0,
            to=255,
            tooltip=(
                "Controls how aggressively the background is removed during matting (0–255; default "
                "10)."
            ),
            collector=self.alpha_spinboxes,
        )
        self._add_labeled_spinbox(
            alpha_frame,
            label="Erode structure size",
            variable=self.am_erode_var,
            from_=0,
            to=255,
            tooltip=(
                "Adjusts the erosion kernel used before refinement. Higher values trim more edge "
                "detail (0–255; default 10)."
            ),
            collector=self.alpha_spinboxes,
        )

        tuning_frame = tb.Frame(parent)
        tuning_frame.pack(fill=BOTH, pady=(15, 0))

        self._add_labeled_spinbox(
            tuning_frame,
            label="Feather radius",
            variable=self.feather_var,
            from_=0,
            to=50,
            tooltip=(
                "Adjusts how soft subject borders appear. Higher values blend edges across a wider "
                "radius (0–50; default 3)."
            ),
        )
        self._add_labeled_spinbox(
            tuning_frame,
            label="Colour key tolerance",
            variable=self.colorkey_var,
            from_=0,
            to=60,
            tooltip=(
                "Useful for solid backgrounds such as green screens. Higher values remove more "
                "similar colours (0–255; default 14)."
            ),
        )

        colorkey_check = tb.Checkbutton(
            parent,
            text="Use colour key fallback",
            variable=self.use_colorkey_var,
        )
        colorkey_check.pack(anchor=W, pady=(5, 0))
        create_tooltip(
            colorkey_check,
            (
                "Applies a colour-key helper when matting is disabled so solid backgrounds still "
                "get removed. Enabled by default."
            ),
        )

        recursive_check = tb.Checkbutton(
            parent,
            text="Include sub-folders",
            variable=self.recursive_var,
        )
        recursive_check.pack(anchor=W, pady=(15, 0))
        create_tooltip(
            recursive_check,
            "Process every supported image within nested folders—ideal for large photo libraries.",
        )

        skip_check = tb.Checkbutton(
            parent,
            text="Skip existing outputs",
            variable=self.skip_existing_var,
        )
        skip_check.pack(anchor=W, pady=5)
        create_tooltip(
            skip_check,
            "Avoid reprocessing files that already have background-free versions on disk.",
        )
        self._sync_alpha_controls()

    def _refresh_provider_badge(self, *, model_key: str | None = None) -> None:
        """Update the hardware provider badge asynchronously."""

        if self.provider_badge is None:
            return
        selected_key = model_key or self.model_var.get()
        self._pending_provider_key = selected_key
        model_name = _REMOVAL_MODEL_LOOKUP.get(selected_key, DEFAULT_MODEL_NAME)

        def worker(target_key: str, target_model: str) -> None:
            """Load provider information for ``target_model`` in a thread."""

            providers: tuple[str, ...]
            try:
                session = ensure_global_session(target_model)
                providers = tuple(getattr(session, "providers_available", ()))
                if not providers:
                    inner = getattr(session, "inner", None)
                    if inner is not None and hasattr(inner, "get_providers"):
                        providers = tuple(inner.get_providers())
                if not providers:
                    providers = ("CPUExecutionProvider",)
            except Exception:
                LOGGER.exception("Failed to detect ONNX providers for %s", target_model)
                providers = ("CPUExecutionProvider",)

            self.after(0, lambda: self._apply_provider_badge(target_key, providers))

        thread = threading.Thread(target=worker, args=(selected_key, model_name), daemon=True)
        thread.start()
        self._provider_refresh_thread = thread

    def _apply_provider_badge(
        self, model_key: str, providers: Iterable[str] | tuple[str, ...]
    ) -> None:
        """Apply provider state to the badge when the UI thread is ready."""

        if model_key != self._pending_provider_key or self.provider_badge is None:
            return

        provider_list = tuple(providers) if providers else ("CPUExecutionProvider",)
        use_gpu = "CUDAExecutionProvider" in provider_list
        status_text = "[ GPU ]" if use_gpu else "[ CPU ]"
        bootstyle = "success" if use_gpu else "secondary"

        self._provider_status.set(status_text)
        self.provider_badge.configure(bootstyle=bootstyle)

        tooltip_text = "Active providers: " + ", ".join(provider_list)
        if self.provider_tooltip is None:
            self.provider_tooltip = create_tooltip(self.provider_badge, tooltip_text)
        else:
            updated = False
            for attr_name in ("configure", "config"):
                updater = getattr(self.provider_tooltip, attr_name, None)
                if callable(updater):
                    try:
                        updater(text=tooltip_text)
                        updated = True
                        break
                    except Exception:
                        continue
            if not updated:
                try:
                    self.provider_tooltip.text = tooltip_text
                    updated = True
                except Exception:
                    updated = False
            if not updated:
                self.provider_tooltip = create_tooltip(self.provider_badge, tooltip_text)

    def _add_labeled_spinbox(
        self,
        parent: tb.Frame,
        *,
        label: str,
        variable: tb.IntVar,
        from_: int,
        to: int,
        tooltip: str,
        collector: list[tb.Spinbox] | None = None,
    ) -> None:
        """Create a labeled spinbox with a tooltip for numeric settings."""

        frame = tb.Frame(parent)
        frame.pack(fill=BOTH, pady=3)
        lbl = tb.Label(frame, text=label)
        lbl.pack(anchor=W)
        create_tooltip(lbl, tooltip)
        spin = tb.Spinbox(
            frame,
            from_=from_,
            to=to,
            textvariable=variable,
            increment=1,
            width=10,
        )
        spin.pack(anchor=W)
        create_tooltip(spin, tooltip)
        if collector is not None:
            collector.append(spin)

    def _build_single_controls(self, parent: tb.Frame) -> None:
        """Create widgets specific to single-image processing."""

        description = tb.Label(
            parent,
            text="Choose an image and apply the configured options.",
            bootstyle="secondary",
        )
        description.pack(anchor=W, pady=(0, 10))

        file_row = tb.Frame(parent)
        file_row.pack(fill=BOTH, pady=5)

        self.file_button = tb.Button(
            file_row,
            text="Select image",
            command=self._choose_single_image,
            bootstyle="primary-outline",
        )
        self.file_button.pack(side=LEFT)
        create_tooltip(self.file_button, "Browse for a single image to process.")

        file_label = tb.Label(
            file_row,
            textvariable=self.selected_file,
            wraplength=420,
            justify=LEFT,
        )
        file_label.pack(side=LEFT, padx=10)

        output_frame = tb.Labelframe(parent, text="Output Settings", padding=10)
        output_frame.pack(fill=BOTH, pady=5)

        tb.Label(output_frame, text="Output Folder:").pack(side=LEFT, padx=5)
        output_entry = tb.Entry(output_frame, textvariable=self.output_folder_var, width=40)
        output_entry.pack(side=LEFT, padx=5)
        output_button = tb.Button(
            output_frame,
            text="Browse…",
            command=self._choose_single_output_folder,
            bootstyle="secondary-outline",
        )
        output_button.pack(side=LEFT)
        ToolTip(
            output_button,
            (
                "Choose where processed images are saved. If none is selected, "
                "results are written to './output' beside the application."
            ),
        )

        self.single_preview = tb.Label(parent)
        self.single_preview.pack(pady=10)

        self.single_status = tb.Label(parent, textvariable=self.output_message, bootstyle="info")
        self.single_status.pack(anchor=W, pady=5)

        action_row = tb.Frame(parent)
        action_row.pack(fill=BOTH, pady=10)

        self.process_button = tb.Button(
            action_row,
            text="Remove background",
            command=self._process_single_image,
            bootstyle="success",
        )
        self.process_button.pack(side=LEFT)
        create_tooltip(
            self.process_button,
            "Generate a background-free version of the selected image.",
        )

    def _build_folder_controls(self, parent: tb.Frame) -> None:
        """Create widgets for folder-based processing."""

        description = tb.Label(
            parent,
            text=(
                "Process every supported image in a folder. Results are saved to an "
                '"output" folder beside your originals.'
            ),
            bootstyle="secondary",
            wraplength=420,
            justify=LEFT,
        )
        description.pack(anchor=W, pady=(0, 10))

        choose_row = tb.Frame(parent)
        choose_row.pack(fill=BOTH, pady=5)

        self.folder_button = tb.Button(
            choose_row,
            text="Select folder",
            command=self._choose_folder,
            bootstyle="primary-outline",
        )
        self.folder_button.pack(side=LEFT)
        create_tooltip(
            self.folder_button,
            "Choose the folder that should be processed in batch mode.",
        )

        folder_label = tb.Label(
            choose_row,
            textvariable=self.selected_folder,
            wraplength=420,
            justify=LEFT,
        )
        folder_label.pack(side=LEFT, padx=10)

        output_row = tb.Frame(parent)
        output_row.pack(fill=BOTH, pady=5)

        self.output_button = tb.Button(
            output_row,
            text="Select output folder",
            command=self._choose_output_folder,
            bootstyle="secondary-outline",
        )
        self.output_button.pack(side=LEFT)
        create_tooltip(
            self.output_button,
            (
                "Choose where processed images should be saved. Leave unset to use the "
                "default output folder beside the selected input folder."
            ),
        )

        output_label = tb.Label(
            output_row,
            textvariable=self.selected_output_folder,
            wraplength=420,
            justify=LEFT,
        )
        output_label.pack(side=LEFT, padx=10)

        summary_label = tb.Label(parent, textvariable=self.folder_summary, bootstyle="info")
        summary_label.pack(anchor=W, pady=5)

        progress_bar = tb.Progressbar(parent, variable=self.progress_var, maximum=1.0)
        progress_bar.pack(fill=BOTH, pady=10)

        progress_info = tb.Frame(parent)
        progress_info.pack(fill=BOTH)

        tb.Label(progress_info, textvariable=self.progress_text).pack(side=LEFT)
        tb.Label(progress_info, textvariable=self.eta_text, bootstyle="secondary").pack(side=RIGHT)

        current_frame = tb.Frame(parent)
        current_frame.pack(fill=BOTH, pady=10)

        tb.Label(current_frame, text="Current image:").pack(anchor=W)
        tb.Label(current_frame, textvariable=self.current_image_name, bootstyle="secondary").pack(anchor=W)
        self.batch_preview = tb.Label(current_frame)
        self.batch_preview.pack(pady=6)

        action_row = tb.Frame(parent)
        action_row.pack(fill=BOTH, pady=10)

        self.start_batch_button = tb.Button(
            action_row,
            text="Start batch",
            command=self._start_batch,
            bootstyle="success",
        )
        self.start_batch_button.pack(side=LEFT)

        self.cancel_batch_button = tb.Button(
            action_row,
            text="Cancel batch",
            command=self._cancel_batch,
            bootstyle="danger-outline",
            state="disabled",
        )
        self.cancel_batch_button.pack(side=LEFT, padx=10)
        create_tooltip(
            self.cancel_batch_button,
            "Stop processing after the current image completes.",
        )

        log_label = tb.Label(parent, text="Status log", font=("Segoe UI", 10, "bold"))
        log_label.pack(anchor=W)

        self.log_console = ScrolledText(parent, height=10, padding=5, state="disabled")
        self.log_console.pack(fill=BOTH, expand=True)
        self._configure_log_console()

    def _get_style_color(self, key: str, fallback: str) -> str:
        """Return a ttkbootstrap theme colour for ``key`` or ``fallback``."""

        colors = getattr(self.app_style, "colors", None)
        if isinstance(colors, dict):
            return colors.get(key, fallback)
        if colors is not None and hasattr(colors, key):
            return getattr(colors, key)
        return fallback

    def _configure_log_console(self) -> None:
        """Prepare the status log for styled, read-only updates."""

        text_widget = getattr(self.log_console, "text", None)
        if text_widget is None:
            return

        default_color = text_widget.cget("foreground")
        success_color = self._get_style_color("success", "#198754")
        error_color = self._get_style_color("danger", "#dc3545")

        text_widget.configure(state="normal")
        text_widget.delete("1.0", END)
        text_widget.tag_configure("info", foreground=default_color)
        text_widget.tag_configure("success", foreground=success_color)
        text_widget.tag_configure("error", foreground=error_color)
        text_widget.configure(state="disabled")

    @safe_callback
    def _on_mode_changed(self) -> None:
        """Switch between single and folder workflows."""

        mode = self.mode_var.get()
        if mode == "single":
            self.folder_frame.forget()
            self.single_frame.pack(fill=BOTH, expand=True)
            self.file_button.configure(state="normal")
            self.process_button.configure(state="normal")
        else:
            self.single_frame.forget()
            self.folder_frame.pack(fill=BOTH, expand=True)
            self.file_button.configure(state="disabled")
            self.process_button.configure(state="disabled")

    @safe_callback
    def _toggle_theme(self) -> None:
        """Toggle between the light and dark ttkbootstrap themes."""

        self._theme_dark = not self._theme_dark
        theme = "darkly" if self._theme_dark else "flatly"
        self.app_style.theme_use(theme)

    @safe_callback
    def _on_format_selected(self, _event: Any) -> None:
        """Synchronise the output format variable with the dropdown."""

        index = self.format_combo.current()
        if 0 <= index < len(OUTPUT_FORMATS):
            self.output_format_var.set(OUTPUT_FORMATS[index].key)

    @safe_callback
    def _on_model_selected(self, _event: Any) -> None:
        """Synchronise the chosen model key."""

        index = self.model_combo.current()
        keys = [option["key"] for option in REMOVAL_MODEL_OPTIONS]
        if 0 <= index < len(keys):
            self.model_var.set(keys[index])
            self._refresh_provider_badge(model_key=keys[index])

    @safe_callback
    def _sync_alpha_controls(self) -> None:
        """Enable or disable alpha matting controls based on the checkbox."""

        state = "normal" if self.alpha_matting_var.get() else "disabled"
        for widget in self.alpha_spinboxes:
            widget.configure(state=state)

    @safe_callback
    def _choose_model_folder(self) -> None:
        """Prompt the user to choose where models are stored or downloaded."""

        folder = filedialog.askdirectory(title="Select Model Storage Folder")
        if folder:
            self.model_folder_var.set(folder)
            self.log_status(f"🧠 Model folder set to: {folder}")
            self._prepare_model_runtime(announce_ready=True)
        else:
            self.log_status("⚠️ No model folder selected; using ./models", color="yellow")
            if not self.model_folder_var.get().strip():
                self.model_folder_var.set("./models")
                self._prepare_model_runtime()

    @safe_callback
    def _choose_single_image(self) -> None:
        """Prompt the user to choose an image file."""

        from tkinter import filedialog

        patterns = ["*.{}".format(ext.lstrip(".")) for ext in SUPPORTED_EXTENSIONS]
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[
                ("Image files", " ".join(patterns)),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.selected_file.set(path)
        self._display_preview(Path(path), self.single_preview, max_size=400)
        self.output_message.set("Ready to process.")

    @safe_callback
    def _process_single_image(self) -> None:
        """Run background removal for the selected image."""

        if self._single_worker and self._single_worker.is_alive():
            Messagebox.show_warning(
                "Processing already in progress. Please wait until it finishes.",
                "Background Remover",
            )
            return

        path_text = self.selected_file.get()
        path = Path(path_text)
        if not path.exists():
            Messagebox.show_error("Please choose an image before processing.", "Background Remover")
            return

        if not self._prepare_model_runtime():
            return

        def worker() -> None:
            self.output_message.set("Processing…")
            options = self._build_common_options()
            try:
                output_dir = self._resolve_single_output_directory()
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                LOGGER.exception("Failed to prepare output directory for %s", path)
                self.output_message.set("Failed to process image.")
                self.log_status(f"❌ Failed to prepare output directory: {exc}", color="red")
                return

            destination = output_dir
            result = None
            try:
                result = remove_bg_file(
                    path,
                    destination,
                    retain_image=True,
                    save_to_disk=True,
                    **options["kwargs"],
                )
            except (FileNotFoundError, OSError, RuntimeError) as error:
                # Surface model loading and execution issues to the user with
                # actionable messaging while persisting the full traceback for
                # diagnostics.
                traceback_text = traceback.format_exc()
                log_path = _write_traceback_to_log(traceback_text)
                print(traceback_text, file=sys.stderr)
                LOGGER.error("Model error while processing %s", path, exc_info=error)
                reason = str(error) or error.__class__.__name__
                self.output_message.set(f"Model error: {reason}")
                self.log_status(f"❌ Model error — {reason}", color="red")
                self.log_status(f"See {log_path} for details.", color="red")
                return
            except Exception as exc:  # pragma: no cover - UI level reporting.
                LOGGER.exception("Failed to process single image %s", path)
                traceback_text = traceback.format_exc()
                _write_traceback_to_log(traceback_text)
                print(traceback_text, file=sys.stderr)
                self.output_message.set("Failed to process image.")
                self._append_log(f"Error processing {path.name}: {exc}")
                return
            finally:
                if result is not None and result.image is not None:
                    result.image.close()

            if result.success and result.path_out:
                self.output_message.set(f"Saved to {result.path_out}")
                self._display_preview(
                    result.path_out,
                    self.single_preview,
                    max_size=400,
                    store_result=True,
                )
                self._append_log(f"✔ Saved {result.path_out}")
            else:
                self.output_message.set("Processing completed without an output file.")

        self._single_worker = threading.Thread(target=worker, daemon=True)
        self._single_worker.start()

    def _build_common_options(self) -> dict[str, Any]:
        """Return keyword arguments shared by single and batch workflows."""

        model_name = _REMOVAL_MODEL_LOOKUP.get(self.model_var.get(), DEFAULT_MODEL_NAME)
        format_spec = get_output_format_spec(self.output_format_var.get())
        kwargs = {
            "output_format": format_spec.key,
            "model_name": model_name,
            "alpha_matting": self.alpha_matting_var.get(),
            "am_foreground": int(self.am_foreground_var.get()),
            "am_background": int(self.am_background_var.get()),
            "am_erode": int(self.am_erode_var.get()),
            "feather_radius": int(self.feather_var.get()),
            "use_colorkey_fallback": self.use_colorkey_var.get(),
            "colorkey_tolerance": int(self.colorkey_var.get()),
        }
        return {"output": None, "kwargs": kwargs}

    def _apply_model_directory(self) -> Path | None:
        """Synchronise the model storage directory with the backend module."""

        raw_value = self.model_folder_var.get().strip()
        if not raw_value:
            raw_value = "./models"
        self.model_folder_var.set(raw_value)
        try:
            directory = set_models_directory(raw_value)
        except Exception as error:
            self._report_model_error("Failed to configure model folder", error)
            return None
        display_value = "./models" if directory == Path("./models") else str(directory)
        self.model_folder_var.set(display_value)
        return directory

    def _prepare_model_runtime(
        self, *, initial: bool = False, announce_ready: bool = False
    ) -> bool:
        """Ensure the requested ONNX models are ready for use."""

        directory = self._apply_model_directory()
        if directory is None:
            return False
        model_name = _REMOVAL_MODEL_LOOKUP.get(self.model_var.get(), DEFAULT_MODEL_NAME)
        try:
            ensure_models_downloaded(prefetch={model_name})
            ensure_global_session(model_name)
        except Exception as error:
            self._report_model_error("Failed to prepare models", error)
            return False
        if announce_ready and not initial:
            self.log_status(f"✅ Models ready in {self.model_folder_var.get()}")
        return True

    def _report_model_error(self, summary: str, error: Exception) -> None:
        """Log ``error`` to the UI and persist the traceback for debugging."""

        traceback_text = traceback.format_exc()
        log_path = _write_traceback_to_log(traceback_text)
        LOGGER.error(summary, exc_info=error)
        reason = str(error) or error.__class__.__name__
        self.log_status(f"❌ {summary}: {reason}", color="red")
        self.log_status(f"See {log_path} for details.", color="red")

    def _display_preview(
        self,
        path: Path,
        target: tb.Label,
        *,
        max_size: int = 280,
        store_result: bool = False,
    ) -> None:
        """Display a thumbnail preview for ``path`` in ``target``."""

        try:
            with Image.open(path) as image:
                image.thumbnail((max_size, max_size))
                photo = ImageTk.PhotoImage(image)
        except Exception:
            return
        target.configure(image=photo)
        target.image = photo
        if store_result:
            self._result_preview = photo
        else:
            self._image_preview = photo

    @safe_callback
    def _choose_folder(self) -> None:
        """Prompt the user to select a folder for batch processing."""

        from tkinter import filedialog

        path = filedialog.askdirectory(title="Choose a folder")
        if not path:
            return
        folder = Path(path)
        self.selected_folder.set(str(folder))
        self._output_folder_override = None
        self._update_folder_summary()
        self._append_log(f"Folder selected: {folder}")

    @safe_callback
    def _choose_output_folder(self) -> None:
        """Prompt the user to choose a custom output destination."""

        from tkinter import filedialog

        folder_text = self.selected_folder.get()
        folder = Path(folder_text)
        if not folder.exists():
            Messagebox.show_warning(
                "Select a folder to process before choosing an output destination.",
                "Background Remover",
            )
            return

        path = filedialog.askdirectory(title="Choose an output folder")
        if not path:
            return

        self._output_folder_override = Path(path)
        self._append_log(f"Output folder selected: {self._output_folder_override}")
        self._update_folder_summary()

    @safe_callback
    def _choose_single_output_folder(self) -> None:
        """Prompt the user to select an output directory for single processing."""

        from tkinter import filedialog

        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder_var.set(folder)
            self.log_status(f"📁 Output folder set to: {folder}")
        else:
            self.log_status("⚠️ No output folder selected; using ./output", color="yellow")

    def _resolve_single_output_directory(self) -> Path:
        """Return the resolved directory for single-image exports."""

        raw_value = (self.output_folder_var.get() or "").strip()
        if not raw_value:
            raw_value = "./output"
            self.output_folder_var.set(raw_value)
        candidate = Path(raw_value).expanduser()
        if not candidate.is_absolute():
            candidate = _runtime_directory() / candidate
        return candidate

    def _scan_folder(self, folder: Path, recursive: bool) -> Iterable[Path]:
        """Yield supported image files inside ``folder``."""

        iterator: Iterable[Path]
        if recursive:
            iterator = folder.rglob("*")
        else:
            iterator = folder.iterdir()
        # To add support for new file types, extend ``SUPPORTED_EXTENSIONS``
        # in :mod:`bg_removal`. The GUI automatically honours that list when
        # filtering candidate files.
        for path in iterator:
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path

    @safe_callback
    def _update_folder_summary(self) -> None:
        """Refresh the folder summary and output location details."""

        folder_text = self.selected_folder.get()
        try:
            folder = Path(folder_text)
        except Exception:
            self.folder_summary.set("")
            self.selected_output_folder.set(
                "Select a folder to choose the output location."
            )
            return

        if not folder.exists():
            self.folder_summary.set("")
            self.selected_output_folder.set(
                "Select a folder to choose the output location."
            )
            return

        sources = list(self._scan_folder(folder, self.recursive_var.get()))
        output_root = self._output_folder_override or (folder.parent / "output")
        self.folder_summary.set(
            f"Found {len(sources)} image(s) · Output: {output_root}"
        )
        if self._output_folder_override:
            self.selected_output_folder.set(f"Custom: {output_root}")
        else:
            self.selected_output_folder.set(f"Default: {output_root}")

    @safe_callback
    def _start_batch(self) -> None:
        """Launch a worker thread for batch processing."""
        if self._folder_worker and self._folder_worker.is_alive():
            Messagebox.show_info("Batch already running.", "Background Remover")
            return

        folder_text = self.selected_folder.get()
        folder = Path(folder_text)
        if not folder.exists():
            Messagebox.show_error(
                "Please choose a folder before starting the batch.",
                "Background Remover",
            )
            return

        if not self._prepare_model_runtime():
            return

        output_root = self._output_folder_override or (folder.parent / "output")
        self._update_folder_summary()

        options = FolderTaskOptions(
            folder=folder,
            output_root=output_root,
            output_format=self.output_format_var.get(),
            model_key=self.model_var.get(),
            recursive=self.recursive_var.get(),
            skip_existing=self.skip_existing_var.get(),
            alpha_matting=self.alpha_matting_var.get(),
            am_foreground=int(self.am_foreground_var.get()),
            am_background=int(self.am_background_var.get()),
            am_erode=int(self.am_erode_var.get()),
            feather_radius=int(self.feather_var.get()),
            use_colorkey_fallback=self.use_colorkey_var.get(),
            colorkey_tolerance=int(self.colorkey_var.get()),
        )

        # reset UI state
        self.progress_var.set(0.0)
        self.progress_text.set("Preparing…")
        self.eta_text.set("")
        self.current_image_name.set("")
        self.batch_preview.configure(image="")
        self.batch_preview.image = None

        self._clear_log()

        self._folder_cancel_event.clear()
        self.cancel_batch_button.configure(state="normal")
        self.start_batch_button.configure(state="disabled")
        self._append_log("Batch started.")

        # 🚀 Start the background worker
        self._folder_worker = FolderProcessor(
            options, self.event_queue, self._folder_cancel_event
        )
        self._folder_worker.start()

    @safe_callback
    def _cancel_batch(self) -> None:
        """Signal the background worker to cancel processing."""

        if self._folder_worker and self._folder_worker.is_alive():
            self._folder_cancel_event.set()
            self._append_log("Cancellation requested. Finishing current image…")
            self.cancel_batch_button.configure(state="disabled")

    @safe_callback
    def _process_event_queue(self) -> None:
        """Handle events emitted from worker threads."""

        try:
            while True:
                event, payload = self.event_queue.get_nowait()
                self._handle_event(event, payload)
                try:
                    self.update_idletasks()
                except Exception:
                    pass
        except queue.Empty:
            pass
        finally:
            self.after(100, self._process_event_queue)

    def _handle_event(self, event: str, payload: dict[str, Any]) -> None:
        """Respond to a single event emitted by a worker."""

        if event == "batch_start":
            total = payload.get("total", 0)
            folder = payload.get("folder", "")
            self.progress_text.set(f"Processing {total} image(s) from {folder}")
            self._append_log(f"Found {total} image(s) to process.")
        elif event == "batch_empty":
            self.progress_text.set("No supported images found.")
            self._append_log("Selected folder does not contain supported images.")
            self._reset_batch_controls()
        elif event == "batch_error":
            message = payload.get("message", "An error occurred.")
            exception = payload.get("exception")
            self._append_log(f"Error: {message} ({exception})")
            self.progress_text.set(message)
            self._reset_batch_controls()
        elif event == "batch_cancelled":
            processed = payload.get("processed", 0)
            total = payload.get("total", 0)
            self.progress_text.set(f"Cancelled after {processed}/{total} images.")
            self._append_log("Batch cancelled by user.")
            self._reset_batch_controls()
        elif event == "batch_complete":
            processed = payload.get("processed", 0)
            total = payload.get("total", 0)
            output = payload.get("output", "")
            self.progress_var.set(1.0)
            self.progress_text.set(f"Completed {processed}/{total} images.")
            self.eta_text.set("Done!")
            self._append_log(f"Batch complete. Files saved to {output}")
            Messagebox.show_info(f"Batch complete! Files saved to {output}", "Background Remover")
            self._reset_batch_controls()
        elif event == "image_start":
            source = payload.get("source", "")
            self.current_image_name.set(Path(source).name)
            self._display_preview(Path(source), self.batch_preview, max_size=200)
            self._append_log(f"Processing {source}")
        elif event == "image_complete":
            destination = payload.get("destination", "")
            source = payload.get("source", "")
            success = payload.get("success", True)
            if success:
                self._append_log(f"✔ Saved {Path(destination).name}")
            else:
                self._append_log(f"⚠ Issue processing {source}")
        elif event == "image_error":
            source = payload.get("source", "")
            error = payload.get("error", "Unknown error")
            self._append_log(f"✖ Failed {source}: {error}")
        elif event == "image_skipped":
            destination = payload.get("destination", "")
            self._append_log(f"⏭ Skipped existing output {destination}")
        elif event == "progress":
            processed = payload.get("processed", 0)
            total = payload.get("total", 0)
            eta = payload.get("eta", 0.0)
            fraction = processed / total if total else 0.0
            self.progress_var.set(fraction)
            self.progress_text.set(f"Processed {processed}/{total} images")
            self.eta_text.set(human_readable_duration(eta))

    def _reset_batch_controls(self) -> None:
        """Return batch controls to their idle state."""

        self.cancel_batch_button.configure(state="disabled")
        self.start_batch_button.configure(state="normal")
        self._folder_worker = None
        self._folder_cancel_event.clear()

    def log_status(self, message: str, *, color: str | None = None) -> None:
        """Expose a safe way to append coloured messages to the status log."""

        self._append_log(message, color=color)

    def _append_log(
        self,
        message: str,
        *,
        tag: str | None = None,
        color: str | None = None,
    ) -> None:
        """Safely append ``message`` to the log console from any thread."""

        effective_tag = tag
        if color and effective_tag is None:
            sanitized = "".join(ch for ch in color if ch.isalnum()) or "custom"
            effective_tag = f"color_{sanitized}"

        def resolve_tag(text: str) -> str:
            """Return the tag name used to style ``text`` in the log."""

            if effective_tag is not None:
                return effective_tag
            lowered = text.casefold()
            if "✖" in text or "⚠" in text or "failed" in lowered or "error" in lowered:
                return "error"
            if "✔" in text or "success" in lowered or "saved" in lowered:
                return "success"
            return "info"

        def write_to_log() -> None:
            try:
                text_widget = getattr(self.log_console, "text", None)
                if text_widget is not None:
                    text_widget.configure(state="normal")
                    tag_name = resolve_tag(message)
                    if color and tag_name:
                        text_widget.tag_configure(tag_name, foreground=color)
                    text_widget.insert(END, message + "\n", tag_name)
                    text_widget.see(END)
                    text_widget.configure(state="disabled")
                else:
                    widget = self.log_console
                    if hasattr(widget, "configure"):
                        try:
                            widget.configure(state="normal")
                        except Exception:
                            pass
                    if hasattr(widget, "insert"):
                        widget.insert(END, message + "\n")
                        if hasattr(widget, "see"):
                            widget.see(END)
                    elif hasattr(widget, "configure") and "text" in widget.keys():
                        current = widget["text"]
                        separator = "\n" if current else ""
                        widget.configure(text=current + separator + message)
                    else:
                        print(f"[LOG]: {message}")
                    if hasattr(widget, "configure"):
                        try:
                            widget.configure(state="disabled")
                        except Exception:
                            pass
            except Exception as exc:
                print(f"[LOG ERROR]: {exc}")
            finally:
                try:
                    self.update_idletasks()
                except Exception:
                    pass

        if threading.current_thread() is threading.main_thread():
            write_to_log()
        else:
            self.after(0, write_to_log)

    def _clear_log(self) -> None:
        """Clear all content from the log console safely."""

        def clear_console() -> None:
            try:
                text_widget = getattr(self.log_console, "text", None)
                if text_widget is not None:
                    text_widget.configure(state="normal")
                    text_widget.delete("1.0", END)
                    text_widget.configure(state="disabled")
                else:
                    widget = self.log_console
                    if hasattr(widget, "configure"):
                        try:
                            widget.configure(state="normal")
                        except Exception:
                            pass

                    if hasattr(widget, "delete"):
                        widget.delete("1.0", END)
                    elif hasattr(widget, "configure") and "text" in widget.keys():
                        widget.configure(text="")

                    if hasattr(widget, "configure"):
                        try:
                            widget.configure(state="disabled")
                        except Exception:
                            pass
            except Exception as exc:
                print(f"[LOG CLEAR ERROR]: {exc}")
            finally:
                try:
                    self.update_idletasks()
                except Exception:
                    pass

        if threading.current_thread() is threading.main_thread():
            clear_console()
        else:
            self.after(0, clear_console)


def main() -> None:
    """Entry point used by ``python bg_remover_gui.py``."""

    _configure_logging()
    LOGGER.info("Launching Background Remover GUI.")

    try:
        app = BackgroundRemoverApp()
    except Exception:
        traceback_text = traceback.format_exc()
        log_path = _write_traceback_to_log(traceback_text)
        LOGGER.exception("Failed to initialise the Background Remover GUI.")
        _show_fatal_error_dialog("Background Remover", log_path)
        return

    try:
        app.mainloop()
    except Exception:
        traceback_text = traceback.format_exc()
        log_path = _write_traceback_to_log(traceback_text)
        LOGGER.exception("Unhandled exception within the Tkinter main loop.")
        _show_fatal_error_dialog("Background Remover", log_path)
    finally:
        LOGGER.info("Background Remover GUI stopped.")


if __name__ == "__main__":
    main()
