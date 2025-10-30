"""Tkinter GUI for the background remover runtimes."""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import logging
import threading
import webbrowser
from pathlib import Path
from tkinter import Canvas, filedialog, messagebox
from typing import Any

import ttkbootstrap as tb
from PIL import Image, ImageDraw, ImageTk
from ttkbootstrap.constants import BOTH, END, LEFT, RIGHT, W
from ttkbootstrap.scrolled import ScrolledText
from ttkbootstrap.tooltip import ToolTip

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


class BackgroundRemoverApp(tb.Window):
    """Main application window for background removal."""

    def __init__(self) -> None:
        theme = DEFAULT_SETTINGS.get("theme", "flatly")
        try:
            if GUI_SETTINGS_FILE.exists():
                user_settings = json.loads(GUI_SETTINGS_FILE.read_text(encoding="utf-8"))
                theme = user_settings.get("theme", theme)
        except Exception:
            pass

        super().__init__(themename=theme)
        self.themename = theme
        self.title("Background Remover - PRO")
        self.geometry("1920x1080")
        self.resizable(True, True)

        self.config = load_config()
        init_logging(self.config.log_level)
        self.settings = self._load_settings(self.config)
        self.settings.setdefault("theme", self.themename)
        self._tooltips: dict[object, ToolTip] = {}
        # --- Add this block here ---

        icon_path = Path(os.path.dirname(__file__)) / "bg_icon.ico"
        logging.info(f"Attempting to load window icon from: {icon_path}")

        try:
            if icon_path.exists():
                self.iconbitmap(icon_path)
                logging.info("Successfully applied .ico icon to GUI window.")
            else:
                logging.warning(f"Icon file not found: {icon_path}")
        except Exception as e:
            logging.exception(f"Failed to set .ico icon: {e}")
            try:
                from tkinter import PhotoImage
                png_icon = icon_path.with_suffix(".png")
                if png_icon.exists():
                    self.iconphoto(False, PhotoImage(file=str(png_icon)))
                    logging.info("Fallback: applied .png icon successfully.")
                else:
                    logging.warning(f"No fallback PNG found at {png_icon}")
            except Exception as e2:
                logging.exception(f"Failed to set .png fallback icon: {e2}")
        # --- End block ---
        self.providers = detect_providers(self._provider_hints())
        self._preview_image: Image.Image | None = None
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._preview_output_path: Path | None = None
        self._preview_format_hint: str | None = None
        self._preview_original_name: str | None = None
        self._preview_saved_path: Path | None = None
        self._preview_canvas_image: int | None = None
        self._preview_display_override: Image.Image | None = None
        self._preview_last_fill_color: tuple[int, int, int] | None = None
        self._preview_last_fill_mode: str | None = None
        self.bg_fill_color: tuple[int, int, int] | str | None = None
        self.preview_zoom_var = tb.DoubleVar(value=100.0)
        self._preview_user_zoom_override = False
        self._suppress_zoom_callback = False
        self._pending_auto_fit_job: str | None = None

        self._build_ui()
        self._clear_preview_state()
        self._refresh_badge()

    # ------------------------------------------------------------------
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
            from_=25,
            to=400,
            orient="horizontal",
            variable=self.preview_zoom_var,
            command=lambda _: self._on_preview_zoom(),
        )
        self.preview_zoom_slider.pack(side=LEFT, fill=BOTH, expand=True, padx=6)
        self.preview_zoom_value = tb.Label(zoom_controls, text="100%", width=6)
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
        """Prompt for a background colour, store it, and apply it."""

        from tkinter import colorchooser

        chosen = colorchooser.askcolor(title="Select background colour")
        if not chosen or not chosen[0]:
            self._log("No background colour selected.")
            return

        rgb = tuple(int(component) for component in chosen[0])
        self.bg_fill_color = rgb
        self._log(f"Background colour set to {self.bg_fill_color}.")
        if not self._preview_image:
            return
        self.apply_background_fill()

    def _preview_clear_bg(self) -> None:
        """Select a transparent preview style and refresh the canvas."""

        self.bg_fill_color = "transparent"
        if not self._preview_image:
            return
        self.apply_background_fill()
        self._log("Background cleared — transparency restored.")

    def apply_background_fill(self, *, redraw: bool = True) -> None:
        """Apply the stored background fill selection to the preview canvas."""

        if not self._preview_image:
            return

        selection = getattr(self, "bg_fill_color", None)
        if selection is None:
            if self._preview_last_fill_color is not None or self._preview_last_fill_mode:
                self._preview_display_override = None
                self._preview_last_fill_color = None
                self._preview_last_fill_mode = None
                if redraw:
                    self._render_preview_image()
                    self._update_preview_controls()
            return

        if selection == "transparent":
            self._preview_display_override = self._build_transparency_preview()
            self._preview_last_fill_color = None
            self._preview_last_fill_mode = "transparent"
            if redraw:
                self._render_preview_image()
                self._update_preview_controls()
            return

        fill = self._normalize_fill_color(selection)
        if fill is None:
            return

        rgba_image = self._preview_image.convert("RGBA")
        background = Image.new("RGBA", rgba_image.size, fill + (255,))
        composed = Image.alpha_composite(background, rgba_image).convert("RGB")
        self._preview_display_override = composed
        self._preview_last_fill_color = fill
        self._preview_last_fill_mode = "color"
        if redraw:
            self._render_preview_image()
            self._update_preview_controls()

    def _build_transparency_preview(self) -> Image.Image:
        """Return a checkerboard composite to visualise transparency."""

        rgba_image = self._preview_image.convert("RGBA")
        width, height = rgba_image.size
        checker_size = 16
        checkerboard = Image.new("RGB", (width, height), "white")
        drawer = ImageDraw.Draw(checkerboard)
        for y_position in range(0, height, checker_size):
            for x_position in range(0, width, checker_size):
                if (x_position // checker_size + y_position // checker_size) % 2 == 0:
                    drawer.rectangle(
                        [
                            x_position,
                            y_position,
                            x_position + checker_size,
                            y_position + checker_size,
                        ],
                        fill=(200, 200, 200),
                    )
        return Image.alpha_composite(checkerboard.convert("RGBA"), rgba_image)

    def _normalize_fill_color(
        self, selection: tuple[int, int, int] | str | None
    ) -> tuple[int, int, int] | None:
        """Return a validated RGB tuple for ``selection``."""

        if selection is None:
            return None
        if isinstance(selection, tuple) and len(selection) == 3:
            return tuple(max(0, min(255, int(value))) for value in selection)
        if isinstance(selection, str):
            parsed = _hex_to_rgb(selection)
            if parsed is not None:
                return parsed
        return None

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
        except Exception as error:
            LOGGER.exception("Single image processing failed")
            message = str(error)
            self.after(0, lambda: self._log(f"{input_path.name} failed ✗ — Reason: {message}", error=True))
            self.after(0, lambda: messagebox.showerror("Processing failed", message))
            self.after(0, lambda: self._set_processing_state(False, "single"))

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
        self._preview_last_fill_mode = None
        self.preview_canvas.delete("all")
        self.preview_canvas.configure(scrollregion=(0, 0, 0, 0))
        self.preview_zoom_var.set(100.0)
        self.preview_zoom_value.configure(text="100%")
        self.preview_zoom_slider.configure(from_=25)
        self.preview_info.configure(text="No preview available yet.")
        self._preview_user_zoom_override = False
        if self._pending_auto_fit_job is not None:
            self.after_cancel(self._pending_auto_fit_job)
            self._pending_auto_fit_job = None
        self._update_preview_controls()

    def _render_preview_image(self) -> None:
        """Render the in-memory preview image respecting the zoom slider."""

        if not self._preview_image:
            self.preview_canvas.delete("all")
            self.preview_canvas.configure(scrollregion=(0, 0, 0, 0))
            return

        zoom_value = max(25.0, min(400.0, float(self.preview_zoom_var.get())))
        self.preview_zoom_var.set(zoom_value)
        scale = zoom_value / 100.0
        source_image = self._preview_display_override or self._preview_image
        width = max(1, int(source_image.width * scale))
        height = max(1, int(source_image.height * scale))
        resized = source_image.resize((width, height), Image.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(resized)
        self.preview_canvas.delete("all")
        canvas_width = max(1, self.preview_canvas.winfo_width())
        canvas_height = max(1, self.preview_canvas.winfo_height())
        offset_x = max((canvas_width - width) // 2, 0)
        offset_y = max((canvas_height - height) // 2, 0)
        self._preview_canvas_image = self.preview_canvas.create_image(
            offset_x,
            offset_y,
            anchor="nw",
            image=self._preview_photo,
        )
        scroll_width = max(width, canvas_width)
        scroll_height = max(height, canvas_height)
        self.preview_canvas.configure(scrollregion=(0, 0, scroll_width, scroll_height))
        self.preview_zoom_value.configure(text=f"{int(zoom_value)}%")

    def _on_preview_zoom(self) -> None:
        """Handle zoom slider changes by re-rendering the preview image."""

        if not self._preview_image:
            self.preview_zoom_var.set(100.0)
            self.preview_zoom_value.configure(text="100%")
            return
        if self._suppress_zoom_callback:
            self._render_preview_image()
            return
        self._preview_user_zoom_override = True
        if self._pending_auto_fit_job is not None:
            self.after_cancel(self._pending_auto_fit_job)
            self._pending_auto_fit_job = None
        current_min = float(self.preview_zoom_slider.cget("from"))
        current_zoom = float(self.preview_zoom_var.get())
        if current_min != 25.0 and current_zoom >= 25.0:
            self.preview_zoom_slider.configure(from_=25)
        self._render_preview_image()

    def _on_preview_canvas_resize(self, _event: Any) -> None:
        """Update the canvas scroll region after a resize event."""

        if self._preview_canvas_image is None:
            return
        if self._preview_user_zoom_override:
            self._render_preview_image()
            return
        self._schedule_auto_fit()

    def _schedule_auto_fit(self, *, force: bool = False) -> None:
        """Schedule an auto-fit pass unless the user set a manual zoom."""

        if not force and self._preview_user_zoom_override:
            return
        if self._pending_auto_fit_job is not None:
            self.after_cancel(self._pending_auto_fit_job)
        self._pending_auto_fit_job = self.after(120, lambda: self.auto_fit_preview(force=force))

    def auto_fit_preview(self, *, force: bool = False) -> None:
        """Fit the preview image within the visible canvas area."""

        self._pending_auto_fit_job = None
        if not self._preview_image:
            return
        if not force and self._preview_user_zoom_override:
            return
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            self._schedule_auto_fit(force=force)
            return
        source_image = self._preview_display_override or self._preview_image
        if source_image.width == 0 or source_image.height == 0:
            return
        width_ratio = canvas_width / source_image.width
        height_ratio = canvas_height / source_image.height
        scale = min(width_ratio, height_ratio, 1.0)
        zoom_percent = max(25.0, min(400.0, scale * 100.0))
        if zoom_percent < float(self.preview_zoom_slider.cget("from")):
            self.preview_zoom_slider.configure(from_=max(1.0, zoom_percent))
        elif float(self.preview_zoom_slider.cget("from")) != 25.0:
            self.preview_zoom_slider.configure(from_=25)
        self._suppress_zoom_callback = True
        self.preview_zoom_var.set(zoom_percent)
        self.preview_zoom_value.configure(text=f"{int(zoom_percent)}%")
        self._suppress_zoom_callback = False
        self._preview_user_zoom_override = False
        self._render_preview_image()

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
        self._preview_last_fill_mode = None
        self.preview_zoom_var.set(100.0)
        self.preview_info.configure(text=f"Preview ready: {original_name}")
        self.apply_background_fill(redraw=False)
        self.auto_fit_preview(force=True)
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
