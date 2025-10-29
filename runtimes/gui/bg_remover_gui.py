"""Tkinter GUI for the background remover runtimes."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox
from typing import Any

import numpy as np
import ttkbootstrap as tb
from PIL import Image
from ttkbootstrap.constants import BOTH, END, LEFT, RIGHT, W
from ttkbootstrap.scrolled import ScrolledText
from ttkbootstrap.tooltip import ToolTip

from bgremover_core import (
    Config,
    init_logging,
    load_config,
    persist_config,
    process_folder,
    remove_background,
)
from bgremover_core.io.image_io import save_image_to_path
from bgremover_core.models.loader import detect_providers
from bgremover_core.models.specs import MODEL_SPECS
from bgremover_core.processing.pipeline import ReportEntry

LOGGER = logging.getLogger(__name__)

GUI_SETTINGS_FILE = Path.home() / ".bgremover_gui.json"

DEFAULT_SETTINGS: dict[str, Any] = {
    "model_key": "isnet-general-use",
    "input_resize": "auto",
    "alpha_matting": False,
    "alpha_foreground_threshold": 240,
    "alpha_background_threshold": 10,
    "alpha_erode_size": 10,
    "smoothing": 0.3,
    "edge_refinement": False,
    "feather_radius": 3,
    "background_color": "",
    "output_format": "PNG",
    "preserve_names": False,
    "output_directory": "",
    "device": "Auto",
    "recursive": False,
    "parallel_threads": 4,
    "model_dir": "",
}


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


class BackgroundRemoverApp(tb.Window):
    """Main application window for background removal."""

    def __init__(self) -> None:
        super().__init__(themename="flatly")
        self.title("Background Remover")
        self.geometry("900x720")
        self.resizable(True, True)

        self.config = load_config()
        init_logging(self.config.log_level)
        self.settings = self._load_settings(self.config)
        self._tooltips: dict[object, ToolTip] = {}

        self.providers = detect_providers(self._provider_hints())
        self._build_ui()
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
        return settings

    def _save_settings(self) -> None:
        """Persist current GUI settings to disk."""

        try:
            GUI_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            payload = self.settings.copy()
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

        device = (self.settings.get("device") or "Auto").lower()
        if device == "gpu":
            return ("CUDAExecutionProvider", "CPUExecutionProvider")
        if device == "cpu":
            return ("CPUExecutionProvider",)
        return self.config.provider_hints

    def _active_config(self) -> Config:
        """Return a :class:`Config` reflecting interactive selections."""

        updates: dict[str, Any] = {}
        model_dir_text = (self.settings.get("model_dir") or "").strip()
        if model_dir_text:
            updates["model_dir"] = Path(model_dir_text)
        provider_hints = self._provider_hints()
        if provider_hints != self.config.provider_hints:
            updates["provider_hints"] = provider_hints
        return self.config.with_updates(**updates) if updates else self.config

    def _processing_kwargs(self) -> dict[str, Any]:
        """Return advanced processing keyword arguments."""

        return {
            "resize_mode": self.settings.get("input_resize", "auto"),
            "alpha_matting": bool(self.settings.get("alpha_matting", False)),
            "alpha_foreground_threshold": self.settings.get("alpha_foreground_threshold", 240),
            "alpha_background_threshold": self.settings.get("alpha_background_threshold", 10),
            "alpha_erode_size": self.settings.get("alpha_erode_size", 10),
            "smoothing": self.settings.get("smoothing", 0.0),
            "edge_refinement": bool(self.settings.get("edge_refinement", False)),
            "background_color": _hex_to_rgb(self.settings.get("background_color")),
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

        header = tb.Frame(container)
        header.pack(fill=BOTH, expand=False)

        tb.Label(header, text="Background Remover", font=("Helvetica", 20, "bold")).pack(side=LEFT)

        self.badge = tb.Label(header, padding=(10, 4))
        self.badge.pack(side=RIGHT)
        self._add_tooltip(self.badge, "Providers: detecting…")

        notebook = tb.Notebook(container, bootstyle="tabs")
        notebook.pack(fill=BOTH, expand=True, pady=(20, 10))

        self.single_tab = tb.Frame(notebook, padding=10)
        notebook.add(self.single_tab, text="Single Image")
        self._build_single_tab(self.single_tab)

        self.batch_tab = tb.Frame(notebook, padding=10)
        notebook.add(self.batch_tab, text="Batch Folder")
        self._build_batch_tab(self.batch_tab)

        advanced_frame = tb.Frame(container)
        advanced_frame.pack(fill=BOTH, expand=False, pady=(0, 10))
        self._build_advanced_panel(advanced_frame)

        log_frame = tb.Labelframe(container, text="Activity Log", padding=10)
        log_frame.pack(fill=BOTH, expand=True)
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

        tb.Button(
            parent,
            text="Process Image",
            bootstyle="primary",
            command=self._process_single,
        ).pack(pady=(10, 0))

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

        tb.Button(
            parent,
            text="Process Folder",
            bootstyle="primary",
            command=self._process_batch,
        ).pack(pady=(10, 10))

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
        self.advanced_visible = tb.BooleanVar(value=True)
        self.toggle_button = tb.Button(header, text="Hide", command=self._toggle_advanced)
        self.toggle_button.pack(side=RIGHT)

        self.advanced_body = tb.Frame(parent, padding=10)
        self.advanced_body.pack(fill=BOTH, expand=True)
        self.advanced_body.grid_columnconfigure(1, weight=1)

        self._build_general_section(self.advanced_body)
        self._build_alpha_section(self.advanced_body)
        self._build_mask_section(self.advanced_body)
        self._build_output_section(self.advanced_body)
        self._build_batch_section(self.advanced_body)
        self._toggle_alpha_controls()

    def _build_general_section(self, parent: tb.Frame) -> None:
        """Create general processing preference controls."""

        general = tb.Frame(parent)
        general.grid(row=0, column=0, columnspan=2, sticky="we", pady=(0, 10))
        general.grid_columnconfigure(1, weight=1)

        tb.Label(general, text="Model").grid(row=0, column=0, sticky=W)
        self.model_var = tb.StringVar(value=self.settings.get("model_key", self.config.default_model))
        model_combo = tb.Combobox(
            general,
            values=sorted(MODEL_SPECS.keys()),
            textvariable=self.model_var,
            width=40,
        )
        model_combo.grid(row=0, column=1, sticky="we", padx=8)
        self._add_tooltip(
            model_combo,
            "Select the model used for background removal. Larger models are slower but more accurate.",
        )
        self.model_var.trace_add("write", lambda *_: self._on_model_change())

        tb.Label(general, text="Input resize").grid(row=1, column=0, sticky=W)
        self.resize_var = tb.StringVar(value=self.settings.get("input_resize", "auto"))
        resize_combo = tb.Combobox(
            general,
            values=["auto", "keep-aspect", "crop", "stretch"],
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
            "Controls how input images are resized before inference. Use 'auto' for best balance.",
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

        tb.Label(general, text="Background fill").grid(row=3, column=0, sticky=W)
        color_frame = tb.Frame(general)
        color_frame.grid(row=3, column=1, sticky="we", padx=8)
        color_frame.grid_columnconfigure(1, weight=1)
        tb.Button(
            color_frame,
            text="Choose",
            command=self._choose_background_color,
        ).grid(row=0, column=0, padx=(0, 6))
        tb.Button(
            color_frame,
            text="Clear",
            command=self._clear_background_color,
        ).grid(row=0, column=1, padx=(0, 6))
        self.color_label = tb.Label(color_frame, text=self._background_label_text(), width=18)
        self.color_label.grid(row=0, column=2, sticky=W)
        self._add_tooltip(
            color_frame,
            "Choose background fill color. Transparent if left unset.",
        )

        tb.Label(general, text="Model directory").grid(row=4, column=0, sticky=W)
        self.model_dir_var = tb.StringVar(value=self.settings.get("model_dir", str(self.config.model_dir)))
        model_dir_frame = tb.Frame(general)
        model_dir_frame.grid(row=4, column=1, sticky="we", padx=8)
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

        frame = tb.Labelframe(parent, text="Alpha Matting Refinement", padding=10)
        frame.grid(row=1, column=0, columnspan=2, sticky="we", pady=(0, 10))
        frame.grid_columnconfigure(1, weight=1)

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

        frame = tb.Labelframe(parent, text="Mask Refinement", padding=10)
        frame.grid(row=2, column=0, columnspan=2, sticky="we", pady=(0, 10))
        frame.grid_columnconfigure(1, weight=1)

        self.smoothing_var = tb.DoubleVar(value=float(self.settings.get("smoothing", 0.3)))
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
            "Applies smoothing to soften mask edges. Range: 0–1. Recommended: 0.3–0.7",
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

        frame = tb.Labelframe(parent, text="Output", padding=10)
        frame.grid(row=3, column=0, columnspan=2, sticky="we", pady=(0, 10))
        frame.grid_columnconfigure(1, weight=1)

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

        frame = tb.Labelframe(parent, text="Batch Processing", padding=10)
        frame.grid(row=4, column=0, columnspan=2, sticky="we")
        frame.grid_columnconfigure(1, weight=1)

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

    def _toggle_advanced(self) -> None:
        """Toggle visibility of the advanced settings frame."""

        if self.advanced_visible.get():
            self.advanced_body.forget()
            self.advanced_visible.set(False)
            self.toggle_button.configure(text="Show")
        else:
            self.advanced_body.pack(fill=BOTH, expand=True)
            self.advanced_visible.set(True)
            self.toggle_button.configure(text="Hide")

    def _background_label_text(self) -> str:
        """Return a human readable summary of the background selection."""

        color = self.settings.get("background_color", "")
        return color or "Transparent"

    def _toggle_alpha_controls(self) -> None:
        """Enable or disable alpha control widgets."""

        state = "normal" if self.alpha_enabled_var.get() else "disabled"
        for widget in self.alpha_controls:
            widget.configure(state=state)

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

    def _clear_background_color(self) -> None:
        """Reset the background fill setting."""

        self._update_setting("background_color", "")
        self.color_label.configure(text=self._background_label_text())

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_model_change(self) -> None:
        """Handle updates to the selected model."""

        model_key = self.model_var.get()
        self._update_setting("model_key", model_key)

    def _on_device_change(self) -> None:
        """Handle device preference updates."""

        self._update_setting("device", self.device_var.get())
        self.providers = detect_providers(self._provider_hints())
        self._refresh_badge()

    def _choose_background_color(self) -> None:
        """Display a color chooser dialog and store the result."""

        _, hex_value = colorchooser.askcolor(title="Choose background color")
        if hex_value:
            self._update_setting("background_color", hex_value)
            self.color_label.configure(text=self._background_label_text())

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

        directory = filedialog.askdirectory()
        if directory:
            self.model_dir_var.set(directory)
            self._on_model_dir_change()

    def _choose_output_dir(self) -> None:
        """Display a directory chooser for the output directory."""

        directory = filedialog.askdirectory()
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

        self._update_setting("smoothing", float(self.smoothing_var.get()))

    def _update_smoothing(self, label: tb.Label) -> None:
        """Refresh smoothing label and persist value."""

        value = float(self.smoothing_var.get())
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
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tiff")]
        )
        if filename:
            self.single_input_var.set(filename)

    def _choose_single_output(self) -> None:
        """Prompt the user for an optional single-image output path."""

        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")],
        )
        if filename:
            self.single_output_var.set(filename)

    def _choose_batch_folder(self) -> None:
        """Prompt the user for a batch input directory."""

        directory = filedialog.askdirectory()
        if directory:
            self.batch_input_var.set(directory)

    def _choose_batch_output(self) -> None:
        """Prompt the user for an optional batch output directory."""

        directory = filedialog.askdirectory()
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
            base_dir = input_path.parent / "output"
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
        threading.Thread(target=self._run_single, args=(path, output_path), daemon=True).start()

    def _run_single(self, input_path: Path, output_path: Path) -> None:
        """Worker that performs single image processing."""

        try:
            with Image.open(input_path) as image:
                array = np.asarray(image.convert("RGBA"))
            config = self._active_config()
            kwargs = self._processing_kwargs()
            kwargs["feather_radius"] = int(self.settings.get("feather_radius", 3))
            result = remove_background(
                array,
                self.model_var.get(),
                config=config,
                **kwargs,
            )
            result_image = Image.fromarray(result)
            format_hint, _ = _format_meta(self.settings.get("output_format", "PNG"))
            if format_hint != "PNG" and result_image.mode != "RGB":
                result_image = result_image.convert("RGB")
            save_image_to_path(result_image, output_path, format_hint=format_hint)
            self.after(0, lambda: self._log(f"{input_path.name} processed successfully ✓"))
        except Exception as error:
            LOGGER.exception("Single image processing failed")
            message = str(error)
            self.after(0, lambda: self._log(f"{input_path.name} failed ✗ — Reason: {message}", error=True))
            self.after(0, lambda: messagebox.showerror("Processing failed", message))

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
            report = process_folder(
                input_dir,
                output_dir,
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
        except Exception as error:
            LOGGER.exception("Batch processing failed")
            message = str(error)
            self.after(0, lambda: self._log(f"Batch failed ✗ — Reason: {message}", error=True))
            self.after(0, lambda: messagebox.showerror("Processing failed", message))

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


def main() -> None:
    """Entry point used by ``python -m`` execution."""

    app = BackgroundRemoverApp()
    app.mainloop()


if __name__ == "__main__":
    main()
