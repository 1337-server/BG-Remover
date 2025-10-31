"""Tkinter GUI for the background remover runtimes."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import ttkbootstrap as tb
from ttkbootstrap.constants import BOTH, LEFT, RIGHT, W
from ttkbootstrap.scrolled import ScrolledText

try:
    import tkinter as tk  # noqa: I001
    from tkinterdnd2 import DND_FILES, TkinterDnD  # noqa: I001

    _DND_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    import tkinter as tk

    _DND_AVAILABLE = False
    DND_FILES = None
    TkinterDnD = None

from tkinter import Canvas

try:
    from bgremover_core import init_logging, load_config
    from bgremover_core.models.loader import detect_providers
    from bgremover_core.models.specs import MODEL_SPECS
except ImportError:  # pragma: no cover - allow running from source without package install
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from bgremover_core import init_logging, load_config
    from bgremover_core.models.loader import detect_providers
    from bgremover_core.models.specs import MODEL_SPECS

from .drag_and_drop_helpers import DragAndDropMixin
from .file_chooser_helpers import FileChooserHelpersMixin
from .processing_helpers import (
    PREVIEW_ZOOM_DEFAULT,
    PREVIEW_ZOOM_MAX,
    PREVIEW_ZOOM_MIN,
    PREVIEW_ZOOM_WHEEL_STEP,
    ProcessingHelpersMixin,
    run_gui_pipeline_for_parity,
)
from .settings_helpers import (
    DEFAULT_SETTINGS,
    GUI_SETTINGS_FILE,
    VALID_RESIZE_MODES,
    SettingsHelpersMixin,
)
from .ui_helpers import BatchProgressWidget, CollapsibleSection, UIHelpersMixin

LOGGER = logging.getLogger(__name__)

__all__ = [
    "BackgroundRemoverApp",
    "main",
    "run_gui_pipeline_for_parity",
    "PREVIEW_ZOOM_DEFAULT",
    "PREVIEW_ZOOM_MIN",
    "PREVIEW_ZOOM_MAX",
    "PREVIEW_ZOOM_WHEEL_STEP",
]


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

    app._dnd_available = _DND_AVAILABLE
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
    app._batch_total_count = 0
    app._batch_processed_count = 0
    app.batch_progress_text = tb.StringVar(value="")
    app.batch_progress_container: tb.Frame | None = None
    app.batch_progress_bar: tb.Progressbar | None = None
    app.batch_progress_label: tb.Label | None = None
    app._batch_progress_widgets: list[BatchProgressWidget] = []
    app._batch_tree_output_paths: dict[str, Path] = {}

    app._init_styles()

    app.config = load_config()
    init_logging(app.config.log_level)
    app.settings = app._load_settings(app.config)
    app.settings.setdefault("theme", app.themename)
    app._tooltips: dict[object, Any] = {}

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
    app._preview_image = None
    app._preview_photo = None
    app._preview_output_path: Path | None = None
    app._preview_format_hint: str | None = None
    app._preview_original_name: str | None = None
    app._preview_saved_path: Path | None = None
    app._preview_canvas_image: int | None = None
    app._preview_display_override = None
    app._preview_last_fill_color = None
    app._preview_fill_color = None
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


class BackgroundRemoverApp(
    DragAndDropMixin,
    SettingsHelpersMixin,
    UIHelpersMixin,
    FileChooserHelpersMixin,
    ProcessingHelpersMixin,
    _TkRoot,
):
    """Main application window for background removal."""

    def __init__(self) -> None:
        super().__init__()
        _initialize_background_remover_app(self)

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

        self.max_perf_var = tk.BooleanVar(
            value=bool(self.settings.get("max_performance", False))
        )
        self.max_perf_check = tb.Checkbutton(
            control_frame,
            text="Max Performance (use all CPU/GPU resources)",
            variable=self.max_perf_var,
            bootstyle="danger-round-toggle",
            command=self._on_max_performance_toggle,
        )
        self.max_perf_check.pack(fill="x", pady=(10, 0))
        self._add_tooltip(
            self.max_perf_check,
            (
                "When enabled, the app will use all available GPU VRAM, CPU cores, "
                "and concurrent workers for fastest processing."
            ),
        )

        notebook = tb.Notebook(control_frame, bootstyle="tabs")
        notebook.pack(fill=BOTH, expand=True, pady=(20, 10))

        self.single_tab = tb.Frame(notebook, padding=10)
        notebook.add(self.single_tab, text="Single Image")
        self._build_single_tab(self.single_tab)

        self.batch_tab = tb.Frame(notebook, padding=10)
        notebook.add(self.batch_tab, text="Batch Folder")
        self._build_batch_tab(self.batch_tab)

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

        # --- universal bindings for zoom + pan ---
        self.preview_canvas.bind("<Enter>", self._on_preview_canvas_enter)
        self.preview_canvas.bind("<Leave>", lambda e: self.preview_canvas.config(cursor=""))
        self.preview_canvas.bind("<ButtonPress-1>", self._on_preview_drag_start)
        self.preview_canvas.bind("<B1-Motion>", self._on_preview_drag_motion)
        self.preview_canvas.bind("<ButtonRelease-1>", self._on_preview_drag_end)
        self.preview_canvas.bind("<Configure>", self._on_preview_canvas_resize)

        # Zoom (Windows/macOS = <MouseWheel>, Linux = <Button-4/5>)
        self.preview_canvas.bind("<MouseWheel>", self._on_preview_mouse_wheel)
        self.preview_canvas.bind("<Button-4>", self._on_preview_mouse_wheel)
        self.preview_canvas.bind("<Button-5>", self._on_preview_mouse_wheel)

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

        # Bottom row: Advanced Settings + Log Window (Resizable)
        bottom_pane = tb.Panedwindow(container, orient="horizontal", bootstyle="default")
        bottom_pane.pack(fill=BOTH, expand=True, pady=(12, 0))

        # --- Left: Advanced Settings (scrollable) ---
        advanced_frame_container = tb.Labelframe(bottom_pane, text="Advanced Settings", padding=12)
        bottom_pane.add(advanced_frame_container)

        # Canvas + Scrollbar
        canvas = tk.Canvas(advanced_frame_container, highlightthickness=0)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)

        scrollbar = tb.Scrollbar(advanced_frame_container, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=RIGHT, fill="y")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Inner frame (actual content)
        scrollable_frame = tb.Frame(canvas)
        scrollable_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfig(scrollable_window, width=event.width)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        scrollable_frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Build advanced settings content here
        self._build_advanced_panel(scrollable_frame)

        # --- Right: Log Window ---
        log_frame = tb.Labelframe(bottom_pane, text="Activity Log", padding=10)
        bottom_pane.add(log_frame)  # add to same PanedWindow
        self.log_widget = ScrolledText(log_frame, height=10)
        self.log_widget.pack(fill=BOTH, expand=False)
        self.log_widget.tag_config("error", foreground="#b91c1c")

        # Make sure settings takes up most width
        bottom_pane.update_idletasks()  # ensures accurate width info
        total_width = bottom_pane.winfo_width()
        bottom_pane.sashpos(0, int(total_width * 2 / 3))  # 2/3 for advanced settings

    def _build_single_tab(self, parent: tb.Frame) -> None:
        """Create widgets for single image processing."""

        io_frame = tb.Frame(parent)
        io_frame.pack(fill="x", pady=5)

        label_opts = dict(sticky="w", padx=(0, 4))
        entry_opts = dict(sticky="ew", padx=(0, 4))
        button_opts = dict(padx=(0, 8))

        tb.Label(io_frame, text="Input image").grid(row=0, column=0, **label_opts)
        self.single_input_var = tb.StringVar(value="")
        single_input_entry = tb.Entry(io_frame, textvariable=self.single_input_var)
        single_input_entry.grid(row=0, column=1, **entry_opts)
        single_input_browse = tb.Button(
            io_frame,
            text="Browse",
            command=self._choose_single_file,
        )
        single_input_browse.grid(row=0, column=2, **button_opts)

        tb.Label(io_frame, text="Output file").grid(row=1, column=0, **label_opts, pady=(4, 0))
        self.single_output_var = tb.StringVar(value="")
        single_output_entry = tb.Entry(io_frame, textvariable=self.single_output_var)
        single_output_entry.grid(row=1, column=1, **entry_opts, pady=(4, 0))
        tb.Button(
            io_frame,
            text="Browse",
            command=self._choose_single_output,
        ).grid(row=1, column=2, **button_opts, pady=(4, 0))

        process_frame = tb.Frame(io_frame)
        process_frame.grid(row=0, column=3, rowspan=2, sticky="ns", padx=(6, 0))

        self.single_process_button = tb.Button(
            process_frame,
            text="Process Image",
            bootstyle="primary",
            command=self._process_single,
            width=16,
        )
        self.single_process_button.pack(fill="both", expand=True)

        self.single_spinner = tb.Progressbar(
            process_frame, mode="indeterminate", length=120, bootstyle="info-striped"
        )
        self.single_spinner.pack(pady=(4, 0))
        self.single_spinner.stop()
        self.single_spinner.pack_forget()

        io_frame.columnconfigure(1, weight=1)

        drop_zone = tb.Frame(parent, padding=16, style="DropZone.TFrame", relief="ridge", borderwidth=2)
        drop_zone.pack(fill=BOTH, expand=True, pady=(10, 5))
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

        batch_frame = tb.Frame(parent)
        batch_frame.pack(fill="x", pady=5)

        label_opts = dict(sticky="w", padx=(0, 4))
        entry_opts = dict(sticky="ew", padx=(0, 4))
        button_opts = dict(padx=(0, 8))

        tb.Label(batch_frame, text="Input folder").grid(row=0, column=0, **label_opts)
        self.batch_input_var = tb.StringVar(value="")
        batch_input_entry = tb.Entry(batch_frame, textvariable=self.batch_input_var)
        batch_input_entry.grid(row=0, column=1, **entry_opts)
        batch_input_browse = tb.Button(
            batch_frame,
            text="Browse",
            command=self._choose_batch_folder,
        )
        batch_input_browse.grid(row=0, column=2, **button_opts)

        batch_output_label = tb.Label(
            batch_frame,
            text="Output folder (optional)",
        )
        batch_output_label.grid(row=1, column=0, **label_opts, pady=(4, 0))
        self.batch_output_var = tb.StringVar(value="")
        batch_output_entry = tb.Entry(batch_frame, textvariable=self.batch_output_var)
        batch_output_entry.grid(row=1, column=1, **entry_opts, pady=(4, 0))
        tb.Button(
            batch_frame,
            text="Browse",
            command=self._choose_batch_output,
        ).grid(row=1, column=2, **button_opts, pady=(4, 0))

        process_frame = tb.Frame(batch_frame)
        process_frame.grid(row=0, column=3, rowspan=2, sticky="ns", padx=(6, 0))

        self.batch_process_button = tb.Button(
            process_frame,
            text="Process Batch/Folder",
            bootstyle="primary",
            command=self._process_batch,
            width=16,
        )
        self.batch_process_button.pack(fill="both", expand=True)

        self.batch_spinner = tb.Progressbar(
            process_frame, mode="indeterminate", length=120, bootstyle="info-striped"
        )
        self.batch_spinner.pack(pady=(4, 0))
        self.batch_spinner.stop()
        self.batch_spinner.pack_forget()

        batch_frame.columnconfigure(1, weight=1)

        drop_zone = tb.Frame(parent, padding=16, style="DropZone.TFrame", relief="ridge", borderwidth=2)
        drop_zone.pack(fill=BOTH, expand=True, pady=(10, 5))
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

        self.batch_spinner = tb.Progressbar(parent, mode="indeterminate", length=220)
        self.batch_spinner.pack(fill="x", pady=(0, 10))
        self.batch_spinner.stop()
        self.batch_spinner.pack_forget()

        self._register_batch_progress_widget(
            parent,
            pack_kwargs={"fill": "x", "expand": False, "pady": (0, 10)},
        )

        progress_frame = tb.Labelframe(parent, text="Batch Progress", padding=6)
        columns = ("file", "status", "details")

        self.batch_tree = tb.Treeview(progress_frame, columns=columns, show="headings", height=6)
        self.batch_tree.heading("file", text="File")
        self.batch_tree.heading("status", text="Status")
        self.batch_tree.heading("details", text="Details")
        self.batch_tree.column("file", width=160, anchor=W)
        self.batch_tree.column("status", width=70, anchor=W)
        self.batch_tree.column("details", anchor=W)
        self.batch_tree.pack(fill=BOTH, expand=True)

        self.batch_tree.bind("<Double-1>", self._on_batch_item_double_click)
        self._batch_tree_output_paths: dict[str, Path] = {}
        self._add_tooltip(self.batch_tree, "Shows progress and results for each processed file.")

        progress_frame.pack_forget()
        self.batch_progress_frame = progress_frame

    def _register_batch_progress_widget(
        self,
        parent: tk.Misc,
        *,
        pack_kwargs: dict[str, Any] | None = None,
        padding: tuple[int, int, int, int] = (0, 8, 0, 0),
    ) -> BatchProgressWidget:
        """Create, hide, and register a batch progress widget for ``parent``."""

        options: dict[str, Any] = {"fill": "x", "expand": False, "pady": (6, 0)}
        if pack_kwargs:
            options.update(pack_kwargs)
        container = tb.Frame(parent, padding=padding)
        bar = tb.Progressbar(container, mode="determinate", bootstyle="info-striped")
        bar.pack(fill="x")
        label = tb.Label(
            container,
            textvariable=self.batch_progress_text,
            anchor="w",
            padding=(0, 4, 0, 0),
        )
        label.pack(anchor="w")
        container.pack(**options)
        container.pack_forget()
        widget = BatchProgressWidget(container=container, bar=bar, label=label, pack_kwargs=options)
        self._batch_progress_widgets.append(widget)
        if len(self._batch_progress_widgets) == 1:
            self.batch_progress_container = container
            self.batch_progress_bar = bar
            self.batch_progress_label = label
        return widget

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

        if not self.advanced_visible.get():
            self.advanced_body.pack(fill=BOTH, expand=True)
            self.advanced_visible.set(True)
            if self.toggle_button:
                self.toggle_button.configure(text="Hide")

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
                text="Collapse All" if expand else "Expand All"
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
                "Controls how input images are resized before inference. 'stretch' matches "
                "CLI and Flask defaults, 'keep-aspect' pads to preserve framing, 'crop' "
                "fills the frame, and 'auto' picks padding or cropping dynamically."
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


def main() -> None:
    """Entry point used by ``python -m`` execution."""

    app = BackgroundRemoverApp()
    app.mainloop()


if __name__ == "__main__":
    main()
