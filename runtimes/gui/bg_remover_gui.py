"""Tkinter GUI for the background remover runtimes."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np
import ttkbootstrap as tb
from PIL import Image
from ttkbootstrap.constants import BOTH, END
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

LOGGER = logging.getLogger(__name__)


class BackgroundRemoverApp(tb.Window):
    """Main application window for background removal."""

    def __init__(self) -> None:
        super().__init__(themename="flatly")
        self.title("Background Remover")
        self.geometry("760x620")
        self.resizable(True, True)

        self.config = load_config()
        init_logging(self.config.log_level)

        self.providers = detect_providers(self.config.provider_hints)
        self._setup_ui()

    def _setup_ui(self) -> None:
        container = tb.Frame(self, padding=20)
        container.pack(fill=BOTH, expand=True)

        header = tb.Frame(container)
        header.pack(fill=BOTH, expand=False)

        title_label = tb.Label(
            header,
            text="Background Remover",
            font=("Helvetica", 18, "bold"),
        )
        title_label.pack(side="left")

        badge_style = (
            "success"
            if self.providers and self.providers[0].lower().startswith("cuda")
            else "secondary"
        )
        badge_text = "GPU" if badge_style == "success" else "CPU"
        self.badge = tb.Label(
            header,
            text=badge_text,
            bootstyle=f"{badge_style}-inverse",
            padding=(10, 4),
        )
        self.badge.pack(side="right")
        self.badge_tooltip = ToolTip(
            self.badge,
            text="Providers: " + ", ".join(self.providers or ["CPUExecutionProvider"]),
        )

        notebook = tb.Notebook(container, bootstyle="tabs")
        notebook.pack(fill=BOTH, expand=True, pady=(20, 10))

        self.single_tab = tb.Frame(notebook, padding=10)
        notebook.add(self.single_tab, text="Single Image")
        self._build_single_tab(self.single_tab)

        self.batch_tab = tb.Frame(notebook, padding=10)
        notebook.add(self.batch_tab, text="Batch Folder")
        self._build_batch_tab(self.batch_tab)

        options_frame = tb.Labelframe(container, text="Advanced Options", padding=10)
        options_frame.pack(fill=BOTH, expand=False, pady=(0, 10))
        self._build_options(options_frame)

        log_frame = tb.Labelframe(container, text="Activity Log", padding=10)
        log_frame.pack(fill=BOTH, expand=True)
        self.log_widget = ScrolledText(log_frame, height=10)
        self.log_widget.pack(fill=BOTH, expand=True)
        self.log_widget.tag_config("error", foreground="#b91c1c")

    def _build_single_tab(self, parent: tb.Frame) -> None:
        input_frame = tb.Frame(parent)
        input_frame.pack(fill=BOTH, expand=False, pady=5)

        tb.Label(input_frame, text="Input image").pack(anchor="w")
        control = tb.Frame(input_frame)
        control.pack(fill=BOTH, expand=False)
        self.single_input_var = tb.StringVar(value="")
        entry = tb.Entry(control, textvariable=self.single_input_var, width=60)
        entry.pack(side="left", padx=(0, 8))
        tb.Button(control, text="Browse", command=self._choose_single_file).pack(side="left")

        output_frame = tb.Frame(parent)
        output_frame.pack(fill=BOTH, expand=False, pady=5)
        tb.Label(output_frame, text="Output file (optional)").pack(anchor="w")
        self.single_output_var = tb.StringVar(value="")
        output_entry = tb.Entry(output_frame, textvariable=self.single_output_var, width=60)
        output_entry.pack(side="left", padx=(0, 8))
        tb.Button(output_frame, text="Browse", command=self._choose_single_output).pack(side="left")

        tb.Button(
            parent,
            text="Process Image",
            bootstyle="primary",
            command=self._process_single,
        ).pack(pady=(10, 0))

    def _build_batch_tab(self, parent: tb.Frame) -> None:
        input_frame = tb.Frame(parent)
        input_frame.pack(fill=BOTH, expand=False, pady=5)
        tb.Label(input_frame, text="Input folder").pack(anchor="w")
        control = tb.Frame(input_frame)
        control.pack(fill=BOTH, expand=False)
        self.batch_input_var = tb.StringVar(value="")
        tb.Entry(control, textvariable=self.batch_input_var, width=60).pack(side="left", padx=(0, 8))
        tb.Button(control, text="Browse", command=self._choose_batch_folder).pack(side="left")

        output_frame = tb.Frame(parent)
        output_frame.pack(fill=BOTH, expand=False, pady=5)
        tb.Label(output_frame, text="Output folder (optional)").pack(anchor="w")
        self.batch_output_var = tb.StringVar(value="")
        tb.Entry(output_frame, textvariable=self.batch_output_var, width=60).pack(side="left", padx=(0, 8))
        tb.Button(output_frame, text="Browse", command=self._choose_batch_output).pack(side="left")

        tb.Button(
            parent,
            text="Process Folder",
            bootstyle="primary",
            command=self._process_batch,
        ).pack(pady=(10, 0))

    def _build_options(self, parent: tb.Labelframe) -> None:
        tb.Label(parent, text="Model").grid(row=0, column=0, sticky="w")
        self.model_var = tb.StringVar(value=self.config.default_model)
        model_keys = sorted(MODEL_SPECS.keys())
        tb.Combobox(
            parent,
            values=model_keys,
            textvariable=self.model_var,
            width=40,
        ).grid(row=0, column=1, sticky="we", padx=8)

        tb.Label(parent, text="Feather radius").grid(row=1, column=0, sticky="w")
        self.feather_var = tb.IntVar(value=3)
        tb.Spinbox(parent, from_=0, to=50, increment=1, textvariable=self.feather_var, width=10).grid(
            row=1, column=1, sticky="w", padx=8
        )

        tb.Label(parent, text="Model directory").grid(row=2, column=0, sticky="w")
        self.model_dir_var = tb.StringVar(value=str(self.config.model_dir))
        dir_frame = tb.Frame(parent)
        dir_frame.grid(row=2, column=1, sticky="we", padx=8)
        tb.Entry(dir_frame, textvariable=self.model_dir_var, width=40).pack(side="left", padx=(0, 8))
        tb.Button(dir_frame, text="Browse", command=self._choose_model_dir).pack(side="left")
        tb.Button(dir_frame, text="Save", command=self._persist_model_dir).pack(side="left")

        parent.columnconfigure(1, weight=1)

    def _choose_single_file(self) -> None:
        filename = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")])
        if filename:
            self.single_input_var.set(filename)

    def _choose_single_output(self) -> None:
        filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if filename:
            self.single_output_var.set(filename)

    def _choose_batch_folder(self) -> None:
        directory = filedialog.askdirectory()
        if directory:
            self.batch_input_var.set(directory)

    def _choose_batch_output(self) -> None:
        directory = filedialog.askdirectory()
        if directory:
            self.batch_output_var.set(directory)

    def _choose_model_dir(self) -> None:
        directory = filedialog.askdirectory()
        if directory:
            self.model_dir_var.set(directory)

    def _persist_model_dir(self) -> None:
        model_dir_text = self.model_dir_var.get().strip()
        if not model_dir_text:
            messagebox.showerror("Error", "Please choose a directory before saving.")
            return
        updated = self.config.with_updates(model_dir=Path(model_dir_text))
        try:
            persist_config(updated)
            self.config = updated
            self._log("Model directory saved.")
        except Exception as error:  # pragma: no cover - GUI feedback only
            messagebox.showerror("Error", f"Unable to save configuration: {error}")

    def _log(self, message: str, *, error: bool = False) -> None:
        tag = "error" if error else None
        self.log_widget.insert(END, message + "\n", tag)
        self.log_widget.see(END)

    def _active_config(self) -> Config:
        model_dir_text = self.model_dir_var.get().strip()
        if model_dir_text:
            return self.config.with_updates(model_dir=Path(model_dir_text))
        return self.config

    def _process_single(self) -> None:
        path = Path(self.single_input_var.get())
        if not path.exists():
            messagebox.showerror("Error", "Please choose a valid input image.")
            return
        output_path = (
            Path(self.single_output_var.get())
            if self.single_output_var.get()
            else path.with_stem(f"{path.stem}_no_bg")
        )
        self._log(f"Starting processing for {path.name}…")
        threading.Thread(target=self._run_single, args=(path, output_path), daemon=True).start()

    def _run_single(self, input_path: Path, output_path: Path) -> None:
        try:
            with Image.open(input_path) as image:
                array = np.asarray(image.convert("RGBA"))
            config = self._active_config()
            result = remove_background(
                array,
                self.model_var.get(),
                config=config,
                feather_radius=self.feather_var.get(),
            )
            result_image = Image.fromarray(result)
            save_image_to_path(result_image, output_path)
            self.after(0, lambda: self._log(f"{input_path.name} processed successfully ✓"))
        except Exception as error:
            LOGGER.exception("Single image processing failed")
            message = str(error)
            self.after(0, lambda: self._log(f"{input_path.name} failed ✗ — Reason: {message}", error=True))
            self.after(0, lambda: messagebox.showerror("Processing failed", message))

    def _process_batch(self) -> None:
        path = Path(self.batch_input_var.get())
        if not path.exists() or not path.is_dir():
            messagebox.showerror("Error", "Please choose a valid input folder.")
            return
        if self.batch_output_var.get():
            output_dir = Path(self.batch_output_var.get())
        else:
            output_dir = path.parent / "output"
        self._log(f"Starting processing for {path.name}…")
        threading.Thread(target=self._run_batch, args=(path, output_dir), daemon=True).start()

    def _run_batch(self, input_dir: Path, output_dir: Path) -> None:
        try:
            config = self._active_config()
            report = process_folder(
                input_dir,
                output_dir,
                "*",
                model_key=self.model_var.get(),
                config=config,
                feather_radius=self.feather_var.get(),
            )
            for entry in report.entries:
                if entry.success:
                    message = f"{entry.path_in.name} processed successfully ✓"
                    self.after(0, lambda m=message: self._log(m))
                else:
                    message = f"{entry.path_in.name} failed ✗ — Reason: {entry.error}"
                    self.after(0, lambda m=message: self._log(m, error=True))
            summary = f"Batch complete: {report.successes}/{report.total} succeeded"
            self.after(0, lambda: self._log(summary))
            if report.failures:
                self.after(0, lambda: messagebox.showerror("Batch finished with errors", summary))
        except Exception as error:
            LOGGER.exception("Batch processing failed")
            message = str(error)
            self.after(0, lambda: self._log(f"Batch failed ✗ — Reason: {message}", error=True))
            self.after(0, lambda: messagebox.showerror("Processing failed", message))


def main() -> None:
    app = BackgroundRemoverApp()
    app.mainloop()


if __name__ == "__main__":
    main()
