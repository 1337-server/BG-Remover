"""File chooser helper mixin for the background remover GUI."""
from __future__ import annotations

import sys
from pathlib import Path
from tkinter import filedialog

try:
    from bgremover_core.paths import INPUT_DIR, OUTPUT_DIR
except ImportError:  # pragma: no cover - allow running from source without package install
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from bgremover_core.paths import INPUT_DIR, OUTPUT_DIR

# ------------------------------------------------------------------
# File chooser helpers
# ------------------------------------------------------------------


class FileChooserHelpersMixin:
    """Mixin wrapping the various file chooser dialog callbacks."""

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
