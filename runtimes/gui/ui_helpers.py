"""UI helper mixin and widgets for the background remover GUI."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox
from typing import Any

import ttkbootstrap as tb
from PIL import Image, ImageDraw
from ttkbootstrap.constants import BOTH
from ttkbootstrap.tooltip import ToolTip

try:
    from bgremover_core import persist_config
    from bgremover_core.models.loader import detect_providers
    from bgremover_core.paths import MODELS_DIR, OUTPUT_DIR
except ImportError:  # pragma: no cover - allow running from source without package install
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from bgremover_core import persist_config
    from bgremover_core.models.loader import detect_providers
    from bgremover_core.paths import MODELS_DIR, OUTPUT_DIR

from .settings_helpers import _coerce_smoothing

# ------------------------------------------------------------------
# UI helpers
# ------------------------------------------------------------------


@dataclass(slots=True)
class BatchProgressWidget:
    """Container grouping the UI pieces used for batch progress feedback."""

    container: tb.Frame
    bar: tb.Progressbar
    label: tb.Label
    pack_kwargs: dict[str, Any]


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


class UIHelpersMixin:
    """Mixin providing general-purpose UI helper behaviours."""

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

    def _on_max_performance_toggle(self) -> None:
        """Persist the Max Performance toggle and refresh indicators."""

        enabled = self._max_performance_enabled()
        self._update_setting("max_performance", enabled)
        self._update_batch_progress_label()

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
