"""Processing and preview helper mixin for the background remover GUI."""
from __future__ import annotations

import gc
import logging
import multiprocessing
import sys
import threading
import webbrowser
from pathlib import Path
from tkinter import messagebox
from typing import TYPE_CHECKING, Any

import ttkbootstrap as tb
from PIL import Image, ImageTk

try:
    from bgremover_core import Config, process_folder
    from bgremover_core.background_remover import process_image
    from bgremover_core.io.image_io import image_to_numpy, save_image_to_path
    from bgremover_core.paths import OUTPUT_DIR
    from bgremover_core.processing.pipeline import ProcessingResult, ReportEntry
    from bgremover_core.processing.utils import iter_image_files
except ImportError:  # pragma: no cover - allow running from source without package install
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from bgremover_core import Config, process_folder
    from bgremover_core.background_remover import process_image
    from bgremover_core.io.image_io import image_to_numpy, save_image_to_path
    from bgremover_core.paths import OUTPUT_DIR
    from bgremover_core.processing.pipeline import ProcessingResult, ReportEntry
    from bgremover_core.processing.utils import iter_image_files

if TYPE_CHECKING:
    from .ui_helpers import BatchProgressWidget

LOGGER = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Processing helpers
# ------------------------------------------------------------------
PREVIEW_ZOOM_MIN = 25.0
PREVIEW_ZOOM_MAX = 400.0
PREVIEW_ZOOM_DEFAULT = 100.0
PREVIEW_ZOOM_WHEEL_STEP = 10.0

# ``PreviewAnchor`` tracks a canvas coordinate and the corresponding widget
# pointer location to keep stable while zooming.
type PreviewAnchor = tuple[float, float, float, float]


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


class ProcessingHelpersMixin:
    """Mixin implementing processing, preview, and logging helpers."""

    def _log(self, message: str, *, error: bool = False) -> None:
        """Append ``message`` to the activity log."""

        tag = "error" if error else None
        self.log_widget.insert(tb.constants.END, message + "\n", tag)
        self.log_widget.see(tb.constants.END)

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
        format_hint, suffix = self._format_meta(self.settings.get("output_format", "PNG"))
        if self.settings.get("preserve_names"):
            return base_dir / f"{input_path.stem}.{suffix}"
        return base_dir / f"{input_path.stem}_no_bg.{suffix}"

    def _format_meta(self, format_name: str) -> tuple[str, str]:
        """Return the Pillow format hint and file suffix for ``format_name``."""

        match format_name.upper():
            case "JPEG" | "JPG":
                return "JPEG", "jpg"
            case "WEBP":
                return "WEBP", "webp"
            case _:
                return "PNG", "png"

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
            config.max_performance = self._max_performance_enabled()
            logging.info("Max Performance Mode: %s", config.max_performance)
            kwargs = self._processing_kwargs()
            kwargs["feather_radius"] = int(self.settings.get("feather_radius", 3))
            result = process_image(
                array,
                model_key=self.model_var.get(),
                config=config,
                **kwargs,
            )
            result_image = result.image
            format_hint, _ = self._format_meta(self.settings.get("output_format", "PNG"))
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
        finally:
            gc.collect()

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
        gc.collect()

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
        self.preview_canvas.config(cursor="hand2")

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
        self.preview_canvas.config(cursor="fleur")
        self.preview_canvas.scan_mark(event.x, event.y)

    def _on_preview_drag_motion(self, event: Any) -> None:
        """Pan the image preview while the left mouse button is held."""

        self.preview_canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_preview_drag_end(self, _event: Any) -> None:
        """Restore the hand cursor after dragging ends."""
        self.preview_canvas.config(cursor="hand2")

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
        self._start_batch_processing()

        self._log(f"Starting processing for {path.name}…")
        self._set_processing_state(True, "batch")

        threading.Thread(
            target=self._run_batch, args=(path, output_dir), daemon=True
        ).start()

    def _start_batch_processing(self):
        """Prepare UI for active batch run."""
        if not self.batch_progress_frame.winfo_ismapped():
            self.batch_progress_frame.pack(fill="both", expand=True, pady=(10, 5))

        for item in self.batch_tree.get_children():
            self.batch_tree.delete(item)

        self.batch_process_button.config(state="disabled", text="")
        self.batch_spinner.place(relx=0.5, rely=0.5, anchor="center")
        self.batch_spinner.start()

    def _end_batch_processing(self):
        """Restore UI after batch completes."""
        self.batch_spinner.stop()
        self.batch_spinner.place_forget()
        self.batch_process_button.config(state="normal", text="Process Batch")

        self._hide_batch_progress()

    def _reset_batch_progress(self) -> None:
        """Clear progress indicators for a new batch run."""
        for item in self.batch_tree.get_children():
            self.batch_tree.delete(item)
        self._batch_tree_output_paths.clear()
        self._batch_total_count = 0
        self._batch_processed_count = 0

        self._hide_batch_progress()

    def _show_batch_progress(self, total: int) -> None:
        """Display the batch progress bar configured for ``total`` entries."""

        widgets = getattr(self, "_batch_progress_widgets", [])
        if not widgets:
            return
        if total <= 0:
            self._hide_batch_progress()
            return
        for widget in widgets:
            widget.bar.configure(maximum=total, value=0)
            if not widget.container.winfo_manager():
                widget.container.pack(**widget.pack_kwargs)
        self._update_batch_progress_label()

    def _update_batch_progress_label(self) -> None:
        """Refresh the batch progress label and bar based on current counts."""

        total = max(0, int(self._batch_total_count))
        processed = max(0, min(int(self._batch_processed_count), total))
        widgets: list[BatchProgressWidget] = getattr(self, "_batch_progress_widgets", [])
        maximum = total if total else 1
        max_perf = self._max_performance_enabled()
        for widget in widgets:
            widget.bar.configure(maximum=maximum, value=processed)
            if hasattr(widget.label, "configure"):
                bootstyle = "danger" if max_perf else ""
                try:
                    widget.label.configure(bootstyle=bootstyle)
                except Exception:  # pragma: no cover - visual hint best-effort
                    pass
        if total <= 0:
            self.batch_progress_text.set("⚡ MAX" if max_perf else "")
            return
        percentage = int((processed / total) * 100)
        prefix = "⚡ MAX — " if max_perf else ""
        self.batch_progress_text.set(
            f"{prefix}{percentage}% — {processed} / {total} processed"
        )

    def _hide_batch_progress(self) -> None:
        """Hide the batch progress UI and reset interactive widgets."""

        if getattr(self, "batch_progress_frame", None) is not None:
            try:
                if self.batch_progress_frame.winfo_ismapped():
                    self.batch_progress_frame.pack_forget()
            except Exception:  # pragma: no cover - defensive UI call
                pass

        widgets: list[BatchProgressWidget] = getattr(self, "_batch_progress_widgets", [])
        for widget in widgets:
            if widget.container.winfo_manager():
                widget.container.pack_forget()
            widget.bar.configure(value=0, maximum=1)
            if hasattr(widget.label, "configure"):
                try:
                    widget.label.configure(bootstyle="")
                except Exception:  # pragma: no cover - visual hint best-effort
                    pass
        self.batch_progress_text.set("")

    def _run_batch(self, input_dir: Path, output_dir: Path | None) -> None:
        """Worker that performs batch processing."""

        try:
            config = self._active_config()
            config.max_performance = self._max_performance_enabled()
            logging.info("Max Performance Mode: %s", config.max_performance)
            kwargs = self._processing_kwargs()
            kwargs.update(
                {
                    "feather_radius": int(self.settings.get("feather_radius", 3)),
                    "recursive": bool(self.settings.get("recursive", False)),
                    "output_format": self.settings.get("output_format", "PNG"),
                    "preserve_names": bool(self.settings.get("preserve_names", False)),
                }
            )
            recursive = bool(kwargs.get("recursive", False))
            total_items = sum(1 for _ in iter_image_files(input_dir, recursive=recursive))
            self._batch_total_count = total_items
            self._batch_processed_count = 0
            self.after(0, lambda: self._show_batch_progress(total_items))
            batch_output_dir = output_dir or OUTPUT_DIR
            batch_output_dir.mkdir(parents=True, exist_ok=True)
            if config.max_performance:
                try:
                    cpu_total = multiprocessing.cpu_count()
                except NotImplementedError:  # pragma: no cover - platform specific
                    cpu_total = 1
                requested_workers = kwargs.get("max_workers") or cpu_total
                if not isinstance(requested_workers, int):
                    try:
                        requested_workers = int(requested_workers)
                    except (TypeError, ValueError):
                        requested_workers = cpu_total
                if total_items > 0:
                    requested_workers = max(1, min(requested_workers, total_items))
                kwargs["max_workers"] = max(1, requested_workers)
            else:
                kwargs.pop("max_workers", None)
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
        item_id = self.batch_tree.insert("", tb.constants.END, values=(entry.path_in.name, status, details))
        if entry.path_out is not None:
            self._batch_tree_output_paths[item_id] = entry.path_out
        self._batch_processed_count += 1
        self._update_batch_progress_label()
        if entry.success:
            log_message = f"{entry.path_in.name} processed successfully ✓"
        else:
            log_message = f"{entry.path_in.name} failed ✗ — Reason: {details}"
        self._log(log_message, error=not entry.success)

    def _on_batch_item_double_click(self, event: Any) -> None:
        """Load and display the selected batch file in the preview window."""

        selection = self.batch_tree.selection()
        if not selection:
            return
        item_id = selection[0]
        values = self.batch_tree.item(item_id, "values")
        if not values or not values[0]:
            return
        file_path = self._batch_tree_output_paths.get(item_id)
        if file_path is None:
            message = (
                "Cannot preview selected batch result: no output file is available for "
                f"{values[0]}."
            )
            messagebox.showinfo("Preview unavailable", message)
            self._log(message, error=True)
            return

        filename = file_path.name
        absolute_path = file_path.resolve(strict=False)
        if not file_path.exists():
            message = (
                "Cannot preview selected batch result: file not found at "
                f"{absolute_path}."
            )
            messagebox.showerror("File not found", message)
            self._log(message, error=True)
            return

        try:
            with Image.open(file_path) as img:
                img = img.convert("RGBA")
            format_hint, _ = self._format_meta(self.settings.get("output_format", "PNG"))
            self._show_preview(img, file_path, format_hint, filename)
            self._log(f"Loaded preview for {filename} from batch output ✓")
        except Exception as error:  # pragma: no cover - defensive log for preview failures
            message = f"Unable to load preview from {absolute_path}: {error}"
            messagebox.showerror("Preview failed", message)
            self._log(
                f"Failed to load preview for {filename} ✗ — Reason: {message}",
                error=True,
            )

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
            if context == "batch":
                self._hide_batch_progress()
