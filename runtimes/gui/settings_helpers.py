"""Settings-related helpers for the background remover GUI."""
from __future__ import annotations

import json
import logging
import multiprocessing
import sys
from pathlib import Path
from typing import Any

try:
    from bgremover_core import Config
    from bgremover_core.paths import CONFIG_FILE, MODELS_DIR
except ImportError:  # pragma: no cover - allow running from source without package install
    ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from bgremover_core import Config
    from bgremover_core.paths import CONFIG_FILE, MODELS_DIR

LOGGER = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Settings helpers
# ------------------------------------------------------------------
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
    "max_performance": False,
    "model_dir": str(MODELS_DIR),
    "theme": "flatly",
}

VALID_RESIZE_MODES: tuple[str, ...] = ("auto", "keep-aspect", "crop", "stretch")


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


class SettingsHelpersMixin:
    """Mixin implementing helpers for loading and persisting settings."""

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
        settings["max_performance"] = bool(settings.get("max_performance", False))
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

    def _max_performance_enabled(self) -> bool:
        """Return ``True`` when Max Performance mode is currently active."""

        var = getattr(self, "__dict__", {}).get("max_perf_var")
        if var is not None:
            try:
                return bool(var.get())
            except Exception:  # pragma: no cover - fall back to stored state
                pass
        settings = getattr(self, "__dict__", {}).get("settings") or {}
        if "max_performance" in settings:
            return bool(settings.get("max_performance"))
        config = getattr(self, "__dict__", {}).get("config")
        if config is not None and hasattr(config, "max_performance"):
            try:
                return bool(config.max_performance)
            except Exception:  # pragma: no cover - safeguard
                return False
        return False

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
        updates["max_performance"] = self._max_performance_enabled()
        return self.config.with_updates(**updates)

    def _processing_kwargs(self) -> dict[str, Any]:
        """Return advanced processing keyword arguments."""

        kwargs: dict[str, Any] = {
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
        if self._max_performance_enabled():
            try:
                cpu_total = multiprocessing.cpu_count()
            except NotImplementedError:  # pragma: no cover - platform specific
                cpu_total = 1
            kwargs["max_workers"] = max(1, cpu_total)
        return kwargs
