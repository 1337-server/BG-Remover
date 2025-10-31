"""Configuration loading and logging helpers for bg-remover runtimes."""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .paths import CONFIG_FILE, MODELS_DIR

DEFAULT_MODEL_KEY = "isnet-general-use"
CONFIG_FILENAME = CONFIG_FILE.name
ERROR_LOG_NAME = "error.log"


class ConfigError(RuntimeError):
    """Raised when configuration values cannot be parsed or persisted."""


@dataclass(slots=True)
class Config:
    """In-memory configuration shared by every runtime."""

    model_dir: Path
    default_model: str = DEFAULT_MODEL_KEY
    provider_hints: tuple[str, ...] = ()
    log_level: str = "INFO"
    max_performance: bool = False
    config_path: Path | None = None

    def with_updates(self, **kwargs: Any) -> Config:
        """Return a new :class:`Config` with ``kwargs`` applied."""

        return replace(self, **kwargs)

    def resolved_model_dir(self) -> Path:
        """Return the configured model directory ensuring it exists."""

        self.model_dir.mkdir(parents=True, exist_ok=True)
        return self.model_dir


def _default_config_path() -> Path:
    """Return the default path used to persist user configuration."""

    custom_path = os.getenv("BGR_CONFIG_PATH")
    if custom_path:
        return Path(custom_path).expanduser()
    return CONFIG_FILE


def _normalise_provider_hints(raw: Iterable[str] | None) -> tuple[str, ...]:
    """Return a canonical tuple of provider hints from ``raw``."""

    hints = []
    for hint in raw or ():
        value = hint.strip()
        if not value:
            continue
        hints.append(value)
    return tuple(dict.fromkeys(hints))


def load_config(
    *, config_path: Path | None = None, overrides: dict[str, Any] | None = None
) -> Config:
    """Load configuration from the environment, persisted file, and overrides."""

    path = config_path or _default_config_path()
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            raise ConfigError(f"Failed to read configuration file {path}: {exc}") from exc

    env_model_dir = os.getenv("MODEL_DIR")
    env_default_model = os.getenv("BGR_DEFAULT_MODEL")
    env_provider_hints = os.getenv("BGR_PROVIDER_HINTS")
    env_log_level = os.getenv("BGR_LOGLEVEL")

    if env_model_dir:
        data["model_dir"] = env_model_dir
    if env_default_model:
        data["default_model"] = env_default_model
    if env_provider_hints:
        data["provider_hints"] = [item.strip() for item in env_provider_hints.split(",")]
    if env_log_level:
        data["log_level"] = env_log_level

    if overrides:
        data.update(overrides)

    model_dir = Path(data.get("model_dir") or _default_model_dir()).expanduser()
    default_model = str(data.get("default_model") or DEFAULT_MODEL_KEY)
    provider_hints = _normalise_provider_hints(data.get("provider_hints"))
    log_level = str(data.get("log_level") or "INFO").upper()

    max_performance = bool(data.get("max_performance", False))

    return Config(
        model_dir=model_dir,
        default_model=default_model,
        provider_hints=provider_hints,
        log_level=log_level,
        max_performance=max_performance,
        config_path=path,
    )


def _default_model_dir() -> Path:
    """Return the default model cache directory."""

    env_dir = os.getenv("MODEL_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    return MODELS_DIR


def persist_config(config: Config, *, path: Path | None = None) -> Path:
    """Persist ``config`` to ``path`` and return the path."""

    target = path or config.config_path or _default_config_path()
    payload = {
        "model_dir": str(config.model_dir.expanduser()),
        "default_model": config.default_model,
        "provider_hints": list(config.provider_hints),
        "log_level": config.log_level,
        "max_performance": bool(config.max_performance),
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        existing: dict[str, Any] = {}
        if target.exists():
            try:
                existing = json.loads(target.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = {}
        existing.update(payload)
        target.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        raise ConfigError(f"Failed to persist configuration: {exc}") from exc
    return target


_LOGGING_CONFIGURED = False


def init_logging(level: str = "INFO") -> None:
    """Configure root logging for all runtimes with consistent formatting."""

    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        logging.getLogger(__name__).setLevel(level.upper())
        return

    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    log_path = Path.cwd() / ERROR_LOG_NAME
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    _LOGGING_CONFIGURED = True

