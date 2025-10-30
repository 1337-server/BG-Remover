"""High-quality background removal helpers shared by all runtimes."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .config import Config
from .processing.pipeline import ProcessingResult
from .processing.pipeline import process_image as _pipeline_process_image

__all__ = ["ProcessingResult", "process_image", "remove_background"]


def process_image(
    image: np.ndarray,
    *,
    model_key: str | None = None,
    config: Config | None = None,
    feather_radius: int = 3,
    providers: Sequence[str] | None = None,
    **advanced_options: object,
) -> ProcessingResult:
    """Return a :class:`ProcessingResult` produced by the shared pipeline.

    The GUI and Flask runtimes previously imported different helpers which
    diverged in preprocessing behaviour.  By funnelling calls through this
    module we guarantee both front-ends exercise the same corrected pipeline
    configuration.  All parameters are forwarded to the underlying core
    implementation without modification.
    """

    return _pipeline_process_image(
        image,
        model_key=model_key,
        config=config,
        feather_radius=feather_radius,
        providers=providers,
        **advanced_options,
    )


def remove_background(
    image: np.ndarray,
    model_key: str,
    *,
    config: Config | None = None,
    feather_radius: int = 3,
    providers: Sequence[str] | None = None,
    **advanced_options: object,
) -> np.ndarray:
    """Return an RGBA array with the background removed using ``model_key``."""

    result = process_image(
        image,
        model_key=model_key,
        config=config,
        feather_radius=feather_radius,
        providers=providers,
        **advanced_options,
    )
    return np.asarray(result.image)
