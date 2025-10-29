"""Model specification and loading utilities."""
from .loader import BackgroundRemovalSession, detect_providers, get_session
from .specs import MODEL_SPECS, ModelSpec

__all__ = [
    "BackgroundRemovalSession",
    "MODEL_SPECS",
    "ModelSpec",
    "detect_providers",
    "get_session",
]
