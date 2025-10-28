"""Pytest configuration ensuring the repository root is importable."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

os.environ.setdefault("BR_SKIP_RUNTIME_CHECKS", "1")

stub_rembg = ModuleType("rembg")
stub_rembg.new_session = lambda *args, **kwargs: object()  # type: ignore[attr-defined]
stub_rembg.remove = lambda *args, **kwargs: b""  # type: ignore[attr-defined]
sys.modules.setdefault("rembg", stub_rembg)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

