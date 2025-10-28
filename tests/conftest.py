"""Pytest configuration ensuring lightweight rembg stubs."""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

stub_rembg = ModuleType("rembg")
stub_rembg.new_session = lambda *_, **__: object()  # type: ignore[attr-defined]
stub_rembg.remove = lambda data, *_, **__: data  # type: ignore[attr-defined]
sys.modules.setdefault("rembg", stub_rembg)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
