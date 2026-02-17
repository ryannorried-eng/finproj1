"""Pytest configuration for deterministic local-package imports."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = str(_REPO_ROOT / "src")

if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
else:
    sys.path.remove(_SRC_PATH)
    sys.path.insert(0, _SRC_PATH)

if "line_tracker" in sys.modules:
    del sys.modules["line_tracker"]
