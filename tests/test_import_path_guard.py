"""Regression guards for import path ambiguity."""

from pathlib import Path

import line_tracker


def test_line_tracker_imports_from_src_tree():
    module_path = Path(line_tracker.__file__).resolve()
    repo_root = Path(__file__).resolve().parents[1]

    assert repo_root / "src" / "line_tracker" in module_path.parents
    assert "build" not in module_path.parts
    assert "finproj1/finproj1/finproj1" not in module_path.as_posix()


def test_dashboard_has_no_sys_path_injection():
    repo_root = Path(__file__).resolve().parents[1]
    dashboard_src = (repo_root / "src" / "line_tracker" / "dashboard.py").read_text()

    assert "sys.path.insert" not in dashboard_src
    assert "_REPO_SRC" not in dashboard_src
