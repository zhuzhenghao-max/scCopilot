# -*- coding: utf-8 -*-
"""Tests for core/provenance.py — 审计追踪。"""

import sys, pathlib, json, tempfile, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from core.provenance import ProvenanceTracker


def test_tracker_records_steps():
    tracker = ProvenanceTracker()
    tracker.log_step("load_h5ad", {"path": "test.h5ad"}, {"n_cells": 100}, duration_s=0.5)
    tracker.log_step("qc", {"min_genes": 200}, {"passed": 90}, duration_s=1.2)
    assert len(tracker) == 2
    steps = tracker.get_steps()
    assert steps[0]["tool_name"] == "load_h5ad"
    assert steps[1]["tool_name"] == "qc"


def test_export_json():
    tracker = ProvenanceTracker()
    tracker.log_step("test", {"a": 1}, {"b": 2}, duration_s=0.1)
    data = tracker.export_json()
    assert isinstance(data, dict)
    assert "steps" in data
    assert data["total_steps"] == 1
    assert data["steps"][0]["tool_name"] == "test"
    assert "timestamp" in data["steps"][0]
    assert "duration_s" in data["steps"][0]


def test_save_json():
    tracker = ProvenanceTracker()
    tracker.log_step("test", {"a": 1}, {"b": 2}, duration_s=0.1)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name
    try:
        tracker.save_json(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert data["total_steps"] == 1
    finally:
        os.unlink(path)


def test_step_record_fields():
    tracker = ProvenanceTracker()
    tracker.log_step("tool", {"param": "val"}, {"result": "ok"}, duration_s=0.5)
    steps = tracker.get_steps()
    assert steps[0]["tool_name"] == "tool"
    assert steps[0]["params"] == {"param": "val"}
    assert steps[0]["result_summary"] == {"result": "ok"}
    assert steps[0]["duration_s"] == 0.5
    assert steps[0]["timestamp"] > 0


# --------------------------------------------------------------------------- #
# 运行
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
                passed += 1
            except Exception as e:
                print(f"[FAIL] {name}  {e}")
                failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(1 if failed else 0)
