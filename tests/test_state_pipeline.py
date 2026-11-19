# -*- coding: utf-8 -*-
"""Tests for core/pipeline.StateAwarePipeline — 状态机引擎。"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

from core.pipeline import StateAwarePipeline, StateError, PipelineConfig
from make_demo_h5ad import build


def test_state_machine_transitions():
    """测试状态机正常转换：raw → qc_passed → embedded → clustered。"""
    adata = build(seed=42)
    cfg = PipelineConfig(n_markers=5, resolution=0.8)
    sp = StateAwarePipeline(adata, config=cfg)
    assert sp.state == "raw"

    sp.run_qc()
    assert sp.state == "qc_passed"

    sp.run_embedding()
    assert sp.state == "embedded"

    result = sp.run_clustering()
    assert sp.state == "clustered"
    assert result is not None


def test_state_error_on_illegal_transition():
    """测试非法状态转换抛出 StateError。"""
    adata = build(seed=43)
    cfg = PipelineConfig(n_markers=5)
    sp = StateAwarePipeline(adata, config=cfg)

    # 不能跳过 QC 直接 embedding
    try:
        sp.run_embedding()
        assert False, "Should have raised StateError"
    except StateError:
        pass


def test_run_full_pipeline():
    """测试 run_full 一键跑完全流程。"""
    adata = build(seed=44)
    cfg = PipelineConfig(n_markers=5, resolution=0.8)
    sp = StateAwarePipeline(adata, config=cfg)
    result = sp.run_full()
    assert sp.state == "clustered"
    assert result is not None
    assert len(result.clusters) > 0


def test_provenance_tracker_integrated():
    """测试 ProvenanceTracker 被正确集成。"""
    adata = build(seed=45)
    cfg = PipelineConfig(n_markers=5, resolution=0.8)
    sp = StateAwarePipeline(adata, config=cfg)
    sp.run_full()
    assert sp.provenance is not None
    assert len(sp.provenance) >= 3  # qc, embedding, clustering


def test_retrieve_cell_types():
    """测试 retrieve_cell_types 方法。"""
    adata = build(seed=46)
    cfg = PipelineConfig(n_markers=10, resolution=0.8)
    sp = StateAwarePipeline(adata, config=cfg)
    result = sp.run_full()

    # 取第一个簇的 marker
    cluster = result.clusters["cluster"].iloc[0]
    candidates = sp.retrieve_cell_types(cluster, top_k=3)
    assert isinstance(candidates, list)
    if len(candidates) > 0:
        assert "name" in candidates[0]
        assert "score" in candidates[0]


def test_state_error_message():
    """测试 StateError 的错误消息是否清晰。"""
    adata = build(seed=47)
    cfg = PipelineConfig(n_markers=5)
    sp = StateAwarePipeline(adata, config=cfg)

    try:
        sp.run_embedding()
    except StateError as e:
        assert "qc_passed" in str(e) or "raw" in str(e)


# --------------------------------------------------------------------------- #
# 运行
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}  ", end="")
                passed += 1
            except Exception as e:
                print(f"[FAIL] {name}  {e}")
                failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(1 if failed else 0)
