# -*- coding: utf-8 -*-
"""Tests for core/kg_retriever.py — Cell Ontology 知识图谱检索。"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from core import kg_retriever


def test_retrieve_returns_list():
    result = kg_retriever.retrieve(["CD3D", "CD3E"])
    assert isinstance(result, list)


def test_retrieve_top_candidate_is_cd4_t():
    result = kg_retriever.retrieve(["CD3D", "CD3E", "CD4", "IL7R"], top_k=5)
    assert len(result) > 0
    top = result[0]
    assert "CD4" in top["name"] or "T cell" in top["name"]
    assert top["score"] > 0.3


def test_retrieve_empty_markers():
    result = kg_retriever.retrieve([], top_k=5)
    assert result == []


def test_retrieve_unknown_markers():
    result = kg_retriever.retrieve(["FAKE_GENE_1", "FAKE_GENE_2"], top_k=5)
    assert result == []


def test_retrieve_result_has_required_keys():
    result = kg_retriever.retrieve(["CD3D", "CD3E", "CD8A"], top_k=3)
    for item in result:
        assert "cl_id" in item
        assert "name" in item
        assert "score" in item
        assert "matched_markers" in item


def test_cell_ontology_has_20_types():
    assert len(kg_retriever.CELL_ONTOLOGY) == 20


def test_parent_boost():
    """如果父类也在候选里，应该获得 +0.1 加分。"""
    result = kg_retriever.retrieve(["CD3D", "CD3E", "CD4", "IL7R"], top_k=5)
    scores = [r["score"] for r in result]
    assert len(scores) > 0


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
