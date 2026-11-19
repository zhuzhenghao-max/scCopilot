# -*- coding: utf-8 -*-
"""core/kg_expression.py 的回归测试：候选细胞类型的表达层核实。

跑法：
    python tests/test_kg_expression.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import anndata as ad  # noqa: E402

from core import kg_expression as K  # noqa: E402


def _adata(kind: str = "sparse") -> ad.AnnData:
    """造一个 4 基因 × 20 细胞的假数据：
    簇 0（10 个细胞）：CD3D 人人都有、CD4 只有 1 个细胞有、GNLY 全无
    簇 1（10 个细胞）：GNLY 全有
    另有一个恒定高表达的 B2M 做对照；矩阵里没有的基因用来验证 missing 分支。
    """
    n = 10
    cols = {
        "CD3D": np.array([3.0] * n + [0.0] * n, dtype=np.float32),
        "CD4": np.array([5.0] + [0.0] * (2 * n - 1), dtype=np.float32),
        "GNLY": np.array([0.0] * n + [4.0] * n, dtype=np.float32),
        "B2M": np.full(2 * n, 6.0, dtype=np.float32),
    }
    X = np.column_stack([cols[g] for g in cols])
    if kind == "sparse":
        X = sp.csr_matrix(X)
    genes = list(cols)
    a = ad.AnnData(
        X=X,
        var=pd.DataFrame(index=genes),
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(2 * n)]),
    )
    a.obs["leiden"] = ["0"] * n + ["1"] * n
    return a


def test_detected_marker_counts_as_evidence():
    res = K.marker_support(_adata(), "leiden", "0", ["CD3D"])
    assert res["support_ratio"] == 1.0, res
    assert res["per_gene"][0]["gene"] == "CD3D"
    assert res["per_gene"][0]["pct_expressed"] == 1.0


def test_name_match_without_expression_is_not_evidence():
    """CD4 名字在列表里，但簇内只有 1/10 个细胞检测到 → 不能算实证据。"""
    res = K.marker_support(_adata(), "leiden", "0", ["CD3D", "CD4"])
    assert res["support_ratio"] == 0.5, res
    silent = [d["gene"] for d in res["per_gene"] if not d["detected"]]
    assert silent == ["CD4"], res["per_gene"]


def test_absent_gene_is_missing_not_silent():
    res = K.marker_support(_adata(), "leiden", "0", ["CD3D", "NOT_IN_MATRIX"])
    assert res["missing"] == ["NOT_IN_MATRIX"], res
    assert res["present"] == ["CD3D"]
    assert res["support_ratio"] == 1.0


def test_dense_and_sparse_give_identical_numbers():
    dense = K.marker_support(_adata("dense"), "leiden", "0", ["CD3D", "CD4", "GNLY"])
    sparse = K.marker_support(_adata("sparse"), "leiden", "0", ["CD3D", "CD4", "GNLY"])
    assert dense["per_gene"] == sparse["per_gene"], (dense, sparse)


def test_same_input_is_reproducible():
    a = _adata()
    first = K.marker_support(a, "leiden", "0", ["CD3D", "CD4"])
    second = K.marker_support(a, "leiden", "0", ["CD3D", "CD4"])
    assert first == second, "同一份数据两次核实结果不同，复现性承诺就破了"


def test_unknown_cluster_reports_reason():
    res = K.marker_support(_adata(), "leiden", "9", ["CD3D"])
    assert res["support_ratio"] == 0.0
    assert "不存在" in res.get("reason", ""), res


def test_missing_cluster_column_reports_reason():
    res = K.marker_support(_adata(), "no_such_col", "0", ["CD3D"])
    assert res["support_ratio"] == 0.0 and "no_such_col" in res.get("reason", ""), res


def test_empty_gene_list_is_safe():
    res = K.marker_support(_adata(), "leiden", "0", [])
    assert res["support_ratio"] == 0.0 and res["per_gene"] == []


def test_support_for_candidates_annotates_each_hit():
    candidates = [
        {"cl_id": "CL:0000624", "name": "CD4-positive T cell", "score": 0.42,
         "matched_markers": ["CD3D", "CD4"]},
        {"cl_id": "CL:0000545", "name": "natural killer cell", "score": 0.2,
         "matched_markers": ["GNLY"]},
    ]
    out = K.support_for_candidates(_adata(), "leiden", "0", candidates)
    assert len(out) == 2
    assert out[0]["expression"]["support_ratio"] == 0.5, out[0]
    assert out[0]["expression"]["detected"] == ["CD3D"]
    assert out[0]["expression"]["silent"] == ["CD4"]
    assert out[1]["expression"]["support_ratio"] == 0.0, "NK 的 GNLY 在簇 0 里应全程静默"
    assert out[0]["score"] == 0.42, "不能把原始候选字段改掉"


def test_unit_is_labelled_from_the_matrix():
    """log 归一化矩阵与 counts 矩阵的单位必须标出来，否则 mean_expr 没法解读。"""
    log_like = K.marker_support(_adata(), "leiden", "0", ["CD3D"])
    assert log_like["unit"] == "log1p(CPM)", log_like
    a = _adata()
    a.X = a.X.toarray() * 1000.0
    counts_like = K.marker_support(a, "leiden", "0", ["CD3D"])
    assert counts_like["unit"] == "counts", counts_like
    assert counts_like["support_ratio"] == 1.0, "检测率口径不应随归一化改变"


def _main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"[PASS] {name:<52s} {time.perf_counter() - t0:6.2f}s")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
