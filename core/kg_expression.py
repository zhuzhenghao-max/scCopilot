# -*- coding: utf-8 -*-
"""知识图谱候选的表达层核实。

为什么要这一步：core.kg_retriever 只看"基因名重不重合"，它无法回答
"这个 marker 在这簇里到底有没有表达"。名字对上了但表达是 0，是单细胞注释里
最典型的一类假阳性——参考列表里有 CD3D，本簇 Top 基因也写着 CD3D，但那是
log2FC 排出来的噪声峰，簇内只有 2% 的细胞真的检测到它。

本模块做的事很朴素，也正因为朴素才可复现：对该簇的细胞，逐个 marker 统计
    · pct_expressed —— 有多少比例的细胞检测到该基因（log 归一化后 > 0）
    · mean_expr     —— 该簇的平均表达量
然后按候选细胞类型汇总成一个 support_ratio。它不依赖任何模型、不联网、
不看注释以外的东西，同一个 .h5ad 跑一万次得到同一个数。

注意口径：pct_expressed 用 X>0 判定，所以 counts 矩阵与 log1p 矩阵都成立；
mean_expr 只在 log 归一化后的 X 上有生物学可比性，因此输出里明确标注 unit。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np


__all__ = ["marker_support", "support_for_candidates", "DEFAULT_MIN_PCT"]

# 一个 marker 在簇内至少要有这么多比例的细胞检测到，才算"这条证据是实的"
DEFAULT_MIN_PCT = 0.15


def _column(X: Any, idx: int) -> np.ndarray:
    """取第 idx 列并压成一维 float32，稀疏/稠密都适用。"""
    col = X[:, idx]
    if hasattr(col, "toarray"):          # scipy sparse 返回 (n,1) 矩阵
        col = col.toarray()
    return np.asarray(col, dtype=np.float32).reshape(-1)


def marker_support(adata: Any, cluster_key: str, cluster: str, genes: Sequence[str], *,
                   min_pct: float = DEFAULT_MIN_PCT) -> dict[str, Any]:
    """统计 genes 在指定簇内的检测率与平均表达量。

    Returns
    -------
    dict
        - ``per_gene``   : [{gene, pct_expressed, mean_expr, detected}]，按 pct_expressed 降序
        - ``present``    : 表达矩阵里找得到的基因
        - ``missing``    : 该簇 Top 列表之外 / 矩阵里根本没有的基因（不能当证据，也不算造假）
        - ``support_ratio``: present 里检测率达标的比例（0~1）
        - ``n_cells``    : 参与统计的细胞数
        - 矩阵为空或簇不存在时，返回 ``support_ratio=0.0`` 且 ``reason`` 说明原因
    """
    out: dict[str, Any] = {"per_gene": [], "present": [], "missing": [],
                           "support_ratio": 0.0, "n_cells": 0, "min_pct": float(min_pct),
                           "unit": "log1p(CPM)" if _looks_log(adata) else "counts"}
    wanted = [str(g) for g in genes if str(g).strip()]
    if not wanted or adata is None or adata.n_obs == 0:
        out["reason"] = "没有可比对的基因或数据为空"
        return out

    var_names = [str(v) for v in adata.var_names]
    index = {v: i for i, v in enumerate(var_names)}
    try:
        mask = np.asarray(adata.obs[cluster_key].astype(str) == str(cluster))
    except KeyError:
        out["reason"] = f"obs 里没有聚类列 {cluster_key}"
        return out
    if not mask.any():
        out["reason"] = f"簇 {cluster} 不存在"
        return out
    out["n_cells"] = int(mask.sum())

    X = adata.X
    # 逐列取再按行号切片：比先 adata[mask] 复制整个子集省内存（大矩阵动辄几 GB）
    rows = np.flatnonzero(mask)
    for gene in wanted:
        gi = index.get(gene)
        if gi is None:
            out["missing"].append(gene)
            continue
        col = _column(X, gi)[rows]
        pct = float(np.mean(col > 0)) if col.size else 0.0
        mean_expr = float(np.mean(col)) if col.size else 0.0
        out["present"].append(gene)
        out["per_gene"].append({
            "gene": gene,
            "pct_expressed": round(pct, 4),
            "mean_expr": round(mean_expr, 4),
            "detected": pct >= float(min_pct),
        })
    out["per_gene"].sort(key=lambda d: (-d["pct_expressed"], d["gene"]))
    n_present = len(out["present"])
    n_ok = sum(1 for d in out["per_gene"] if d["detected"])
    out["support_ratio"] = round(n_ok / n_present, 3) if n_present else 0.0
    return out


def _looks_log(adata: Any) -> bool:
    """粗略判断 X 是否已取过 log，只为给前端标个单位，不参与任何判定。"""
    try:
        X = adata.X
        rows = min(50, X.shape[0])
        probe = X[:rows]
        probe = probe.toarray() if hasattr(probe, "toarray") else np.asarray(probe)
        return float(probe.max()) < 20.0
    except Exception:  # noqa: BLE001 - 单位标注失败不能影响主流程
        return False


def support_for_candidates(adata: Any, cluster_key: str, cluster: str,
                           candidates: Iterable[Mapping[str, Any]], *,
                           min_pct: float = DEFAULT_MIN_PCT,
                           marker_field: str = "matched_markers") -> list[dict[str, Any]]:
    """给 kg_retriever.retrieve() 的候选逐个补上表达层证据。

    默认只看该候选**真正匹配上的 marker**（matched_markers）：这些是它进入候选的理由，
    核实它们比核实整条本体定义更诚实——没匹配上的基因本来就没参与打分。
    """
    enriched: list[dict[str, Any]] = []
    for cand in candidates:
        genes = list(cand.get(marker_field) or [])
        support = marker_support(adata, cluster_key, cluster, genes, min_pct=min_pct)
        item = dict(cand)
        item["expression"] = {
            "support_ratio": support["support_ratio"],
            "detected": [d["gene"] for d in support["per_gene"] if d["detected"]],
            "silent": [d["gene"] for d in support["per_gene"] if not d["detected"]],
            "per_gene": support["per_gene"],
            "n_cells": support["n_cells"],
            "unit": support["unit"],
        }
        if support.get("reason"):
            item["expression"]["reason"] = support["reason"]
        enriched.append(item)
    return enriched
