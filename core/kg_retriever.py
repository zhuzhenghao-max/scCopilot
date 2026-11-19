# -*- coding: utf-8 -*-
"""最小可行的免疫学知识图谱（Cell Ontology 子集）。

设计意图：
- 不依赖外部数据库 / 网络，保证离线可用；
- 用 Python 字典硬编码 20 个常见免疫细胞类型（CL ID / 名称 / 父类 / 特征 Marker）；
- 提供 retrieve(marker_genes) 方法：通过基因重合度打分，返回候选细胞类型列表；
- 供多智能体系统的 Annotator / Critic 调用，作为"本地事实来源"。

扩展方式：
- 新增细胞类型：在 CELL_ONTOLOGY 字典里加一条；
- 修改 Marker：直接改 markers 列表；
- 父类关系：parent 字段指向另一个 CL ID（或 None 表示根节点）。
"""

from __future__ import annotations

from typing import Any


# --------------------------------------------------------------------------- #
# 20 个免疫细胞类型（Cell Ontology 子集）
# --------------------------------------------------------------------------- #
CELL_ONTOLOGY: dict[str, dict[str, Any]] = {
    # ---- T 细胞谱系 ----
    "CL:0000542": {
        "name": "T cell",
        "parent": None,
        "markers": ["CD3D", "CD3E", "CD3G", "TRAC", "TRBC2"],
    },
    "CL:0000624": {
        "name": "CD4-positive, alpha-beta T cell",
        "parent": "CL:0000542",
        "markers": ["CD4", "IL7R", "CD3D"],
    },
    "CL:0000625": {
        "name": "CD8-positive, alpha-beta T cell",
        "parent": "CL:0000542",
        "markers": ["CD8A", "CD8B", "CD3D"],
    },
    "CL:0000897": {
        "name": "CD4-positive, alpha-beta regulatory T cell",
        "parent": "CL:0000624",
        "markers": ["FOXP3", "IL2RA", "CTLA4"],
    },
    "CL:0000623": {
        "name": "naive thymus-derived CD4-positive, alpha-beta T cell",
        "parent": "CL:0000624",
        "markers": ["CCR7", "SELL", "LEF1"],
    },
    "CL:0000900": {
        "name": "naive thymus-derived CD8-positive, alpha-beta T cell",
        "parent": "CL:0000625",
        "markers": ["CCR7", "SELL", "LTB"],
    },
    "CL:0000789": {
        "name": "memory T cell",
        "parent": "CL:0000542",
        "markers": ["IL7R", "CD27"],
    },
    "CL:0000798": {
        "name": "gamma-delta T cell",
        "parent": "CL:0000542",
        "markers": ["TRDC", "TRGC1", "TRGC2"],
    },
    # ---- B 细胞谱系 ----
    "CL:0000236": {
        "name": "B cell",
        "parent": None,
        "markers": ["MS4A1", "CD79A", "CD79B"],
    },
    "CL:0000788": {
        "name": "naive B cell",
        "parent": "CL:0000236",
        "markers": ["IGHD", "TCL1A", "FCER2"],
    },
    "CL:0000813": {
        "name": "memory B cell",
        "parent": "CL:0000236",
        "markers": ["CD27", "AICDA"],
    },
    "CL:0000622": {
        "name": "plasma cell",
        "parent": "CL:0000236",
        "markers": ["MZB1", "JCHAIN", "SDC1"],
    },
    # ---- 髓系谱系 ----
    "CL:0000864": {
        "name": "monocyte",
        "parent": None,
        "markers": ["CD14", "LYZ"],
    },
    "CL:0000860": {
        "name": "classical monocyte",
        "parent": "CL:0000864",
        "markers": ["CD14", "S100A8", "S100A9", "FCN1"],
    },
    "CL:0002396": {
        "name": "non-classical monocyte",
        "parent": "CL:0000864",
        "markers": ["FCGR3A", "MS4A7", "CX3CR1"],
    },
    "CL:0000451": {
        "name": "dendritic cell",
        "parent": None,
        "markers": ["HLA-DRA", "HLA-DQB1"],
    },
    "CL:0000888": {
        "name": "CD141-positive myeloid dendritic cell",
        "parent": "CL:0000451",
        "markers": ["THBD", "CLEC9A"],
    },
    "CL:0000890": {
        "name": "CD1c-positive myeloid dendritic cell",
        "parent": "CL:0000451",
        "markers": ["CD1C", "FCER1A"],
    },
    "CL:0000545": {
        "name": "natural killer cell",
        "parent": None,
        "markers": ["GNLY", "NKG7", "KLRD1", "PRF1"],
    },
    "CL:0000775": {
        "name": "neutrophil",
        "parent": None,
        "markers": ["FCGR3B", "CSF3R", "MPO"],
    },
}


# --------------------------------------------------------------------------- #
# 检索接口
# --------------------------------------------------------------------------- #
def retrieve(marker_genes: list[str], *, top_k: int = 5) -> list[dict[str, Any]]:
    """根据输入的 marker 基因列表，返回候选细胞类型（按重合度降序）。

    打分逻辑：
    1. 对每个细胞类型，计算其 marker 集合与输入的重合数；
    2. 归一化：重合数 / max(len(input), len(cell_markers))，避免偏向大集合；
    3. 父类加成：如果某个细胞类型的父类也在候选列表里（且得分 > 0.3），给子类 +0.1 加成；
    4. 按最终得分降序返回 top_k。

    Parameters
    ----------
    marker_genes : list[str]
        用户输入的基因列表（通常是某簇的 Top 差异基因）。
    top_k : int
        返回前 k 个候选。

    Returns
    -------
    list[dict]
        每项包含：
        - cl_id : str (如 "CL:0000624")
        - name : str (如 "CD4-positive, alpha-beta T cell")
        - score : float (0~1)
        - matched_markers : list[str] (重合的基因)
        - parent_cl_id : str | None
    """
    input_set = {g.upper() for g in marker_genes}
    if not input_set:
        return []

    candidates: list[dict[str, Any]] = []
    for cl_id, info in CELL_ONTOLOGY.items():
        cell_markers = {g.upper() for g in info["markers"]}
        matched = input_set & cell_markers
        if not matched:
            continue
        # 归一化：用并集大小（Jaccard 变体）
        union = input_set | cell_markers
        score = len(matched) / len(union)
        candidates.append({
            "cl_id": cl_id,
            "name": info["name"],
            "score": score,
            "matched_markers": sorted(matched),
            "parent_cl_id": info["parent"],
        })

    # 父类加成：如果父类也在候选里且得分 > 0.3，给子类 +0.1
    parent_scores = {c["cl_id"]: c["score"] for c in candidates if c["score"] > 0.3}
    for c in candidates:
        if c["parent_cl_id"] in parent_scores:
            c["score"] = min(1.0, c["score"] + 0.1)

    # 按得分降序
    candidates.sort(key=lambda x: -x["score"])
    return candidates[:top_k]


# --------------------------------------------------------------------------- #
# 辅助查询
# --------------------------------------------------------------------------- #
def get_cell_type(cl_id: str) -> dict[str, Any] | None:
    """根据 CL ID 返回细胞类型信息（不存在则返回 None）。"""
    return CELL_ONTOLOGY.get(cl_id)


def get_ancestors(cl_id: str) -> list[str]:
    """返回指定细胞类型的祖先链（从父到根）。"""
    chain: list[str] = []
    current = cl_id
    while True:
        info = CELL_ONTOLOGY.get(current)
        if not info or info["parent"] is None:
            break
        chain.append(info["parent"])
        current = info["parent"]
    return chain


def is_ancestor_of(ancestor_cl_id: str, descendant_cl_id: str) -> bool:
    """判断 ancestor 是否是 descendant 的祖先（含直接父类）。"""
    return ancestor_cl_id in get_ancestors(descendant_cl_id)
