# -*- coding: utf-8 -*-
"""Tests for core/heterogeneity.py — 簇内异质性判定（一簇未必是一类细胞）。"""

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

from ai import gene_kb
from agents import orchestrator
from core import heterogeneity
from ai.explainer import assess_evidence

T_PANEL = [str(g) for g in gene_kb.CELL_TYPES["CD4+ T 细胞"]["canonical"]]
B_PANEL = [str(g) for g in gene_kb.CELL_TYPES["初始 B 细胞 (naive B)"]["canonical"]]
M_PANEL = [str(g) for g in gene_kb.CELL_TYPES["经典单核细胞 (CD14+)"]["canonical"]]
NOISE = [f"NOISE{i}" for i in range(300)]
PANELS = {"CD4+ T 细胞": T_PANEL, "初始 B 细胞 (naive B)": B_PANEL}


def build(n_t=250, n_b=150, mode="block", hvg_noise_only=False, dense=False, mouse=False,
          n_ref=200, ref_genes=None):
    """造一份**两簇** AnnData：簇 0 是被测簇，簇 R 是参照簇。

    判型分数是"相对簇外参照细胞的 z 分"，所以夹具必须自带参照系：只给一个簇时，被测簇
    自己就是整份数据，新口径下会得到 not_assessed（这是设计，不是 bug）。
    mode=block 两个真亚群，noise 逐细胞逐基因抽签，gradient 连续过渡，pure 单一类型。
    """
    ref_genes = [str(g) for g in (ref_genes if ref_genes is not None else M_PANEL)]
    marker_genes = list(dict.fromkeys(T_PANEL + B_PANEL))
    genes = ([g[0] + g[1:].lower() for g in marker_genes] if mouse else marker_genes) \
        + ([g[0] + g[1:].lower() for g in ref_genes] if mouse else ref_genes) + NOISE
    genes = list(dict.fromkeys(genes))
    n = n_t + n_b if mode != "pure" else n_t
    n_all = n + n_ref
    rng = np.random.default_rng(0)
    mat = np.zeros((n_all, len(genes)), dtype=np.float32)
    col = {g: i for i, g in enumerate(genes)}

    def fill(rows, grp, lo=1.2, hi=2.0):
        for g in grp:
            key = (g[0] + g[1:].lower()) if mouse else str(g)
            if key not in col:
                continue
            mat[rows, col[key]] = rng.uniform(lo, hi, size=len(rows)).astype(np.float32)

    if mode == "pure":
        fill(np.arange(n), T_PANEL)
    elif mode == "noise":
        for grp in (T_PANEL, B_PANEL):
            for g in grp:
                key = (g[0] + g[1:].lower()) if mouse else g
                hit = np.zeros(n_all, dtype=bool)
                hit[:n] = rng.random(n) < 0.5
                mat[hit, col[key]] = rng.uniform(1.2, 2.0, size=int(hit.sum())).astype(np.float32)
    elif mode == "gradient":
        frac = np.linspace(0.0, 1.0, n)
        for g in T_PANEL:
            mat[:n, col[(g[0] + g[1:].lower()) if mouse else g]] = 2.0 * (1.0 - frac)
        for g in B_PANEL:
            mat[:n, col[(g[0] + g[1:].lower()) if mouse else g]] = 2.0 * frac
    else:  # block
        fill(np.arange(n_t), T_PANEL)
        fill(np.arange(n_t, n_t + n_b), B_PANEL)

    fill(np.arange(n, n_all), ref_genes)          # 参照簇：另一种细胞，不参与被测分布
    mat[:, [col[g] for g in NOISE]] = rng.normal(0.0, 1.0, size=(n_all, len(NOISE))).astype(np.float32)
    mat = np.clip(mat, 0.0, None)
    X = mat if dense else sp.csr_matrix(mat)
    a = ad.AnnData(X=X, var=pd.DataFrame(index=genes),
                   obs=pd.DataFrame(index=[f"c{i}" for i in range(n_all)]))
    a.obs["leiden"] = ["0"] * n + ["R"] * n_ref
    # 默认所有基因都算高变（真实流水线里 marker 就是高变基因）；
    # hvg_noise_only=True 把类型信号藏在**非**高变基因里，用来验拆分子图不只用 HVG。
    a.var["highly_variable"] = (
        np.array([g in NOISE for g in genes]) if hvg_noise_only else True)
    return a


def assess(a, **kw):
    return heterogeneity.assess_cluster(a, "leiden", "0", PANELS, **kw)


# --------------------------------------------------------------------------- #
# 判定本身
# --------------------------------------------------------------------------- #
def test_block_mixture_is_heterogeneous():
    """250 T + 150 B 挤在一个簇里：主标签只覆盖 62.5%，必须判异质。"""
    res = assess(build())
    assert res["verdict"] == "heterogeneous", res["reason"]
    assert res["dominant"] == "CD4+ T 细胞"
    assert 0.60 < res["dominant_prop"] < 0.70, res["dominant_prop"]
    assert res["runner_up"]["label"] == "初始 B 细胞 (naive B)"
    assert res["entropy_bits"] > 0.9, res["entropy_bits"]
    assert "第二种细胞" in res["reason"], res["reason"]


def test_coherent_substructure_is_reported():
    """真亚群：亚簇拆分后纯度接近 100%，理由里要给出"分歧成片"。"""
    res = assess(build())
    assert res["subcluster_run"] is True
    assert res["coherent_substructure"] is True, res["reason"]
    assert res["subcluster_mean_purity"] > 0.9, res
    assert len(res["subclusters"]) == 2
    assert set(res["subcluster_dominants"]) == {"CD4+ T 细胞", "初始 B 细胞 (naive B)"}
    assert "分歧成片" in res["reason"], res["reason"]


def test_incoherent_disagreement_is_noise_not_substructure():
    """每细胞逐基因抽签：判型确实分歧，但拆不出成片亚簇 → 只能报"零散噪声/过渡状态"。"""
    res = assess(build(mode="noise"))
    assert res["verdict"] == "heterogeneous", res["reason"]
    assert res["subcluster_run"] is True
    assert res["coherent_substructure"] is False, res["reason"]
    assert "零散噪声" in res["reason"], res["reason"]


def test_subcluster_split_uses_marker_genes_not_only_hvg():
    """类型信号不在高变基因注释里时也必须拆得开：拆分子图用的是"判别基因 ∪ HVG"。

    踩过反过来的一脚：只用 HVG 拆，任何结构都看不见，真亚群会被误报成噪声。
    """
    res = assess(build(hvg_noise_only=True))
    assert res["verdict"] == "heterogeneous", res["reason"]
    assert res["subcluster_run"] is True
    assert res["coherent_substructure"] is True, res["reason"]
    assert set(res["subcluster_dominants"]) == {"CD4+ T 细胞", "初始 B 细胞 (naive B)"}


def test_pure_cluster_is_homogeneous():
    res = assess(build(n_t=400, n_b=0, mode="pure"))
    assert res["verdict"] == "homogeneous", res["reason"]
    assert res["dominant_prop"] > 0.9
    assert res["subcluster_run"] is False, "同质簇不必花拆分的时间"


def test_small_minority_is_not_flagged():
    """20/400 = 5% 的少数细胞不算"第二种类型"，否则每个簇都会被判异质。"""
    res = assess(build(n_t=380, n_b=20))
    assert res["verdict"] == "homogeneous", res["reason"]
    assert res["runner_up"]["share"] < heterogeneity.MIN_RUNNER_UP_SHARE


def test_unreadable_cluster_is_ambiguous():
    """全零矩阵：谁都不表达 → 判"读不出"，而不是硬选一个主标签。"""
    a = build()
    a.X = sp.csr_matrix(a.X.shape) if sp.issparse(a.X) else np.zeros_like(a.X)
    res = assess(a)
    assert res["verdict"] == "ambiguous", res["reason"]
    assert res["dominant"] == heterogeneity.UNASSIGNED
    assert res["assigned_prop"] == 0.0
    assert "判读" in res["reason"] and "整体命名" in res["reason"], res["reason"]


def test_noise_mixture_is_ambiguous_or_heterogeneous_not_homogeneous():
    res = assess(build(mode="noise"))
    assert res["verdict"] in ("ambiguous", "heterogeneous"), res["reason"]


def test_gradient_cluster_is_flagged():
    res = assess(build(n_t=200, n_b=200, mode="gradient"))
    assert res["verdict"] == "heterogeneous", res["reason"]


# --------------------------------------------------------------------------- #
# 边界：不该给出判定的时候必须明说
# --------------------------------------------------------------------------- #
def test_small_cluster_not_assessed():
    res = assess(build(n_t=40, n_b=0, mode="pure"))
    assert res["verdict"] == "not_assessed", res["reason"]
    assert "40 个细胞" in res["reason"], res["reason"]
    assert res["params"]["min_cells"] == heterogeneity.MIN_CELLS


def test_missing_cluster_not_assessed():
    res = heterogeneity.assess_cluster(build(), "leiden", "9", PANELS)
    assert res["verdict"] == "not_assessed"
    assert "不存在" in res["reason"]


def test_single_candidate_type_not_assessed():
    """只有一个候选类型时没有"分布"可言，不能假装判过。"""
    res = heterogeneity.assess_cluster(build(), "leiden", "0", {"CD4+ T 细胞": T_PANEL})
    assert res["verdict"] == "not_assessed"
    assert "候选类型只有 1 个" in res["reason"]


def test_missing_column_not_assessed():
    res = heterogeneity.assess_cluster(build(), "seurat", "0", PANELS)
    assert res["verdict"] == "not_assessed" and "seurat" in res["reason"]


def test_no_reference_cells_not_assessed():
    """参照簇小到没有参照系（本簇几乎就是整份数据）时，相对差异无从谈起，必须明说。"""
    res = assess(build(n_t=250, n_b=150, n_ref=20))
    assert res["verdict"] == "not_assessed", res["reason"]
    assert "参照细胞" in res["reason"], res["reason"]
    assert res["params"]["min_ref_cells"] == heterogeneity.MIN_REF_CELLS


# --------------------------------------------------------------------------- #
# 判型口径：共享 marker 不能冒充"第二种细胞"
# --------------------------------------------------------------------------- #
def test_shared_marker_type_is_excluded_with_reason():
    """CD8 与 NK 共用细胞毒基因：判别基因不足的类型不参与判型，并写清它跟谁分不开。

    真实数据里踩过这一刀：NK 簇中 CD8+ T 面板一个判别基因都没有，却靠共享的
    GNLY/NKG7/PRF1 抢走 49% 的细胞，把干净簇读成"混了第二种细胞"。
    """
    disc, blocked = heterogeneity.discriminative_panels(
        {"NK 细胞": ["GNLY", "NKG7", "PRF1", "KLRD1", "FCGR3B", "NCAM1", "CD247", "XCL1"],
         "CD8+ T 细胞": ["GNLY", "NKG7", "PRF1", "KLRD1"]})
    assert list(disc) == ["NK 细胞"], disc
    assert blocked["CD8+ T 细胞"]["n_discriminative"] == 0
    assert blocked["CD8+ T 细胞"]["n_panel_genes"] == 4
    assert blocked["CD8+ T 细胞"]["shared_with"] == ["NK 细胞"], blocked


def test_runner_up_type_ignores_unassigned():
    """Unassigned 是"读不出来"，不是"另一种细胞"——它不能当成"这一簇混了"的证据。

    踩过：真实数据的干净 T 簇 64% T + 35% Unassigned，旧口径把 Unassigned 当次多标签，
    于是每个簇都被判成混合簇。
    """
    lab = (["CD4+ T 细胞"] * 64 + [heterogeneity.UNASSIGNED] * 35 + ["初始 B 细胞 (naive B)"])
    d = heterogeneity.label_distribution(lab)
    assert d["runner_up"]["label"] == heterogeneity.UNASSIGNED
    assert d["runner_up_type"] == "初始 B 细胞 (naive B)"
    assert abs(d["runner_up_type_share"] - 0.01) < 1e-6, d["runner_up_type_share"]


def test_assign_labels_needs_both_height_and_margin():
    """贴标签要同时"高出参照"和"明显领先第二名"，否则宁缺毋滥地记 Unassigned。"""
    lab = heterogeneity.assign_labels(pd.DataFrame(
        {"T": [3.0, 0.2, 3.0], "B": [0.0, 0.0, 2.9]}), min_z=0.5, min_margin=0.3)
    assert list(lab) == ["T", heterogeneity.UNASSIGNED, heterogeneity.UNASSIGNED], list(lab)


def test_assigned_cells_below_threshold_is_ambiguous():
    """判读得出的细胞不足六成 → 分布结论不可信，报 ambiguous 而不是硬选主标签。

    夹具里参照细胞对被测面板全零，z 分量级在 1e6，所以用"大到没人能通过"的 margin
    来造这一分支，而不是去伪造分数。
    """
    res = assess(build(), min_margin=1e7)
    assert res["verdict"] == "ambiguous", res["reason"]
    assert res["assigned_prop"] < res["params"]["min_assigned_prop"]
    assert res["dominant"] == heterogeneity.UNASSIGNED


def test_unresolvable_panel_is_not_assessed():
    """候选类型互相共享全部面板基因时判不出类型，要报"没判过"而不是编一个分布。"""
    shared = ["GNLY", "NKG7", "PRF1", "KLRD1"]
    res = heterogeneity.assess_cluster(
        build(), "leiden", "0", {"类型甲": shared, "类型乙": shared})
    assert res["verdict"] == "not_assessed", res["reason"]
    assert "无法区分类型" in res["reason"], res["reason"]
    assert res["types_used"] == {}


def test_none_adata_not_assessed():
    res = heterogeneity.assess_cluster(None, "leiden", "0", PANELS)
    assert res["verdict"] == "not_assessed"


# --------------------------------------------------------------------------- #
# 复现性与口径
# --------------------------------------------------------------------------- #
def test_numbers_reproduce_exactly():
    a, b = build(), build()
    r1, r2 = assess(a), assess(b)
    assert r1 == r2, "同一份数据两次跑，判定与每个数字都必须逐字相同"


def test_dense_and_sparse_agree():
    assert assess(build()) == assess(build(dense=True))


def test_mouse_style_gene_names_are_matched():
    """面板是大写（CD3D），矩阵是小鼠写法（Cd3d）→ 必须落到同一列，否则整份判定会假判"读不出"。"""
    res = assess(build(mouse=True))
    assert res["verdict"] == "heterogeneous", res["reason"]
    assert res["dominant_prop"] > 0.6


def test_assessment_does_not_write_back():
    """判定不能污染共享 adata：obs/var 的列和 X 都得原样。"""
    a = build()
    obs_cols, var_cols = list(a.obs.columns), list(a.var.columns)
    X_before = a.X.toarray()
    assess(a)
    assert list(a.obs.columns) == obs_cols and list(a.var.columns) == var_cols
    assert np.array_equal(np.asarray(a.X.toarray() if sp.issparse(a.X) else a.X), X_before)


def test_params_are_echoed_for_the_ui():
    res = assess(build())
    assert res["params"]["min_dominant_prop"] == heterogeneity.MIN_DOMINANT_PROP
    assert res["params"]["min_panel_genes"] == heterogeneity.MIN_PANEL_GENES
    unit = res["params"]["score_unit"]
    assert "判别基因" in unit and "参照" in unit, unit
    assert "score_genes" not in unit, "口径已改，别再对外宣称用的是 scanpy score_genes"
    assert res["n_reference_cells"] > 0, "参照细胞数要能对外交代"
    assert set(res["z_median"]) == set(res["types_used"])


# --------------------------------------------------------------------------- #
# 面板来源：ai/gene_kb
# --------------------------------------------------------------------------- #
def test_panels_exclude_uninformative_genes():
    panels = gene_kb.heterogeneity_panels(T_PANEL + B_PANEL)
    assert panels, "投票列表非空才有面板"
    for genes in panels.values():
        assert not any(gene_kb.is_uninformative(g) for g in genes), genes


def test_panels_include_reference_types():
    """参照集保证"少数那一头"也参与单细胞判型：只用本簇 Top 基因投票会漏掉它。"""
    panels = gene_kb.heterogeneity_panels(T_PANEL)
    assert "经典单核细胞 (CD14+)" in panels, list(panels)
    assert "初始 B 细胞 (naive B)" in panels, list(panels)


# --------------------------------------------------------------------------- #
# 与 Critic / 工作流的接线
# --------------------------------------------------------------------------- #
def _wf(adata, **kw):
    genes = T_PANEL[:4]
    return orchestrator.run_annotation_workflow("0", genes, use_llm=False, adata=adata, **kw)


def test_workflow_escalates_on_heterogeneous():
    wf = _wf(build())
    het = wf["heterogeneity"]
    assert het["verdict"] == "heterogeneous", het["reason"]
    assert wf["critique"]["status"] == "escalate", wf["critique"]
    assert "不建议整体命名" in wf["critique"]["reason"], wf["critique"]
    assert "主标签" in wf["critique"]["reason"]


def test_critic_escalates_on_ambiguous():
    """Critic 直接吃 ambiguous 判定：全零数据在工作流里会先被表达核实拦下，
    所以这条按单元方式验，避免把"哪道闸门先响"的次序当成被测对象。"""
    critic = orchestrator.CriticAgent()
    evidence = assess_evidence(T_PANEL, cluster_stats={"n_cells": 100})
    itp = orchestrator._local_interpretation(T_PANEL, "0", note="测试")
    kg = [{"cl_id": "CL:0000624", "name": "CD4-positive, alpha-beta T cell",
           "score": 0.8, "matched_markers": ["CD4"]}]
    out = critic.critique(itp, kg, evidence,
                          {"verdict": "ambiguous", "reason": "只有 38% 的细胞能被本次面板判读"})
    assert out["status"] == "escalate", out
    assert "无法由本次 marker 面板判读" in out["reason"], out


def test_workflow_keeps_approve_for_pure_cluster():
    wf = _wf(build(n_t=400, n_b=0, mode="pure"))
    assert wf["heterogeneity"]["verdict"] == "homogeneous", wf["heterogeneity"]["reason"]
    assert wf["critique"]["status"] == "approve", wf["critique"]


def test_workflow_not_assessed_does_not_block():
    """小簇判不了 ≠ 簇有问题：not_assessed 不能反过来把结论判死。"""
    wf = _wf(build(n_t=40, n_b=0, mode="pure"))
    assert wf["heterogeneity"]["verdict"] == "not_assessed"
    assert wf["critique"]["status"] == "approve", wf["critique"]


def test_check_heterogeneity_off_keeps_old_behaviour():
    wf = _wf(build(), check_heterogeneity=False)
    assert wf["heterogeneity"] is None
    assert all("heterogeneity" not in t["output_summary"] for t in wf["trace"])


def test_critic_signature_backwards_compatible():
    """老调用（不传第四个参数）行为不变。"""
    critic = orchestrator.CriticAgent()
    evidence = assess_evidence(T_PANEL, cluster_stats={"n_cells": 100})
    itp = orchestrator._local_interpretation(T_PANEL, "0", note="测试")
    kg = [{"cl_id": "CL:0000624", "name": "CD4-positive, alpha-beta T cell",
           "score": 0.8, "matched_markers": ["CD4"]}]
    assert critic.critique(itp, kg, evidence)["status"] == "approve"
    assert critic.critique(itp, kg, evidence, {"verdict": "not_assessed", "reason": "x"})["status"] == "approve"


# --------------------------------------------------------------------------- #
# 前端渲染（chip 文案 + 不可信输入转义）
# --------------------------------------------------------------------------- #
def test_het_chip_wording_per_verdict():
    from app.main import het_chip
    assert "单一类型" in het_chip({"verdict": "homogeneous", "dominant": "CD4+ T 细胞",
                                  "dominant_prop": 0.93})
    assert "混合了多种细胞" in het_chip({"verdict": "heterogeneous", "dominant": "CD4+ T 细胞",
                                        "dominant_prop": 0.62})
    assert "判读不出来" in het_chip({"verdict": "ambiguous"})
    assert "未做判定" in het_chip({"verdict": "not_assessed"})
    assert "62%" in het_chip({"verdict": "heterogeneous", "dominant": "T", "dominant_prop": 0.62})


def test_het_chip_silent_without_assessment():
    from app.main import het_chip
    assert het_chip(None) == ""
    assert het_chip({}) == ""      # 老缓存里没这个键，不能渲染出半句话


def test_het_chip_escapes_untrusted_names():
    """簇里的基因名/类型名来自用户数据，进 DOM 前必须转义（和 chip 同一条规矩）。"""
    from app.main import het_chip
    evil = "<img src=x onerror=alert(1)>"
    out = het_chip({"verdict": "heterogeneous", "dominant": evil, "dominant_prop": 0.5})
    assert "<img" not in out and "&lt;img" in out, out


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
                print(f"[FAIL] {name}  {type(e).__name__}: {e}")
                failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(1 if failed else 0)
