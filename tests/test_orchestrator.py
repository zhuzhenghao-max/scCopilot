# -*- coding: utf-8 -*-
"""Tests for agents/orchestrator.py — 多智能体编排层。"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agents import orchestrator
from agents.orchestrator import PlannerAgent, AnnotatorAgent, CriticAgent
from ai.explainer import assess_evidence, Interpretation


def test_planner_decides_use_llm():
    """Planner 根据证据充足度决定是否使用 LLM。"""
    planner = PlannerAgent()

    # 证据充足
    evidence_ok = assess_evidence(["CD3D", "CD3E", "CD4", "IL7R"], cluster_stats={"n_cells": 100})
    plan = planner.plan("0", ["CD3D", "CD3E", "CD4", "IL7R"], evidence_ok)
    assert plan["use_llm"] is True
    assert plan["kg_top_k"] == 5

    # 证据不足（基因太少）
    evidence_weak = assess_evidence(["CD3D"], cluster_stats={})
    plan2 = planner.plan("1", ["CD3D"], evidence_weak)
    assert plan2["use_llm"] is False


def test_planner_does_not_claim_llm_when_none_configured():
    """没配 API Key 的运行里，Planner 不能写「调用 LLM」而 Annotator 实际走本地规则——
    轨迹自相矛盾，等于把项目的核心主张（每个数字都能复现）自己推翻。"""
    planner = PlannerAgent()
    genes = ["CD3D", "CD3E", "CD4", "IL7R"]
    evidence = assess_evidence(genes, cluster_stats={"n_cells": 100})
    assert evidence.ok, "构造样本本身得是充足证据，否则这条测试测不到东西"

    plan = planner.plan("0", genes, evidence, llm_available=False)
    assert plan["use_llm"] is False
    assert "未配置大模型" in plan["reason"], plan
    assert "调用 LLM +" not in plan["reason"], plan

    # 整条工作流离线跑时，Planner 的文案必须和实际执行路径一致
    wf = orchestrator.run_annotation_workflow("0", genes, evidence_rows=None, llm=None, use_llm=False)
    planner_step = next(t for t in wf["trace"] if t["agent_name"] == "Planner")
    annotator_step = next(t for t in wf["trace"] if t["agent_name"] == "Annotator")
    assert planner_step["output_summary"]["use_llm"] is False
    assert "调用 LLM" not in planner_step["output_summary"]["reason"]
    assert annotator_step["output_summary"]["predicted_type"], "离线模式仍要给出结论"


def test_annotator_returns_interpretation():
    """Annotator 返回 Interpretation 对象和 KG 候选。"""
    annotator = AnnotatorAgent()
    evidence = assess_evidence(["CD3D", "CD3E", "CD4"], cluster_stats={"n_cells": 50})
    plan = {"use_llm": False, "kg_top_k": 3}

    itp, kg_candidates = annotator.annotate("0", ["CD3D", "CD3E", "CD4"], evidence, plan)
    assert isinstance(itp, Interpretation)
    assert isinstance(kg_candidates, list)


def test_critic_approves_good_result():
    """Critic 对谱系一致的结果返回 approve。"""
    critic = CriticAgent()
    evidence = assess_evidence(["CD3D", "CD3E", "CD4", "IL7R"], cluster_stats={"n_cells": 100})

    # 本地 KB 出中文类型名，KG 出 CL 英文名——Critic 必须按谱系比，不能按字符串比
    itp = Interpretation(
        cluster="0",
        predicted_type="CD4+ T 细胞",
        confidence_label="高",
        confidence_score=0.9,
        reasoning="Test",
        source="local_kb",
        cited_genes=["CD3D", "CD3E", "CD4"],
        violations=[],
    )
    kg_candidates = [{"cl_id": "CL:0000624", "name": "CD4-positive, alpha-beta T cell",
                      "score": 0.8, "matched_markers": ["CD4"], "parent_cl_id": "CL:0000542"}]

    critique = critic.critique(itp, kg_candidates, evidence)
    assert critique["status"] == "approve", critique


def test_critic_escalates_on_mismatch():
    """预测谱系与 KG 最优候选谱系不一致时返回 escalate。"""
    critic = CriticAgent()
    evidence = assess_evidence(["CD3D", "CD3E", "CD4", "IL7R"], cluster_stats={"n_cells": 100})

    itp = Interpretation(
        cluster="0",
        predicted_type="B 细胞",
        confidence_label="中",
        confidence_score=0.5,
        reasoning="Test",
        source="local_kb",
        cited_genes=["CD3D"],
        violations=[],
    )
    # KG 说的是 T 谱系（CL:0000625 的根是 T cell）
    kg_candidates = [{"cl_id": "CL:0000625", "name": "CD8-positive, alpha-beta T cell",
                      "score": 0.6, "matched_markers": ["CD3D"], "parent_cl_id": "CL:0000542"}]

    critique = critic.critique(itp, kg_candidates, evidence)
    assert critique["status"] == "escalate"
    assert "不一致" in critique["reason"]


def test_critic_approves_when_kg_has_no_candidate():
    """KG 覆盖不到时不能反向判死：本地 KB 的结论照常放行。"""
    critic = CriticAgent()
    evidence = assess_evidence(["CD3D", "CD3E", "CD4", "IL7R"], cluster_stats={"n_cells": 100})

    itp = Interpretation(
        cluster="0", predicted_type="血小板", confidence_label="低", confidence_score=0.3,
        reasoning="Test", source="local_kb", cited_genes=["CD3D"], violations=[],
    )
    critique = critic.critique(itp, [], evidence)
    assert critique["status"] == "approve"


def test_critic_escalates_on_refusal():
    """Critic 对已拒绝的结果直接透传 escalate。"""
    critic = CriticAgent()
    evidence = assess_evidence([], cluster_stats={})

    itp = Interpretation(
        cluster="0",
        predicted_type="未知",
        confidence_label="none",
        confidence_score=0.0,
        reasoning="",
        source="local",
        cited_genes=[],
        violations=[],
        refused=True,
    )

    critique = critic.critique(itp, [], evidence)
    assert critique["status"] == "escalate"
    assert "拒绝" in critique["reason"]


def test_run_annotation_workflow():
    """测试完整工作流入口。"""
    result = orchestrator.run_annotation_workflow(
        cluster_id="0",
        marker_genes=["CD3D", "CD3E", "CD4", "IL7R"],
        cluster_stats={"n_cells": 100, "median_counts": 1000, "median_pct_mito": 5.0},
        llm=None,
        use_llm=False,
    )

    assert "interpretation" in result
    assert "kg_candidates" in result
    assert "critique" in result
    assert "trace" in result
    assert isinstance(result["trace"], list)
    assert len(result["trace"]) >= 3  # Planner + Annotator + Critic


def test_trace_structure():
    """测试 trace 结构是否正确。"""
    result = orchestrator.run_annotation_workflow(
        cluster_id="0",
        marker_genes=["CD3D", "CD3E"],
        use_llm=False,
    )

    trace = result["trace"]
    for step in trace:
        assert "agent_name" in step
        assert "action" in step
        assert "duration_s" in step
        assert step["agent_name"] in ["Planner", "Annotator", "Critic"]


# --------------------------------------------------------------------------- #
# 表达层核实：名字重合只是嫌疑，簇内检出率才是证据
# --------------------------------------------------------------------------- #
def _adata_with(genes_pct: dict) -> object:
    """造一个 40 细胞的假 AnnData：每个基因按给定比例的细胞检测到（值 2.0）。"""
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import anndata as ad
    n = 40
    cols = list(genes_pct)
    mat = np.zeros((n, len(cols)), dtype=np.float32)
    for j, frac in enumerate(genes_pct.values()):
        mat[: int(round(frac * n)), j] = 2.0
    a = ad.AnnData(X=sp.csr_matrix(mat), var=pd.DataFrame(index=cols),
                   obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]))
    a.obs["leiden"] = "0"
    return a


def test_expression_silence_blocks_a_name_match():
    """marker 名字全对上 T 细胞，但全簇一个细胞都没检测到 → 必须 escalate。"""
    a = _adata_with({"CD3D": 0.0, "CD3E": 0.0, "CD4": 0.0, "GNLY": 0.9})
    wf = orchestrator.run_annotation_workflow("0", ["CD3D", "CD3E", "CD4"],
                                              use_llm=False, adata=a)
    assert wf["critique"]["status"] == "escalate", wf["critique"]
    assert "名字对上了但表达不支持" in wf["critique"]["reason"], wf["critique"]
    assert wf["kg_candidates"][0]["expression"]["support_ratio"] == 0.0


def test_expression_support_keeps_the_verdict():
    a = _adata_with({"CD3D": 0.9, "CD3E": 0.8, "CD4": 0.7})
    wf = orchestrator.run_annotation_workflow("0", ["CD3D", "CD3E", "CD4"],
                                              use_llm=False, adata=a)
    assert wf["critique"]["status"] == "approve", wf["critique"]
    assert wf["kg_candidates"][0]["expression"]["support_ratio"] == 1.0


def test_workflow_without_adata_keeps_old_behaviour():
    """不传 adata 时不做表达核实，也别凭空长出 expression 字段。"""
    wf = orchestrator.run_annotation_workflow("0", ["CD3D", "CD3E", "CD4"], use_llm=False)
    assert "expression" not in wf["kg_candidates"][0], wf["kg_candidates"][0]
    assert wf["critique"]["status"] == "approve", wf["critique"]


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
