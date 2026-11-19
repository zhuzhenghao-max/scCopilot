# -*- coding: utf-8 -*-
"""Tests for core/audit_report.py — 单文件 HTML 存证包。

存证包的全部价值在于「别人能照着核对」，所以这里盯三件事：
数字是搬运来的而不是重算的、不可信文本一律转义、prompt 状态标签不许撒谎。
"""

import re
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

from ai.explainer import Interpretation
from agents.orchestrator import run_annotation_workflow
from core import audit_report
from core.pipeline import StateAwarePipeline, PipelineConfig
from make_demo_h5ad import build

_META = {"data_source": "demo_pbmc_sim.h5ad",
         "data_sha256": "a" * 64,
         "code_commit": "15474f4",
         "generated_at": "2026-11-19T00:00:00+00:00"}


def _result():
    sp = StateAwarePipeline(build(seed=42), config=PipelineConfig(n_markers=5, resolution=0.8))
    return sp.run_full()


def _itp(cluster, **kw):
    base = dict(cluster=str(cluster), predicted_type="CD4+ T 细胞", confidence_label="中",
                confidence_score=0.72, reasoning="CD3D/CD3E 支持 T 谱系。",
                source="llm", model="test-model", latency_s=1.25, prompt_chars=7,
                prompt="PROMPT-BODY", raw='{"type": "CD4+ T cell"}')
    base.update(kw)
    return Interpretation(**base)


def test_fingerprint_params_and_qc_present():
    r = _result()
    h = audit_report.build_audit_html(r, {}, _META)
    assert _META["data_sha256"] in h and _META["code_commit"] in h
    assert _META["data_source"] in h
    assert f"{r.adata.n_obs} / {r.adata.n_vars}" in h
    for k, v in r.config.items():
        assert str(k) in h, k
    for k in r.qc:
        assert str(k) in h, k


def test_missing_meta_says_unrecorded():
    h = audit_report.build_audit_html(_result(), {})
    assert h.count("未记录") >= 3
    assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", h)


def test_prompt_and_raw_are_verbatim():
    r = _result()
    cid = str(r.clusters["cluster"].iloc[0])
    prompt = "system: 你是单细胞注释助手\nmarker: CD3D, CD3E\n请只输出 JSON。"
    raw = '{"cell_type": "CD4+ T cell", "confidence": 0.8}\n第二行 <b>原文</b>'
    itp = _itp(cid, prompt=prompt, raw=raw, prompt_chars=len(prompt))
    h = audit_report.build_audit_html(r, {cid: {"interpretation": itp}}, _META)
    assert audit_report._e(prompt) in h
    assert audit_report._e(raw) in h
    assert "已发送的 prompt 原文" in h
    assert str(len(prompt)) in h


def test_local_kb_prompt_is_labelled_not_sent():
    r = _result()
    cid = str(r.clusters["cluster"].iloc[0])
    wf = run_annotation_workflow(
        cid, r.genes_of(cid), evidence_rows=r.evidence_records(cid),
        cluster_stats=r.cluster_stats(cid), llm=None, adata=r.adata)
    assert wf["interpretation"].source == "local_kb"
    h = audit_report.build_audit_html(r, {cid: wf}, _META)
    assert audit_report.PROMPT_STATE["local_kb"] in h
    assert "已发送的 prompt 原文" not in h
    assert wf["interpretation"].prompt, "本地分支也要留下按同一输入构造的 prompt"
    assert audit_report._e(wf["interpretation"].prompt) in h
    assert "未调用大模型" in h


def test_gate_prompt_is_labelled_not_constructed():
    r = _result()
    cid = str(r.clusters["cluster"].iloc[0])
    itp = _itp(cid, source="gate", refused=True, prompt="", raw="",
               predicted_type="证据不足", confidence_label="—", confidence_score=None,
               gate={"ok": False, "reasons": ["可用 marker 只有 1 个"],
                     "suggestions": ["提高 n_markers"]})
    h = audit_report.build_audit_html(r, {cid: {"interpretation": itp}}, _META)
    assert audit_report.PROMPT_STATE["gate"] in h
    assert "（无记录）" in h
    assert "拒答" in h and "可用 marker 只有 1 个" in h and "提高 n_markers" in h


def test_untrusted_text_is_escaped():
    r = _result()
    cid = str(r.clusters["cluster"].iloc[0])
    evil = '<script>alert("x")</script> & O\'Brien <b>CD3D</b>'
    itp = _itp(cid, predicted_type=evil, reasoning=evil, raw=evil, prompt=evil,
               notes=[evil], violations=[evil], fabricated_genes=[evil])
    h = audit_report.build_audit_html(r, {cid: {"interpretation": itp}}, _META)
    assert "<script>" not in h
    assert "&lt;script&gt;" in h
    assert "&amp;" in h and "&quot;" in h and "&#x27;" in h


def test_no_external_resources():
    r = _result()
    cid = str(r.clusters["cluster"].iloc[0])
    wf = run_annotation_workflow(cid, r.genes_of(cid), llm=None)
    h = audit_report.build_audit_html(r, {cid: wf}, _META)
    for pat in (r"<script", r"<link", r"<img", r"<iframe", r"src\s*=", r"href\s*=",
                r"@import", r"url\(", r"https?://"):
        assert not re.search(pat, h, re.I), pat


def test_obs_writeback_counts_match_adata():
    r = _result()
    cid = str(r.clusters["cluster"].iloc[0])
    r.annotate({cid: {"type": "CD4+ T 细胞", "conf": 0.72, "source": "llm",
                      "het": "homogeneous", "blocked": False}})
    h = audit_report.build_audit_html(r, {cid: {"interpretation": _itp(cid)}}, _META)
    sub = r.adata.obs[r.adata.obs[r.cluster_key].astype(str) == cid]
    for col in ("sccopilot_type", "sccopilot_source", "sccopilot_het"):
        for val, n in sub[col].value_counts().items():
            assert f"<td>{val}</td><td class=\"num\">{int(n)}</td>" in h, (col, val, n)


def test_uninterpreted_cluster_is_disclosed():
    r = _result()
    cids = [str(c) for c in r.clusters["cluster"]]
    h = audit_report.build_audit_html(r, {cids[0]: {"interpretation": _itp(cids[0])}}, _META)
    for cid in cids:
        assert f"簇 {cid}</h3>" in h
    assert h.count("本簇未解读") == len(cids) - 1


def test_evidence_numbers_are_copied_not_recomputed():
    r = _result()
    cid = str(r.clusters["cluster"].iloc[0])
    h = audit_report.build_audit_html(r, {cid: {"interpretation": _itp(cid)}}, _META)
    ev = r.evidence_of(cid)
    for _, row in ev.iterrows():
        assert str(row["gene"]) in h
        assert f"{float(row['mean_expr']):.6g}" in h
        assert f"{float(row['pct_expressed']):.6g}" in h


def test_cache_hit_and_critic_are_shown():
    r = _result()
    cid = str(r.clusters["cluster"].iloc[0])
    wf = {"interpretation": _itp(cid, cache_hit=True, cached_at="2026-11-18T09:00:00"),
          "kg_candidates": [{"name": "CD4-positive, alpha-beta T cell", "score": 0.9}],
          "heterogeneity": {"dominant": "T cell", "dominant_prop": 0.81,
                            "normalized_entropy": 0.33, "assigned_prop": 0.9},
          "critique": {"status": "escalate", "reason": "谱系不一致"},
          "trace": [{"agent_name": "Critic", "action": "critique",
                     "duration_s": 0.001, "timestamp": 1763500000.0}]}
    h = audit_report.build_audit_html(r, {cid: wf}, _META)
    assert "缓存命中" in h and "2026-11-18T09:00:00" in h
    assert "本次命中缓存，未重新发送" in h
    assert "CD4-positive, alpha-beta T cell" in h and "0.9" in h
    assert "escalate" in h and "谱系不一致" in h
    assert "Critic" in h and "critique" in h


def test_download_callable_survives_bare_context():
    """下载回调在独立线程执行、st.* 命令被忽略；测试进程没有 ScriptRunContext，正是那个裸上下文。

    回归实测抓过的 bug：回调里读 st.session_state 拿到的 ai_cache 是空的，
    页面明明有结论，导出的存证包却整片「本簇未解读」。
    """
    from app import main as app_main
    r = _result()
    cid = str(r.clusters["cluster"].iloc[0])
    wf = run_annotation_workflow(cid, r.genes_of(cid), llm=None)
    h = app_main.audit_bundle_data(r, {cid: wf}, _META)().decode("utf-8")
    assert h.count("本簇未解读") == len(r.clusters) - 1
    assert audit_report.PROMPT_STATE["local_kb"] in h


def test_deterministic_except_timestamp():
    r = _result()
    cid = str(r.clusters["cluster"].iloc[0])
    wf = {cid: {"interpretation": _itp(cid)}}
    a = audit_report.build_audit_html(r, wf, _META)
    b = audit_report.build_audit_html(r, wf, _META)
    assert a == b
    later = dict(_META, generated_at="2026-11-20T00:00:00+00:00")
    c = audit_report.build_audit_html(r, wf, later)
    assert c != a
    assert c.replace("2026-11-20", "2026-11-19") == a


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