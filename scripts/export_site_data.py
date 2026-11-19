# -*- coding: utf-8 -*-
"""把真实分析结果预计算成静态站可用的 JSON。

为什么需要这个脚本
------------------
Qoder Sites 只托管「构建产物 + TypeScript 函数」，没有 Python 运行时，跑不动
常驻的 Streamlit 进程。所以把 scanpy 的重活留在本地离线做完，前端只消费结果：
评委点开就是真实 UMAP、真实 marker、真实多智能体判定，而不是示意图。

产出（web/data/）
----------------
manifest.json     场景索引，前端据此加载
<scenario>.json   单场景全量：UMAP 坐标 / 簇表 / marker 证据 / AI 工作流结果

三个场景都是真的跑出来的，不是手工编数据：
  normal       演示数据，res=1.0，基线
  resolution   同一批细胞，res=1.6，附与基线的真实交叉表对比（拆分/合并）
  unannotated  把基因名换成 Ensembl ID 且不提供 symbol 列 —— 真实 10x 产物常见形态，
               此时 AI 读不懂 marker，应当整簇拒绝下结论，而不是猜一个细胞类型。

启动：
    python scripts/export_site_data.py
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import warnings
import zlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

from ai import explainer as ai           # noqa: E402
from agents import orchestrator          # noqa: E402
from core import pipeline as pl          # noqa: E402
from make_demo_h5ad import build         # noqa: E402

warnings.simplefilter("ignore")

OUT_DIR = ROOT / "web" / "data"
N_MARKERS = 8


# --------------------------------------------------------------------------- #
# JSON 兼容
# --------------------------------------------------------------------------- #
def json_safe(obj: Any) -> Any:
    """递归清掉 numpy 标量、NaN、pandas 类型——JSON 标准里没有 NaN，浏览器会解析失败。"""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, pd.Series):
        return json_safe(obj.tolist())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return json_safe(dataclasses.asdict(obj))
    return obj


def dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"  wrote {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.0f} KB)")


# --------------------------------------------------------------------------- #
# 单场景打包
# --------------------------------------------------------------------------- #
def per_cell(result: pl.PipelineResult) -> dict[str, Any]:
    """逐细胞数据：UMAP 坐标 + 簇标签 + 两条 QC 轴（供前端着色）。"""
    umap = np.asarray(result.adata.obsm["X_umap"], dtype=np.float64)
    obs = result.adata.obs
    labels = obs[result.cluster_key].astype(str).tolist()
    return {
        "x": [round(float(v), 3) for v in umap[:, 0]],
        "y": [round(float(v), 3) for v in umap[:, 1]],
        "cluster": labels,
        "pct_mito": [round(float(v), 3) for v in obs["pct_mito"].to_numpy()],
        "n_counts": [round(float(v), 1) for v in obs["n_counts"].to_numpy()],
    }


def interpret_all(result: pl.PipelineResult) -> dict[str, Any]:
    """对每个簇跑一遍多智能体工作流，把结论与轨迹一起冻结进 JSON。"""
    out: dict[str, Any] = {}
    for cluster in result.clusters["cluster"].tolist():
        wf = orchestrator.run_annotation_workflow(
            cluster_id=str(cluster),
            marker_genes=result.genes_of(cluster, top=N_MARKERS),
            cluster_stats=result.cluster_stats(cluster),
            evidence_rows=result.evidence_records(cluster, top=N_MARKERS),
            llm=None,          # 静态站要能离线复现，不依赖 API Key
            use_llm=False,
            adata=result.adata,            # 传了矩阵才会做簇内异质性判定
            cluster_key=result.cluster_key,
        )
        itp = wf["interpretation"]
        out[str(cluster)] = {
            "predicted_type": itp.predicted_type,
            "confidence_label": itp.confidence_label,
            "confidence_score": itp.confidence_score,
            "reasoning": itp.reasoning,
            "source": itp.source,
            "refused": itp.refused,
            "cited_genes": itp.cited_genes,
            "violations": itp.violations,
            "notes": itp.notes,
            "gate": itp.gate,
            "kg_candidates": wf["kg_candidates"],
            "critique": wf["critique"],
            "heterogeneity": wf["heterogeneity"],
            "trace": wf["trace"],
        }
    return out


def package(scenario_id: str, title: str, blurb: str, result: pl.PipelineResult,
            *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "id": scenario_id,
        "title": title,
        "blurb": blurb,
        "config": result.config,
        "qc": result.qc,
        "timings": result.timings,
        "warnings": result.warnings,
        "n_markers": N_MARKERS,
        "umap": per_cell(result),
        "clusters": result.clusters.to_dict("records"),
        "markers": {
            str(c): result.evidence_records(c, top=N_MARKERS)
            for c in result.clusters["cluster"].tolist()
        },
        "ai": interpret_all(result),
    }
    if extra:
        payload.update(extra)
    return payload


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] normal · res=1.0")
    base = pl.run_pipeline(build(seed=0), pl.PipelineConfig(resolution=1.0, n_markers=N_MARKERS))
    p_normal = package("normal", "PBMC 演示数据 · 基线",
                       "标准流程跑完的 5 簇结果，证据充足的簇给出细胞类型判断。", base)

    print("[2/4] resolution · res=1.6")
    hi = pl.run_pipeline(build(seed=0), pl.PipelineConfig(resolution=1.6, n_markers=N_MARKERS))
    # 拆分/合并数字来自两次标签的真实交叉表，模型只能引用、不能自己发挥
    diff = ai.diff_clusters(
        base.adata.obs[base.cluster_key].astype(str).tolist(),
        hi.adata.obs[hi.cluster_key].astype(str).tolist(),
    )
    p_res = package("resolution", "同一批细胞 · 分辨率 1.0 → 1.6",
                    "细胞没变，只是把 Leiden 分辨率调高，看簇被真实切开。",
                    hi, extra={"resolution_diff": diff, "baseline_resolution": 1.0})

    print("[3/4] unannotated · 只给 Ensembl ID")
    # 真实 10x/Cell Ranger 产物常常只有 Ensembl ID。这里用 crc32 生成稳定假 ID，
    # 保证同一 seed 每次导出得到同一份 ID（Python 内置 hash() 带随机盐，不能用）。
    ens = build(seed=0)
    ens.var_names = pd.Index(
        [f"ENSG{zlib.crc32(str(g).encode()) % 10**9:09d}" for g in ens.var_names]
    )
    ens.var_names_make_unique()
    bad = pl.run_pipeline(ens, pl.PipelineConfig(resolution=1.0, n_markers=N_MARKERS))
    p_bad = package("unannotated", "输入无基因符号注释 · 证据不足场景",
                    "marker 全是 Ensembl ID，AI 读不懂就不猜：整簇拒绝下结论，"
                    "并告诉用户怎么补救（提供 symbol 注释或配置 LLM_API_KEY）。", bad)

    scenarios = [p_normal, p_res, p_bad]
    for p in scenarios:
        dump(OUT_DIR / f"{p['id']}.json", p)

    dump(OUT_DIR / "manifest.json", {
        "generated_from": "scCopilot core.pipeline + agents.orchestrator",
        "note": "全部数字来自本机真实跑批，可用同一 seed 复现：python scripts/export_site_data.py",
        "scenarios": [
            {"id": p["id"], "title": p["title"], "blurb": p["blurb"],
             "n_cells": len(p["umap"]["x"]), "n_clusters": len(p["clusters"]),
             "file": f"{p['id']}.json"}
            for p in scenarios
        ],
    })

    print("[4/4] 校验")
    for p in scenarios:
        refused = sum(1 for v in p["ai"].values() if v["critique"]["status"] == "escalate")
        verdicts: dict[str, int] = {}
        missing: list[str] = []
        for cid, v in p["ai"].items():
            verdict = (v.get("heterogeneity") or {}).get("verdict")
            if not verdict:
                missing.append(cid)
            else:
                verdicts[verdict] = verdicts.get(verdict, 0) + 1
        if missing:
            raise SystemExit(f"{p['id']}: 簇 {missing} 没有异质性判定——"
                             f"run_annotation_workflow 是否没传 adata/cluster_key？")
        print(f"  {p['id']}: {len(p['clusters'])} 簇，其中 {refused} 个被 Critic 判为 escalate"
              f" ｜ 异质性判定 {' '.join(f'{k}={v}' for k, v in sorted(verdicts.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
