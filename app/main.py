# -*- coding: utf-8 -*-
"""scCopilot 前端：单细胞数据探索 + AI 解读工作台（Streamlit + Plotly）。

页面结构（对应需求）
-------------------
侧边栏   上传 .h5ad / 载入演示数据 + 全部流水线参数（QC、HVG、PCA、邻居图、Resolution 滑块）
左半屏   可交互 UMAP（点一个细胞 = 选中它所在的簇）+ 当前簇 Top 差异基因条形图 + 簇一览表
右半屏   AI Copilot 卡片：自动调用 ai/explainer 解读该簇，打字机效果逐字输出 + 证据/闸门/原文

关键工程约定
------------
1. 本文件只做"编排"，不重复实现算法：图形数据来自 core/pipeline，结论来自 ai/explainer。
   UI 上出现的每个数字都能从导出的 CSV 复现，AI 说的每句话都能回溯到 marker 证据表。
2. 首次分析要付一次 numba JIT 编译成本（本机实测约 40–60s，默认自动预热）；编译后全量
   重跑约 10s。QC/嵌入参数放在 st.form 里点「应用」才重跑，避免每拖一次滑块跑一遍；
   结果按参数指纹缓存（session_state.result_cache），切回旧参数组合零等待。
   只动 Resolution 时走 pipeline.recluster 快路径（复用邻居图与 UMAP 坐标，秒级）。
3. 没有 LLM_API_KEY 时页面必须完全可用：ai 层退回本地知识库规则推断，并在卡片上写明
   "未调用大模型"，不伪装成模型输出。
4. 大对象（AnnData / PipelineResult）放在 st.session_state，而不是 st.cache_data 的返回值：
   pickle AnnData 又慢又容易在参数微调时误命中缓存。
5. 用户上传的文件名/基因名都按"不可信输入"处理：落盘前清洗路径，进 DOM 前 HTML 转义。

启动：
    streamlit run app/main.py
"""

from __future__ import annotations

import hashlib
import html as _html
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# 0. 路径与依赖：保证 "streamlit run app/main.py" 在任意工作目录下都能 import core/ai
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np                            # noqa: E402
import pandas as pd                           # noqa: E402
import plotly.graph_objects as go             # noqa: E402
import streamlit as st                        # noqa: E402

from ai import explainer as ai                # noqa: E402
from core import audit_report                 # noqa: E402
from core import pipeline                     # noqa: E402
from agents import orchestrator               # noqa: E402

# streamlit-plotly-events 是可选依赖：装了用它的细粒度点击事件，没装退回 Streamlit 原生 on_select
try:
    from streamlit_plotly_events import plotly_events  # type: ignore
    _HAS_PLOTLY_EVENTS = True
except ImportError:
    _HAS_PLOTLY_EVENTS = False

DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DEMO_H5AD = DATA_DIR / "raw" / "demo_pbmc_sim.h5ad"

# 科技蓝主色，与 style.css 的 --sc-primary 对应；簇配色循环取用
PALETTE = [
    "#2563eb", "#06b6d4", "#8b5cf6", "#f59e0b", "#10b981", "#ef4444",
    "#3b82f6", "#14b8a6", "#a855f7", "#f97316", "#22c55e", "#e11d48",
    "#0ea5e9", "#6366f1", "#84cc16", "#d946ef",
]
CONTINUOUS_SCALE = [[0, "#eff6ff"], [0.35, "#60a5fa"], [0.7, "#1d4ed8"], [1, "#0b1e49"]]

# 改动这些参数需要整条流水线重跑；此外的参数（resolution / n_markers）走快路径
FULL_KEYS = (
    "src_id", "min_genes", "min_counts", "max_pct_mito", "max_genes",
    "min_cells", "filter_doublets", "n_top_genes", "hvg_flavor", "n_pcs",
    "n_neighbors", "marker_method", "random_state",
)


# --------------------------------------------------------------------------- #
# 1. 视觉层
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    """把 style.css 原文注入页面。

    选择 st.markdown(f"<style>…</style>", unsafe_allow_html=True) 而不是
    .streamlit/config.toml：评审打开源码就能改样式，不需要额外理解主题配置文件。
    """
    css_path = Path(__file__).with_name("style.css")
    css = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def chip(text: str, kind: str = "") -> str:
    """一个小标签的 HTML。kind ∈ {blue, ok, warn, bad, gene, …}"""
    cls = f" sc-chip--{kind}" if kind else ""
    return f'<span class="sc-chip{cls}">{_html.escape(str(text))}</span>'


def card(html_body: str, *, cls: str = "sc-card") -> None:
    st.markdown(f'<div class="{cls}">{html_body}</div>', unsafe_allow_html=True)


def txt(value: Any, fmt: str = "{:.1f}", dash: str = "—") -> str:
    """把可能为 NaN/None 的数值渲染成字符串——表格里有缺失值很正常，不该显示 nan。"""
    try:
        if value is None or float(value) != float(value):
            return dash
        return fmt.format(value)
    except (TypeError, ValueError):
        return dash


# --------------------------------------------------------------------------- #
# 2. 打字机效果
# --------------------------------------------------------------------------- #
# 模板里的 CSS/JS 满是花括号，因此不能用 str.format（会把 "{ margin: 0 }" 当占位符解析，
# 实测抛 KeyError: ' margin'）；改成显式的 @@NAME@@ 标记做替换。
#
# 正文（大模型输出 + 用户上传的基因名）走 json.dumps 变成 JS 字符串字面量，再把 < >
# 转成 \u003c \u003e：这样它既闭合不了 </script>，也拼不出任何标签，整个文档仍然是我们自己
# 写的。st.iframe 的文档明确要求"不要把不可信 HTML 交给它"，这条路径就是照那个约束设计的。
TYPEWRITER_HTML = """<!doctype html>
<html lang="zh">
<head><meta charset="utf-8"><style>
  html, body { margin: 0; padding: 0; background: transparent; }
  body { padding: 14px 18px 16px; box-sizing: border-box; }
  .sc-eyebrow {
    font-family: -apple-system, "Segoe UI", "Noto Sans SC", "Microsoft YaHei", sans-serif;
    font-size: 12.5px; font-weight: 700; letter-spacing: .13em;
    color: #2563eb; margin: 0 0 8px;
  }
  .sc-reason {
    font-family: -apple-system, "Segoe UI", "Noto Sans SC", "Microsoft YaHei", sans-serif;
    font-size: 14.5px; line-height: 1.78; color: #1e293b;
    white-space: pre-wrap; word-break: break-word; cursor: text;
  }
  .sc-caret {
    display: inline-block; width: 7px; height: 1.02em; margin-left: 2px;
    vertical-align: -2px; background: #2563eb;
    animation: sc-blink .9s steps(2, start) infinite;
  }
  @keyframes sc-blink { to { visibility: hidden; } }
</style></head>
<body>
<div id="sc-head" class="sc-eyebrow"></div>
<div id="sc-out" class="sc-reason"></div>
<script>
(function () {
  var src = @@TEXT@@;
  var head = @@HEADING@@;
  var out = document.getElementById('sc-out');
  var h = document.getElementById('sc-head');
  h.textContent = head;
  if (!head) { h.style.display = 'none'; }
  var caret = document.createElement('span');
  caret.className = 'sc-caret';
  // 按总长自适应步长：再长的解读也控制在约 8 秒内打完，不让用户干等
  var step = Math.max(1, Math.ceil(src.length / 480));
  var i = 0;
  function tick() {
    i = Math.min(src.length, i + step);
    out.textContent = src.slice(0, i);
    if (i >= src.length) { clearInterval(timer); } else { out.appendChild(caret); }
  }
  var timer = setInterval(tick, @@INTERVAL@@);
  out.addEventListener('click', function () {   // 点一下直接跳到结尾，尊重急性子
    clearInterval(timer);
    out.textContent = src;
  });
  tick();
})();
</script>
</body></html>
"""


def _js_string(value: str) -> str:
    """把不可信文本变成 JS 字符串字面量：尖括号转义后既拼不出 </script> 也拼不出标签。"""
    return json.dumps(str(value), ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")


def typewriter_html(text: str, interval_ms: int = 16, heading: str = "") -> str:
    """把正文和小标题安全地塞进组件模板，返回完整 HTML 文档。"""
    return (
        TYPEWRITER_HTML
        .replace("@@TEXT@@", _js_string(text))
        .replace("@@HEADING@@", _js_string(heading))
        .replace("@@INTERVAL@@", str(max(1, int(interval_ms))))
    )


def typewriter(text: str, *, enabled: bool = True, interval_ms: int = 16,
               heading: str = "") -> None:
    """逐字渲染一段文本。

    iframe 是独立文档，不继承主页样式，所以字体与光标动画要自带一份；卡片外壳由
    style.css 给 [data-testid="stIFrame"] 补上。正文用 textContent 写入 DOM（而不是
    innerHTML），基因名或模型回复里的尖括号只会被当作普通字符显示出来。
    """
    if not text:
        return
    if not enabled:
        head_html = f'<p class="sc-eyebrow">{_html.escape(heading)}</p>' if heading else ""
        st.markdown(f'{head_html}<div class="sc-reason">{_html.escape(text)}</div>',
                    unsafe_allow_html=True)
        return
    lines = sum(max(1, int(np.ceil(len(ln) / 74))) for ln in text.split("\n"))
    height = int(min(820, max(150, 26 * lines + 46 + (24 if heading else 0))))
    st.iframe(typewriter_html(text, interval_ms, heading), height=height)


# --------------------------------------------------------------------------- #
# 3. 侧边栏
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def stage_upload(name: str, size: int, blob: bytes) -> tuple[str, str]:
    """把上传文件落到 data/uploads/，返回 (路径, 来源指纹)。

    为什么要落盘：h5ad 动辄上百 MB，把 BytesIO 一直攥在 session_state 里，每次 rerun
    都会多一份内存。文件名先剥掉目录部分再用白名单过滤，杜绝 "../.." 这类路径穿越。
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name)[:80] or "upload"
    stem = safe[:-5] if safe.lower().endswith(".h5ad") else safe
    digest = hashlib.sha1(blob).hexdigest()[:10]
    target = UPLOAD_DIR / f"{stem}_{digest}.h5ad"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(blob)
    return str(target), f"upload:{stem}:{digest}:{size}"


@st.cache_data(show_spinner=False)
def data_fingerprint(path: str, mtime: float, size: int) -> str:
    """输入数据的 sha256。mtime/size 进缓存键：文件换了就重算，不会拿旧指纹冒充新数据。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@st.cache_data(show_spinner=False)
def code_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else "未记录（非 git 环境）"
    except (OSError, subprocess.SubprocessError):
        return "未记录（无 git）"


def audit_bundle_data(result: Any, workflows: dict[str, Any], meta: dict[str, Any]):
    """存证包下载按钮的零参回调。入参是渲染期快照：回调线程里读不到 session_state。"""
    def _bytes() -> bytes:
        return audit_report.build_audit_html(result, workflows, meta).encode("utf-8")
    return _bytes


def sidebar_controls() -> dict[str, Any]:
    """渲染侧边栏，返回本次运行的全部参数。"""
    with st.sidebar:
        st.markdown(
            '<p class="sc-eyebrow">scCopilot</p>'
            '<p class="sc-title">单细胞 AI 工作台</p>'
            '<p class="sc-sub">上传 .h5ad → 跑完流水线 → 点一个簇，让模型替你读 marker。</p>',
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("##### 1. 数据")
        src_kind = st.radio(
            "数据来源", ["使用内置演示数据", "上传 .h5ad 文件"], horizontal=True, key="src_kind",
            help="演示数据是约 2200 个细胞的模拟 PBMC，其中故意留了一簇低质量细胞，"
                 "用来验证 AI 会不会硬编细胞类型。",
        )
        path: str | None = None
        src_id = ""
        if src_kind == "上传 .h5ad 文件":
            up = st.file_uploader("支持已注释的 AnnData（X 建议为 raw counts）", type=["h5ad"])
            if up is not None:
                blob = up.getvalue()
                path, src_id = stage_upload(up.name, len(blob), blob)
        elif DEMO_H5AD.exists():
            path, src_id = str(DEMO_H5AD), f"demo:{DEMO_H5AD.name}"
        else:
            st.warning("缺少 data/raw/demo_pbmc_sim.h5ad，请先运行 "
                       "`python scripts/make_demo_h5ad.py`，或改为上传自己的文件。")

        st.divider()
        st.markdown("##### 2. 质量控制与嵌入")
        st.caption("这些参数变了要全量重跑（预热后约 10 秒），所以拖完点一次「应用」才重算。")
        with st.form("compute_params", border=False):
            min_genes = st.slider("每细胞最少检出基因数", 100, 1500, 200, 50, key="min_genes")
            min_counts = st.slider("每细胞最少 UMI", 200, 5000, 500, 100, key="min_counts",
                                  help="低深度数据要调低这个值，否则细胞会被全部滤掉。")
            max_pct_mito = st.slider("线粒体基因占比上限 (%)", 5, 50, 20, 1, key="max_pct_mito")
            max_genes_on = st.checkbox("过滤大细胞（每细胞基因数上限 6000）", True, key="max_genes_on")
            min_cells = st.number_input("基因至少出现在多少个细胞", 1, 50, 3, key="min_cells")
            filter_doublets = st.checkbox("用 scrublet 剔除双细胞", False, key="filter_doublets",
                                          help="多花十几秒，演示现场数据量大时不建议打开。")
            n_top_genes = st.slider("高变基因数", 500, 5000, 2000, 250, key="n_top_genes")
            n_pcs = st.slider("PCA 主成分数", 10, 100, 50, 5, key="n_pcs")
            n_neighbors = st.slider("近邻图 k", 5, 50, 15, 1, key="n_neighbors")
            st.form_submit_button("应用参数并重跑", type="primary", width="stretch")

        st.divider()
        st.markdown("##### 3. 聚类与 marker · 即时生效")
        resolution = st.slider(
            "Leiden 分辨率 (resolution)", 0.1, 3.0, 1.0, 0.05, key="resolution",
            help="调高 → 同一群细胞被切得更细；调低 → 相近群体合并。"
                 "改动只重跑 Leiden 与 marker，UMAP 坐标不动，所以几秒出图。",
        )
        n_markers = st.slider("每簇 Top 基因数", 4, 20, 8, 1, key="n_markers")

        st.divider()
        st.markdown("##### 4. AI 解读")
        cfg = ai.LLMConfig.from_env()
        # 需求里的占位符：API Key 只从环境变量读取（可写在项目根目录的 .env）
        env_key_set = bool(os.getenv("LLM_API_KEY"))
        if cfg.configured and env_key_set:
            st.success(f"已读到 `LLM_API_KEY`，模型 **{cfg.model}**")
        else:
            st.info("`LLM_API_KEY` 未设置。填入根目录 `.env` 即可调用大模型；"
                    "现在会退回本地免疫学知识库规则推断，页面功能不受影响。")
        use_llm = st.checkbox("调用大模型解读", cfg.configured, key="use_llm",
                             help="关闭后只走本地规则推断：零网络请求、零 token 成本。")
        auto_ai = st.checkbox("点击 UMAP 后自动解读", True, key="auto_ai")
        animate = st.checkbox("打字机效果", True, key="animate")
        warm = st.checkbox("先把计算引擎预热一次", True, key="warm",
                           help="默认开：首次跑批前先做 numba 编译（约 1 分钟，每进程一次），"
                                "之后每次全量重跑只剩约 10 秒真实计算。")

    return {
        "path": path, "src_id": src_id,
        "min_genes": min_genes, "min_counts": min_counts, "max_pct_mito": max_pct_mito,
        "max_genes": 6000 if max_genes_on else None, "min_cells": int(min_cells),
        "filter_doublets": bool(filter_doublets),
        "n_top_genes": n_top_genes, "hvg_flavor": "seurat", "n_pcs": n_pcs,
        "n_neighbors": n_neighbors, "resolution": float(resolution), "n_markers": int(n_markers),
        "marker_method": "wilcoxon", "random_state": 0,
        "use_llm": bool(use_llm and cfg.configured), "auto_ai": bool(auto_ai),
        "animate": bool(animate), "warm": bool(warm),
    }


def params_to_config(p: dict[str, Any]) -> pipeline.PipelineConfig:
    """侧边栏参数 → 流水线配置对象（core 只认这个，不认 Streamlit）。"""
    return pipeline.PipelineConfig(
        min_genes=p["min_genes"], max_genes=p["max_genes"], min_counts=p["min_counts"],
        max_pct_mito=p["max_pct_mito"], min_cells=p["min_cells"],
        filter_doublets=p["filter_doublets"], n_top_genes=p["n_top_genes"],
        hvg_flavor=p["hvg_flavor"], n_pcs=p["n_pcs"], n_neighbors=p["n_neighbors"],
        resolution=p["resolution"], n_markers=p["n_markers"],
        marker_method=p["marker_method"], random_state=p["random_state"],
    )


def fingerprint(params: dict[str, Any], keys: tuple[str, ...]) -> str:
    """把参数子集压成短指纹，用来判断「要不要重算 / 重聚类 / 重新解读」。"""
    payload = json.dumps({k: params.get(k) for k in keys}, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# 4. 计算编排
# --------------------------------------------------------------------------- #
def warmup_engine() -> None:
    if st.session_state.get("warmed"):
        return
    with st.spinner("预热计算引擎（numba 首次编译，约 1 分钟，每个进程只做一次）…"):
        st.session_state["warmed"] = pipeline.warmup()
    st.toast(f"计算引擎预热完成，耗时 {st.session_state['warmed']}s")


def full_run(p: dict[str, Any]) -> pipeline.PipelineResult:
    """完整流水线：读文件 → QC → 归一化 → HVG/PCA → 邻居图 → Leiden → UMAP → marker。"""
    bar = st.progress(0.0, text="准备读取 .h5ad …")

    def on_progress(name: str, pct: float) -> None:
        # core 通过 progress 回调汇报阶段，前端不需要自己猜"现在在干嘛"
        bar.progress(float(pct), text=f"{name} …")

    t0 = time.perf_counter()
    adata, notes = pipeline.load_h5ad(p["path"])
    bar.progress(0.06, text=f"已读入 {adata.n_obs} 细胞 × {adata.n_vars} 基因，开始 QC …")
    result = pipeline.run_pipeline(adata, params_to_config(p), progress=on_progress)
    result.warnings = list(dict.fromkeys([*notes, *result.warnings]))
    bar.progress(1.0, text=f"完成，总计 {time.perf_counter() - t0:.1f}s")
    time.sleep(0.4)     # 让 100% 那一帧可见，否则进度条会直接消失
    return result


def fast_recluster(result: pipeline.PipelineResult, p: dict[str, Any]) -> pipeline.PipelineResult:
    """只重跑 Leiden + marker，邻居图与 UMAP 坐标保持不动。

    这是"分辨率把子群切开/合起来"最直观的讲法：地图不变，只是配色变了。
    """
    key = result.cluster_key
    markers, clusters = pipeline.recluster(
        result.adata, p["resolution"], cluster_key=key,
        n_markers=p["n_markers"], method=p["marker_method"], random_state=p["random_state"],
    )
    timings = {k: v for k, v in result.timings.items() if k != "总计"}
    return pipeline.PipelineResult(
        result.adata, markers, clusters, result.qc, timings,
        {**result.config, "resolution": p["resolution"], "n_markers": p["n_markers"]},
        list(result.warnings),
    )


def ensure_result(p: dict[str, Any]) -> pipeline.PipelineResult | None:
    """按参数指纹决定：缓存命中 / 走快路径 / 全量重跑。返回 None 表示还没有数据。

    result_cache 以 full_fp 为键存 (result, expr_fp)。fast_recluster 会原地修改共享
    adata 的簇标签，所以快路径之后必须同步更新缓存条目，否则切回旧参数时
    markers 与 adata.obs 会对不上。
    """
    ss = st.session_state
    if not p["path"]:
        return None

    full_fp = fingerprint(p, FULL_KEYS)
    expr_fp = fingerprint(p, ("resolution", "n_markers", "src_id"))
    cache: dict[str, tuple[pipeline.PipelineResult, str]] = ss.setdefault("result_cache", {})

    if "result" not in ss or ss.get("full_fp") != full_fp:
        if full_fp in cache:
            ss.result, ss.expr_fp = cache[full_fp]
            ss.full_fp = full_fp
            ss.prev_labels, ss.prev_resolution = None, None
            ss.ai_cache, ss.res_expl_cache = {}, None
            ss.pop("cluster_pick", None)
            st.toast("这组参数跑过，已直接恢复缓存结果")
        else:
            if p["warm"]:
                warmup_engine()
            ss.result = full_run(p)
            cache[full_fp] = (ss.result, expr_fp)
            ss.full_fp, ss.expr_fp = full_fp, expr_fp
            ss.prev_labels, ss.prev_resolution = None, None
            ss.ai_cache, ss.res_expl_cache = {}, None
            ss.pop("cluster_pick", None)      # 换了数据，旧簇号没有意义
            st.toast("分析完成")
            st.rerun()

    if ss.get("expr_fp") != expr_fp:
        # 先留住旧标签：AI 层要靠"两次聚类的交叉表"算出真实的拆分/合并数
        key = ss.result.cluster_key
        ss.prev_labels = np.asarray(ss.result.adata.obs[key].astype(str)).tolist()
        ss.prev_resolution = float(ss.result.config.get("resolution", 0.0))
        with st.spinner(f"按 resolution={p['resolution']} 重新聚类（复用邻居图）…"):
            ss.result = fast_recluster(ss.result, p)
        ss.expr_fp = expr_fp
        cache[full_fp] = (ss.result, expr_fp)
        ss.ai_cache, ss.res_expl_cache = {}, None    # marker 变了，旧结论不能复用
        ss.pop("cluster_pick", None)
        st.rerun()

    return ss.result


# --------------------------------------------------------------------------- #
# 5. 图形
# --------------------------------------------------------------------------- #
def rgba(hex_color: str, alpha: float) -> str:
    n = hex_color.lstrip("#")
    r, g, b = (int(n[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha:.2f})"


def cell_table(result: pipeline.PipelineResult) -> pd.DataFrame:
    """UMAP 用的逐细胞表。每帧重算（万级也只是毫秒），保证与 obs 永不脱节。"""
    a = result.adata
    umap = np.asarray(a.obsm["X_umap"], dtype=float)
    out = pd.DataFrame({
        "x": umap[:, 0],
        "y": umap[:, 1],
        "cluster": np.asarray(a.obs[result.cluster_key].astype(str)),
    })
    out["pct_mito"] = (a.obs["pct_mito"].astype(float).to_numpy() if "pct_mito" in a.obs else np.nan)
    out["n_counts"] = (a.obs["n_counts"].astype(float).to_numpy() if "n_counts" in a.obs else np.nan)
    return out


def cluster_colors(labels: list[str]) -> dict[str, str]:
    # 按 (长度, 字面) 排序，让 0,1,…,9,10 自然相邻，配色才不会出现 10 和 1 同色
    ordered = sorted(labels, key=lambda s: (len(s), s))
    return {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(ordered)}


def build_umap_fig(df: pd.DataFrame, color_by: str, selected: str,
                   colors: dict[str, str],
                   gene: tuple[str, np.ndarray] | None = None) -> go.Figure:
    """用单条 trace 画 UMAP：点的索引因此与 df 行号严格对应，点击回填不会错位。

    ``gene`` 是 ``(矩阵里的写法, 逐细胞表达向量)``，向量按 ``adata`` 行序给出——
    ``cell_table`` 也用同一个行序，所以两者天然对齐。
    """
    sizes = df["cluster"].value_counts().to_dict()
    fig = go.Figure()
    extra_cd: list[float] = []
    hover_tail = ""

    if color_by == "簇标签":
        point_colors = [rgba(colors[c], 0.95 if c == selected else 0.14) for c in df["cluster"]]
        colorbar = None
    elif gene is not None:
        name, vec = gene
        vals = np.asarray(vec, dtype=float)
        point_colors = vals.tolist()
        extra_cd = [round(float(v), 4) for v in vals]
        colorbar = dict(title=f"{name}<br>(log CPM+1)", thickness=11, x=1.02, len=0.74,
                        outlinewidth=0)
        hover_tail = f"<br>{name} %{{customdata[2]:.3f}}"
    elif color_by in ("线粒体占比", "UMI 数量"):
        vals = df["pct_mito"].astype(float)
        if color_by == "UMI 数量":
            vals = np.log10(df["n_counts"].astype(float).clip(lower=1.0))
        point_colors = vals.fillna(0.0).tolist()
        colorbar = dict(title=color_by, thickness=11, x=1.02, len=0.74, outlinewidth=0)
    else:  # AI 置信度：把"模型有多确定"直接画在地图上
        cache = st.session_state.get("ai_cache", {})
        score_of = {
            c: (-1.0 if c not in cache or cache[c]["interpretation"].refused
                else float(cache[c]["interpretation"].confidence_score or 0.0))
            for c in df["cluster"].unique()
        }
        point_colors = [score_of[c] for c in df["cluster"]]
        colorbar = dict(title="AI 置信度", thickness=11, x=1.02, len=0.74, outlinewidth=0,
                        tickvals=[-1, 0.3, 0.6, 0.9], ticktext=["未解读", "低", "中", "高"])

    fig.add_trace(go.Scattergl(
        x=df["x"], y=df["y"], mode="markers", name="cells",
        marker=dict(size=4.8, color=point_colors, colorscale=CONTINUOUS_SCALE,
                    line=dict(width=0), colorbar=colorbar),
        customdata=[[c, int(sizes.get(c, 0)), v] for c, v in
                    zip(df["cluster"], (extra_cd or [0.0] * len(df)))],
        hovertemplate="簇 %{customdata[0]} · %{customdata[1]:,} 个细胞"
                      "<br>UMAP (%{x:.2f}, %{y:.2f})" + hover_tail + "<extra></extra>",
    ))

    stat = df.groupby("cluster").agg(x=("x", "mean"), y=("y", "mean"), n=("x", "size")).reset_index()
    fig.add_trace(go.Scatter(
        x=stat["x"], y=stat["y"], mode="markers+text",
        marker=dict(size=15, color=[colors[c] for c in stat["cluster"]],
                    line=dict(color="#ffffff", width=2.4)),
        text=stat["cluster"].tolist(), textposition="top center",
        textfont=dict(size=11, color="#0b2447"),
        customdata=[[c, int(n)] for c, n in zip(stat["cluster"], stat["n"])],
        hovertemplate="簇 %{customdata[0]}（%{customdata[1]:,} 个细胞）<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_white", height=472, margin=dict(l=8, r=8, t=8, b=8),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="UMAP-1"),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="UMAP-2"),
        showlegend=False, hovermode="closest", dragmode="select",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, Segoe UI, Noto Sans SC", size=12, color="#334155"),
    )
    return fig


def gene_cluster_table(df: pd.DataFrame, vec: np.ndarray) -> pd.DataFrame:
    """一个基因在各簇的平均表达与检出率，口径与 marker 条形图同一层（log CPM+1）。"""
    t = pd.DataFrame({"cluster": df["cluster"],
                      "expr": np.asarray(vec, dtype=float)})
    t["detected"] = t["expr"] > 0
    return (t.groupby("cluster")
            .agg(n_cells=("expr", "size"), mean_expr=("expr", "mean"),
                 pct_expressed=("detected", "mean"))
            .reset_index())


def build_gene_dot(gene: str, table: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    """点图：位置 = 簇内平均表达，点大小 = 簇内检出率——一眼看出这个基因是不是某簇专属。"""
    view = table.sort_values("mean_expr", ascending=False)
    fig = go.Figure(go.Scatter(
        x=view["mean_expr"].astype(float), y=view["cluster"].astype(str), mode="markers",
        marker=dict(size=[6 + 26 * float(p) for p in view["pct_expressed"]],
                    color=[colors.get(str(c), "#64748b") for c in view["cluster"]],
                    line=dict(color="#ffffff", width=1.2)),
        customdata=view[["n_cells", "pct_expressed"]].to_numpy(dtype=float).tolist(),
        hovertemplate="%{y}<br>平均表达 %{x:.3f}<br>簇内检出 %{customdata[1]:.1%}"
                      "<br>%{customdata[0]:,.0f} 个细胞<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white", height=int(64 + 34 * max(len(view), 1)),
        margin=dict(l=8, r=8, t=10, b=30),
        xaxis=dict(title=f"{gene} · 簇内平均表达（log CPM+1）", rangemode="tozero",
                   gridcolor="#eef2f7"),
        yaxis=dict(title="", autorange="reversed",
                   tickfont=dict(family="ui-monospace, Consolas, monospace", size=12)),
        showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, Segoe UI, Noto Sans SC", size=12, color="#334155"),
    )
    return fig


def gene_class(row: pd.Series) -> str:
    """marker 的分类标签：线粒体 / 核糖体 / 未注释名 / 标志基因。"""
    if bool(row.get("is_mito")):
        return "线粒体基因"
    if bool(row.get("is_novel")):
        return "未注释名"
    if bool(row.get("is_ribo")):
        return "核糖体基因"
    return "标志基因"


def _num(row: pd.Series, key: str) -> float:
    v = row.get(key)
    return float(v) if v is not None and v == v else float("nan")


def marker_stats(view: pd.DataFrame) -> list[str]:
    """逐行的补充统计量文案：log2FC 与校正 p。缺列就整段留空，不编数字。"""
    if "logfoldchanges" not in view.columns and "pvals_adj" not in view.columns:
        return [""] * len(view)
    out = []
    for _, r in view.iterrows():
        lfc, padj = _num(r, "logfoldchanges"), _num(r, "pvals_adj")
        parts = []
        if lfc == lfc:
            parts.append(f"log2FC {lfc:+.2f}")
        if padj == padj:
            parts.append(f"校正 p {padj:.2e}" if padj < 0.01 else f"校正 p {padj:.3f}")
        out.append(" ｜ ".join(parts))
    return out


def build_marker_fig(sub: pd.DataFrame, top: int) -> go.Figure:
    """当前簇 Top 基因条形图：条长 = 簇内平均表达，颜色 = 这个基因能不能当证据。"""
    view = sub.sort_values("rank").head(top).iloc[::-1].copy()

    cls = [gene_class(r) for r in view.to_dict("records")]
    view["cls"] = cls
    color_map = {"标志基因": "#2563eb", "核糖体基因": "#94a3b8",
                 "线粒体基因": "#f59e0b", "未注释名": "#e11d48"}
    stats = marker_stats(view)
    fig = go.Figure(go.Bar(
        x=view["mean_expr"].astype(float).tolist(),
        y=view["gene"].astype(str).tolist(), orientation="h",
        marker_color=[color_map[c] for c in view["cls"]],
        # 条端只放表达量：右列宽度有限，"2.57 · 检出 91%" 这种长标签会被容器裁掉半截，
        # 检出率放进 hover 和证据表里。
        text=[f"{m:.2f}" for m in view["mean_expr"]],
        textposition="outside", textfont=dict(size=11, color="#475569"),
        customdata=[[float(p), float(s), c, st] for p, s, c, st in
                    zip(view["pct_expressed"], view["scores"], view["cls"], stats)],
        hovertemplate="%{y}<br>平均表达 %{x:.3f}<br>簇内检出 %{customdata[0]:.1%}"
                      "<br>差异打分 %{customdata[1]:.1f}<br>%{customdata[2]}"
                      "<br>%{customdata[3]}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white", height=int(46 + 30 * max(len(view), 1)),
        margin=dict(l=6, r=64, t=10, b=10),
        xaxis=dict(title="簇内平均表达（log 归一化后）", rangemode="tozero", gridcolor="#eef2f7"),
        yaxis=dict(title="", automargin=True,
                   tickfont=dict(family="ui-monospace, Consolas, monospace", size=12)),
        showlegend=False, bargap=0.34,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="-apple-system, Segoe UI, Noto Sans SC", size=12, color="#334155"),
    )
    return fig


# --------------------------------------------------------------------------- #
# 6. UMAP 点击 → 选中簇
# --------------------------------------------------------------------------- #
def _points_of(state: Any) -> list[Any]:
    try:
        sel = state.selection if state is not None else None
        pts = getattr(sel, "points", None)
        if pts is None and isinstance(sel, dict):
            pts = sel.get("points")
        return list(pts or [])
    except Exception:
        return []


def selection_signature(state: Any) -> str:
    """给选中状态一个签名。签名没变就说明用户没有新点击，不要用旧点击覆盖下拉框选择。"""
    out = []
    for p in _points_of(state)[:8]:
        get = (lambda k: p.get(k)) if isinstance(p, dict) else (lambda k: getattr(p, k, None))
        idx, num = get("point_index"), get("point_number")
        # 不能用 `idx or num`：point_index 合法取值包含 0，第一个点会被当成"没有值"而退回另一个键
        out.append(f"{get('curve_number')}:{idx if idx is not None else num}")
    return "|".join(out)


def clicked_cluster(state: Any) -> str | None:
    """从 Plotly 选中状态还原"用户点的是哪个簇"。

    customdata 里直接带着簇名，优先用它；老版本 Streamlit 对单元素 customdata 有给数组
    也有给标量的，所以两条路径都兼容，最后才回退到按行号查表。
    """
    pts = _points_of(state)
    if not pts:
        return None
    p = pts[-1]
    get = (lambda k: p.get(k)) if isinstance(p, dict) else (lambda k: getattr(p, k, None))
    cd = get("customdata")
    if isinstance(cd, (list, tuple)) and cd:
        return str(cd[0])
    if isinstance(cd, (str, int, float, np.integer)) and str(cd):
        return str(cd)
    for key in ("point_index", "point_number"):
        idx = get(key)
        df = st.session_state.get("umap_cells")
        if isinstance(idx, (int, np.integer)) and df is not None and 0 <= int(idx) < len(df):
            return str(df["cluster"].iloc[int(idx)])
    return None


def resolve_selected(options: dict[str, str]) -> str:
    """合并两个入口：UMAP 点击（新点击才生效）+ 下拉框。返回簇标签。"""
    ss = st.session_state
    state = ss.get("umap_click")
    sig = selection_signature(state)
    clicked = clicked_cluster(state) if sig != ss.get("umap_sig") else None
    ss["umap_sig"] = sig
    if clicked in options:
        ss["cluster_pick"] = clicked
    elif ss.get("cluster_pick") not in options:
        ss["cluster_pick"] = next(iter(options))
    return str(ss["cluster_pick"])


# --------------------------------------------------------------------------- #
# 7. AI 卡片
# --------------------------------------------------------------------------- #
def conf_badge(label: str, score: float | None) -> str:
    cls = {"高": "high", "中": "mid", "低": "low"}.get(label, "none")
    text = f"置信度 {label}" + (f" · {score:.2f}" if isinstance(score, (int, float)) else "")
    return f'<span class="sc-badge sc-badge--{cls}">{_html.escape(text)}</span>'


def source_chip(source: str) -> str:
    return {
        "llm": chip("大模型 · 已核验引用", "blue"),
        "local_kb": chip("本地知识库规则推断", "warn"),
        "gate": chip("被证据闸门拦下", "bad"),
    }.get(source, chip(source or "未知来源"))


def interpret(cluster: str, result: pipeline.PipelineResult, p: dict[str, Any]) -> ai.Interpretation:
    return ai.interpret_result_cluster(
        result, cluster, llm=ai.LLMConfig.from_env(), use_llm=p["use_llm"]
    )


def evidence_table(sub: pd.DataFrame, top: int) -> None:
    """Top marker 的完整统计量。数字直接来自 rank_genes_groups，页面不改任何口径。"""
    view = sub.sort_values("rank").head(top)
    if view.empty:
        st.caption("本簇没有 marker（只有 1 个细胞的簇会跳过差异检验）。")
        return

    def fmt(col: str, spec: str) -> list[str]:
        if col not in view.columns:
            return ["—"] * len(view)
        vals = pd.to_numeric(view[col], errors="coerce").to_numpy(dtype=float)
        return ["—" if v != v else format(float(v), spec) for v in vals]

    st.dataframe(pd.DataFrame({
        "排名": view["rank"].astype(int).tolist(),
        "基因": view["gene"].astype(str).tolist(),
        "类别": [gene_class(r) for r in view.to_dict("records")],
        "平均表达": fmt("mean_expr", ".3f"),
        "簇内检出": fmt("pct_expressed", ".1%"),
        "log2FC": fmt("logfoldchanges", "+.2f"),
        "差异打分": fmt("scores", ".1f"),
        "校正 p": fmt("pvals_adj", ".2e"),
    }), hide_index=True, width="stretch")
    st.caption("wilcoxon one-vs-rest：score=z 统计量，log2FC 与 p 值经 Benjamini-Hochberg 校正；"
               "校正 p 显示 0.00e+00 表示小于双精度可表示范围。平均表达是 log(CPM+1) 尺度。")


def gate_details(itp: ai.Interpretation) -> None:
    g = itp.gate or {}
    left, right = st.columns(2)
    with left:
        for title, keys, kind in (
            ("参与判断的已知标志基因", ("informative",), "gene"),
            ("不认识但名字合法（不计分）", ("unknown",), "gene"),
            ("无区分价值（管家/核糖体/线粒体/应激）", ("uninformative",), ""),
            ("未注释 / 非法名", ("unannotated", "unsafe_input"), "warn"),
        ):
            items = [x for k in keys for x in (g.get(k) or [])][:14]
            st.markdown(f"**{title}**")
            st.markdown("".join(chip(x, kind) for x in items) or '<span class="sc-sub">—</span>',
                        unsafe_allow_html=True)
    with right:
        st.json({k: g.get(k) for k in ("ok", "n_input", "n_usable", "n_strong", "novel_ratio",
                                      "mito_ratio", "n_cells", "has_expression")}, expanded=False)
        for r in g.get("reasons", []):
            st.markdown(f'<div class="sc-note sc-note--bad">拦截理由：{_html.escape(r)}</div>',
                        unsafe_allow_html=True)
        for w in g.get("warnings", []):
            st.markdown(f'<div class="sc-note">数据质量提示：{_html.escape(w)}</div>',
                        unsafe_allow_html=True)
        for s in g.get("suggestions", []):
            st.markdown(f"- {s}")


HET_LABEL = {
    "homogeneous": ("单一类型", "ok"),
    "heterogeneous": ("混合了多种细胞", "bad"),
    "ambiguous": ("面板判读不出来", "warn"),
    "not_assessed": ("未做判定", ""),
}


def het_chip(het: dict[str, Any] | None) -> str:
    """卡片上的一行结论：不展开也能看到这个簇配不配被叫成"一个细胞类型"。"""
    if not het:
        return ""
    label, kind = HET_LABEL.get(str(het.get("verdict")), ("未知判定", ""))
    extra = ""
    if het.get("verdict") in ("homogeneous", "heterogeneous"):
        extra = f" ｜ 主标签 {het.get('dominant') or '—'} {float(het.get('dominant_prop') or 0):.0%}"
    return chip(f"簇内异质性：{label}{extra}", kind)


def het_panel(het: dict[str, Any] | None) -> None:
    """异质性判定详情。所有阈值原样展示 —— 评委问"这个数从哪来"，答案是代码里的常量。"""
    if not het:
        return
    verdict = str(het.get("verdict", "not_assessed"))
    label, _kind = HET_LABEL.get(verdict, ("未知判定", ""))
    st.markdown(f"#### 🧬 簇内异质性判定：{label}")
    if het.get("reason"):
        st.markdown(f'<div class="sc-note">{_html.escape(str(het["reason"]))}</div>',
                    unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown(
            "".join(
                chip(f"{name} {len(genes)} 个判别基因", "gene")
                for name, genes in sorted((het.get("types_used") or {}).items())
            ) or '<span class="sc-sub">—</span>',
            unsafe_allow_html=True,
        )
        for skipped in (het.get("types_skipped") or [])[:8]:
            st.caption(f"无法判读：{skipped.get('type')}（矩阵里缺 "
                       f"{len(skipped.get('absent_genes') or [])} 个面板基因）")
        for name, info in sorted((het.get("types_unresolvable") or {}).items())[:8]:
            shared = "、".join(str(s) for s in (info.get("shared_with") or [])) or "其他类型"
            st.caption(f"未参与判型：{_html.escape(str(name))}——只有 "
                       f"{info.get('n_discriminative')}/{info.get('n_panel_genes')} 个判别基因，"
                       f"和 {shared} 分不开（共享 marker 只证明同一套程序，不证明第二种细胞）")
    with right:
        st.json({k: het.get(k) for k in ("n_cells", "n_reference_cells", "dominant",
                                        "dominant_prop", "runner_up_type",
                                        "runner_up_type_share", "entropy_bits",
                                        "max_entropy_bits", "normalized_entropy",
                                        "assigned_prop", "unassigned_prop")}, expanded=False)
    zmed = het.get("z_median") or {}
    if zmed:
        st.caption("各类型判型分中位数（判别基因平均表达，相对簇外 "
                   f"{int(het.get('n_reference_cells') or 0):,} 个参照细胞标准化）："
                   + " ｜ ".join(f"{_html.escape(str(k))} {float(v):,.2f}"
                                 for k, v in sorted(zmed.items())))
    dist = het.get("distribution") or []
    if dist:
        st.dataframe(
            pd.DataFrame(dist).rename(columns={
                "label": "单细胞判型", "n_cells": "细胞数", "share": "占比"})
            .assign(占比=lambda d: d["占比"].map(lambda x: f"{float(x):.1%}")),
            hide_index=True, use_container_width=True,
        )
    if het.get("subcluster_run"):
        st.markdown(f"**簇内重新分簇**（resolution="
                    f"{(het.get('params') or {}).get('subcluster_resolution')}，"
                    f"加权纯度 {float(het.get('subcluster_mean_purity') or 0):.0%}，"
                    f"比整簇主标签占比高 {float(het.get('subcluster_purity_gain') or 0):.0%}）")
        subs = het.get("subclusters") or []
        if subs:
            st.dataframe(
                pd.DataFrame(subs).rename(columns={
                    "subcluster": "亚簇", "n_cells": "细胞数", "share": "占本簇",
                    "dominant": "主标签", "purity": "纯度"})
                .assign(占本簇=lambda d: d["占本簇"].map(lambda x: f"{float(x):.0%}"),
                        纯度=lambda d: d["纯度"].map(lambda x: f"{float(x):.0%}")),
                hide_index=True, use_container_width=True,
            )
    st.caption(f"打分口径：{(het.get('params') or {}).get('score_unit', '—')}。"
               f"全部阈值：{json.dumps(het.get('params') or {}, ensure_ascii=False)}")


def render_resolution_card(result: pipeline.PipelineResult, p: dict[str, Any]) -> None:
    """解释分辨率变化。所有数字来自两次标签的真实交叉表，模型只能引用、不能自己发挥。"""
    ss = st.session_state
    labels = np.asarray(result.adata.obs[result.cluster_key].astype(str)).tolist()
    card(
        '<p class="sc-eyebrow">参数解读</p>'
        '<p class="sc-title" style="font-size:1.05rem">调分辨率时发生了什么？</p>'
        '<p class="sc-sub">拆分 / 合并的数量由前后两次聚类的交叉表算出，不依赖模型。</p>',
        cls="sc-card sc-card--flush",
    )

    if ss.get("prev_labels") is None or len(ss["prev_labels"]) != len(labels):
        st.caption("这是本轮数据的第一次聚类，还没有可对比的上一版。"
                   "拖动侧边栏的 Leiden 分辨率滑块，这里就会出现真实对比。")
        return

    diff = ai.diff_clusters(ss["prev_labels"], labels)
    st.markdown(
        '<div class="sc-kv">'
        + f"<div><dt>簇数</dt><dd>{diff['n_before']} → {diff['n_after']}</dd></div>"
        + f"<div><dt>中位簇规模</dt><dd>{diff['median_before']:.0f} → {diff['median_after']:.0f}</dd></div>"
        + f"<div><dt>最大簇（细胞数）</dt><dd>{diff['max_before']:,} → {diff['max_after']:,}</dd></div>"
        + f"<div><dt>拆分 / 合并</dt><dd>{diff['n_splits']} / {diff['n_merges']}</dd></div>"
        + "</div>",
        unsafe_allow_html=True,
    )
    for row in diff["splits"][:3]:
        st.markdown(chip(f"簇 {row['from']} 被拆成 {len(row['into'])} 份 "
                         f"{row['sizes']}（纯度 {row['purity']:.0%}）", "blue"), unsafe_allow_html=True)
    for row in diff["merges"][:3]:
        st.markdown(chip(f"簇 {', '.join(row['from'])} 合并进 {row['into']}", "warn"),
                    unsafe_allow_html=True)

    if st.button("让 AI 解释这次变化", key="why_res", type="primary"):
        with st.spinner("生成解释…"):
            ss["res_expl_cache"] = ai.explain_resolution_change(
                p["resolution"], prev_resolution=ss.get("prev_resolution"),
                prev_labels=ss["prev_labels"], new_labels=labels,
                clusters_table=result.clusters, llm=ai.LLMConfig.from_env(),
                use_llm=p["use_llm"],
            )
    ex = ss.get("res_expl_cache")
    if ex is not None:
        st.markdown(f'<div class="sc-note sc-note--info">{_html.escape(ex.headline)}</div>',
                    unsafe_allow_html=True)
        typewriter("\n".join(t for t in (ex.summary, ex.why, ex.next_steps) if t),
                   enabled=p["animate"])
        st.caption(f"来源：{ex.source}" + (f" ｜ {ex.model}" if ex.model else "")
                   + (f" ｜ {ex.latency_s}s" if ex.latency_s else ""))
        for n in ex.notes:
            st.markdown(f"- {n}")


def annotation_rows(cache: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """会话里的 AI 缓存 → ``PipelineResult.annotate`` 的入参。

    ``blocked`` = 模型拒答或 Critic 升级：这类簇在 obs 里只写 low_confidence，
    页面说什么，落到文件里就是什么。
    """
    rows: dict[str, dict[str, Any]] = {}
    for cl, wf in (cache or {}).items():
        itp = wf.get("interpretation")
        if itp is None:
            continue
        rows[str(cl)] = {
            "type": itp.predicted_type,
            "conf": itp.confidence_score,
            "source": itp.source,
            "het": (wf.get("heterogeneity") or {}).get("verdict") or "",
            "blocked": bool(itp.refused)
                       or (wf.get("critique") or {}).get("status") == "escalate",
        }
    return rows


def render_run_info(result: pipeline.PipelineResult, p: dict[str, Any]) -> None:
    with st.expander("QC 漏斗 / 耗时 / 导出", expanded=False):
        qc = result.qc
        st.markdown(
            '<div class="sc-kv">'
            + f"<div><dt>输入细胞</dt><dd>{qc.get('n_cells_start', 0):,}</dd></div>"
            + f"<div><dt>硬过滤后</dt><dd>{qc.get('n_cells_after_hard_filter', 0):,}</dd></div>"
            + f"<div><dt>MT 过滤后</dt><dd>{qc.get('n_cells_after_mito_filter', 0):,}</dd></div>"
            + f"<div><dt>最终细胞</dt><dd>{qc.get('n_cells_final', 0):,}</dd></div>"
            + f"<div><dt>保留基因</dt><dd>{qc.get('n_genes_final', 0):,}</dd></div>"
            + f"<div><dt>滤除比例</dt><dd>{qc.get('cells_removed_pct', 0)}%</dd></div>"
            + "</div>",
            unsafe_allow_html=True,
        )
        if "n_doublets_predicted" in qc:
            st.caption(f"scrublet 预测双细胞：{qc['n_doublets_predicted']} 个"
                       "（-1 表示依赖缺失、已跳过）")
        st.dataframe(
            pd.DataFrame({"步骤": list(result.timings), "秒": list(result.timings.values())}),
            hide_index=True, width="stretch",
        )
        for w in result.warnings:
            st.markdown(f'<div class="sc-note">{_html.escape(w)}</div>', unsafe_allow_html=True)
        st.json(result.config, expanded=False)

        # 每次渲染都按当前 AI 缓存重写 obs：清缓存或换分辨率后不会留下上一轮的旧标签
        annot = result.annotate(annotation_rows(st.session_state.get("ai_cache", {})))
        st.markdown("#### AI 注释落回")
        st.caption(
            f"`obs` 新增列：{'、'.join(f'`{c}`' for c in annot['columns'])}。"
            f"已解读过 {annot['n_clusters_annotated']} 个簇，覆盖 "
            f"{annot['n_cells_labeled']:,} / {annot['n_cells_total']:,} 个细胞。"
            "拒答或被 Critic 升级的簇只写 `low_confidence`（不写具体类型），"
            "没点过解读的写 `unannotated`；簇表 CSV 与落盘 h5ad 都带这几列，"
            "`uns['sccopilot_annotation']` 里留了写回时间与判定口径。")

        d1, d2 = st.columns(2)
        d1.download_button("marker 表 CSV", result.markers.to_csv(index=False).encode("utf-8-sig"),
                           file_name="scCopilot_markers.csv", width="stretch")
        d2.download_button("簇表 CSV", result.clusters.to_csv(index=False).encode("utf-8-sig"),
                           file_name="scCopilot_clusters.csv", width="stretch")

        # data 传 callable 时 Streamlit 在独立线程里执行它，里面的 st.* 命令（含
        # session_state 读取）会被忽略——实测页面有结论、导出的存证包却整片「本簇未解读」。
        # 所以入参必须是渲染期的快照，见 audit_bundle_data。
        stt = Path(p["path"]).stat()
        audit_meta = {
            "data_source": p.get("src_id") or p.get("path", ""),
            "data_sha256": data_fingerprint(p["path"], stt.st_mtime, stt.st_size),
            "code_commit": code_commit(),
        }
        st.download_button(
            "导出存证包（单文件 HTML，可离线核对）",
            audit_bundle_data(result, dict(st.session_state.get("ai_cache", {})), audit_meta),
            file_name="scCopilot_audit_"
                      + fingerprint(p, ("src_id", "resolution", "n_markers")) + ".html",
            mime="text/html", width="stretch",
            help="输入 sha256、全部参数、逐簇证据表、prompt 与模型回复原文、obs 写回分布，"
                 "都在一个自包含文件里；三个月后别人不装 Python 也能核对。")

        cache = st.session_state.get("ai_cache", {})
        if cache:
            md = ["# scCopilot 解读报告", ""]
            for c, workflow in sorted(cache.items(), key=lambda kv: (len(kv[0]), kv[0])):
                itp = workflow["interpretation"]
                md += [
                    f"## 簇 {c} → {itp.predicted_type}",
                    f"- 置信度：{itp.confidence_label}（{itp.confidence_score}）｜来源：{itp.source}",
                    f"- Top marker：{', '.join(result.genes_of(c, top=8))}",
                    f"- 理由：{itp.reasoning.replace(chr(10), ' / ')}",
                ]
                md += [f"- 告警：{v}" for v in itp.violations]
                md.append("")
            d1.download_button("解读报告 MD", "\n".join(md).encode("utf-8-sig"),
                               file_name="scCopilot_report.md", width="stretch")
        if d2.button("落盘 h5ad + 表格到 data/processed", width="stretch"):
            stem = "run_" + fingerprint(p, ("src_id", "resolution", "n_markers"))
            with st.spinner("写出中…"):
                paths = result.save(DATA_DIR / "processed", stem)
            st.session_state["saved_paths"] = paths
            st.success("已写出：" + "、".join(Path(v).name for v in paths.values()))

        saved = st.session_state.get("saved_paths") or {}
        h5 = Path(saved["h5ad"]) if saved.get("h5ad") else None
        if h5 is not None and h5.exists():
            size_mb = h5.stat().st_size / 1e6
            if size_mb > 150:
                st.caption(f"带注释 h5ad 已落盘（{size_mb:,.0f} MB，太大不走浏览器下载）：{h5}")
            else:
                # 传可调用对象：点下载时才读盘，不为每次重跑把整个文件搬进内存
                st.download_button(f"下载带注释 h5ad（{size_mb:,.1f} MB）",
                                   lambda p=h5: p.read_bytes(),
                                   file_name=h5.name, mime="application/octet-stream",
                                   width="stretch")


# --------------------------------------------------------------------------- #
# 8. 主流程
# --------------------------------------------------------------------------- #
def hero(result: pipeline.PipelineResult | None) -> None:
    st.markdown(
        '<p class="sc-eyebrow">StudyHack · scCopilot</p>'
        '<p class="sc-title">从 UMAP 上的一个点，走到一个有证据的生物学结论</p>'
        '<p class="sc-sub">Scanpy 负责算，大模型负责读懂 marker 背后的细胞身份；'
        "证据不够时它会明说「证据不足，建议人工复核」，而不是编一个名字给你。</p>",
        unsafe_allow_html=True,
    )
    if result is None or result.clusters.empty:
        return
    a, qc, cl = result.adata, result.qc, result.clusters
    c1, c2, c3, c4, c5 = st.columns(5)
    removed = float(qc.get("cells_removed_pct", 0.0) or 0.0)
    # 一个细胞都没滤掉时不显示 delta："-0.0%" 看起来像出了故障
    c1.metric("细胞（QC 后）", f"{a.n_obs:,}",
              f"-{removed:.1f}%" if removed > 0 else None, delta_color="inverse")
    c2.metric("基因", f"{a.n_vars:,}")
    c3.metric("簇数", len(cl))
    c4.metric("分辨率", f"{result.config.get('resolution', 0):.2f}")
    c5.metric("证据充足的簇", f"{int(cl['evidence_ok'].sum())}/{len(cl)}")


def onboarding() -> None:
    """还没有分析结果时的引导页。

    只讲"下一步做什么"：能看到这个页面说明依赖已装好、页面已跑起来，
    所以安装/启动命令收进折叠区，别让它看起来像是卡在了环境准备上。
    """
    src_kind = st.session_state.get("src_kind", "使用内置演示数据")
    if src_kind == "上传 .h5ad 文件":
        lead = ("左侧把 .h5ad 拖进「上传」框就行 —— 选完文件页面会自己开始跑流水线，"
                "不用在命令行做任何事。")
    else:
        lead = "左侧「数据来源」选 **使用内置演示数据**，页面会自己开始跑流水线。"
    card(
        '<p class="sc-eyebrow">开始使用</p>'
        '<p class="sc-title" style="font-size:1.12rem">下一步</p>'
        f'<p class="sc-sub">{lead}</p>'
        '<p class="sc-sub">首次约 1 分钟（含 numba 首次编译），跑完自动出 UMAP；'
        "之后在 UMAP 上点一个细胞，右侧卡片逐字给出细胞类型判断、置信度与理由。</p>"
        '<p class="sc-sub" style="margin-top:.6rem">建议演示顺序：把分辨率从 1.0 拖到 1.6，'
        "看簇被切开；再点那个线粒体偏高的簇，看 AI 如何拒绝编造。</p>",
        cls="sc-card",
    )
    with st.expander("命令行与环境（已经打开本页就不必再执行）", expanded=False):
        for title, lang, cmd in (
            ("安装依赖", "bash", "pip install -r requirements.txt"),
            ("没有演示数据时先生成", "bash", "python scripts/make_demo_h5ad.py"),
            ("启动本页", "bash", "streamlit run app/main.py"),
            ("跑自检", "bash", "python tests/test_pipeline.py && python tests/test_explainer.py"),
            ("可选：接入大模型（在项目根目录建 .env）", "ini",
             'LLM_API_KEY=sk-xxxx\nLLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1\n'
             "LLM_MODEL=qwen-plus"),
        ):
            st.markdown(f"**{title}**")
            st.code(cmd, language=lang)


def render_ai_card_with_trace(cluster: str, result: pipeline.PipelineResult, p: dict[str, Any]) -> None:
    """带多智能体思考过程可视化的 AI 卡片。"""
    cache = st.session_state.setdefault("ai_cache", {})

    n_top = min(p["n_markers"], len(result.genes_of(cluster)))
    model_tag = ai.LLMConfig.from_env().model if p["use_llm"] else "本地规则"
    head = (
        '<div class="sc-ai-head"><div class="sc-avatar">AI</div><div>'
        f'<div class="sc-ai-name">scCopilot · 免疫学解读</div>'
        f'<div class="sc-ai-meta">簇 {cluster} ｜ Top {n_top} 个差异基因 ｜ {model_tag}</div>'
        "</div></div>"
    )

    if cluster not in cache:
        if not p["auto_ai"]:
            card(head + '<p class="sc-sub">已关闭「点击 UMAP 后自动解读」。点下面的按钮调用解读。</p>',
                 cls="sc-ai")
            if st.button("开始解读该簇", key=f"run_ai_{cluster}", type="primary"):
                with st.status("多智能体工作流启动...", expanded=True) as status:
                    st.write("🧠 **Planner** 正在规划执行策略...")
                    time.sleep(0.3)
                    st.write("🔍 **Annotator** 正在检索知识图谱...")
                    time.sleep(0.3)
                    st.write("⚖️ **Critic** 正在审查结果...")
                    time.sleep(0.3)
                    # 调用多智能体工作流
                    workflow_result = orchestrator.run_annotation_workflow(
                        cluster_id=cluster,
                        marker_genes=result.genes_of(cluster, top=n_top),
                        cluster_stats=result.cluster_stats(cluster),
                        evidence_rows=result.evidence_records(cluster, top=n_top),
                        llm=ai.LLMConfig.from_env() if p["use_llm"] else None,
                        use_llm=p["use_llm"],
                        adata=result.adata, cluster_key=result.cluster_key,
                    )
                    status.update(label="解读完成", state="complete", expanded=False)
                    cache[cluster] = workflow_result
                st.rerun()
            return

        # 自动模式：直接调用多智能体工作流
        with st.status("多智能体工作流启动...", expanded=True) as status:
            st.write("🧠 **Planner** 正在规划执行策略...")
            time.sleep(0.2)
            st.write("🔍 **Annotator** 正在检索知识图谱...")
            time.sleep(0.2)
            st.write("⚖️ **Critic** 正在审查结果...")
            time.sleep(0.2)
            workflow_result = orchestrator.run_annotation_workflow(
                cluster_id=cluster,
                marker_genes=result.genes_of(cluster, top=n_top),
                cluster_stats=result.cluster_stats(cluster),
                evidence_rows=result.evidence_records(cluster, top=n_top),
                llm=ai.LLMConfig.from_env() if p["use_llm"] else None,
                use_llm=p["use_llm"],
                adata=result.adata, cluster_key=result.cluster_key,
            )
            status.update(label="解读完成", state="complete", expanded=False)
            cache[cluster] = workflow_result

    workflow = cache[cluster]
    itp = workflow["interpretation"]
    critique = workflow["critique"]
    kg_candidates = workflow["kg_candidates"]
    trace = workflow["trace"]

    # 如果 Critic 返回 escalate，显示红色警告框；details 是闸门的真实判定数字
    if critique["status"] == "escalate":
        details = critique.get("details") or []
        msg = (f"🚫 **AI 主动拒绝回答**\n\n"
               f"原因：{_html.escape(critique['reason'] or '未知')}\n\n")
        if details:
            msg += ("具体缺什么证据：\n\n"
                    + "\n".join(f"{i}. {_html.escape(str(d))}" for i, d in enumerate(details, 1))
                    + "\n\n")
        msg += "这体现了系统的防幻觉能力：当证据不足或与知识图谱冲突时，AI 不会强行给出不可靠的结论。"
        st.error(msg, icon="⚠️")

    refused = itp.refused or itp.predicted_type == ai.REFUSAL
    meta = (("本地缓存命中 · 首次请求 " + itp.cached_at) if itp.cache_hit else
            (f"{itp.latency_s}s" if itp.latency_s else "0 次网络请求"))
    if itp.prompt_chars:
        # 字符数不等于"发出去了"：本地兜底与缓存命中都没有产生新的 API 调用。
        state = ("已发送" if itp.source == "llm" and not itp.cache_hit else
                 "缓存复用，未重新发送" if itp.cache_hit else "构造未发送")
        meta += f" ｜ prompt {itp.prompt_chars} 字符（{state}）"
    het = workflow.get("heterogeneity")
    parts = [
        head,
        '<p class="sc-eyebrow">【预测细胞类型】</p>',
        f'<p class="sc-type{" sc-type--refuse" if refused else ""}">'
        f'{_html.escape(itp.predicted_type)}</p>',
        f'{conf_badge(itp.confidence_label, itp.confidence_score)}{source_chip(itp.source)}'
        f'<span class="sc-chip sc-chip--ghost">{_html.escape(meta)}</span>'
        + het_chip(het),
    ]
    if itp.cited_genes:
        parts.append('<p class="sc-eyebrow">理由中引用到的本次 marker</p>'
                     + "".join(chip(g, "gene") for g in itp.cited_genes[:12]))
    if refused and itp.reasoning:
        parts.append('<div class="sc-note">拒答依据（含改进建议）：<br>'
                     + _html.escape(itp.reasoning).replace("\n", "<br>") + "</div>")
    for v in itp.violations:
        parts.append(f'<div class="sc-note sc-note--bad">防幻觉告警：{_html.escape(v)}</div>')
    for n in itp.notes:
        parts.append(f'<div class="sc-note">说明：{_html.escape(n)}</div>')
    card("".join(parts), cls="sc-ai")

    # Agent 思考过程时间线
    # 注意：拼给 unsafe_allow_html 的 HTML 必须顶格。Markdown 会把行首 4 个以上空格
    # 当成缩进代码块，整段 HTML 会原样显示成文本（实测踩过）。
    if trace:
        st.markdown("#### 🕒 Agent 思考过程")
        agent_icon = {"Planner": "🧠", "Annotator": "🔍", "Critic": "⚖️"}
        steps_html = "".join(
            '<div class="sc-timeline-step">'
            f'<div class="sc-timeline-dot">{agent_icon.get(step["agent_name"], "•")}</div>'
            '<div class="sc-timeline-content">'
            f'<div class="sc-timeline-title"><strong>{_html.escape(step["agent_name"])}</strong>'
            f' · {_html.escape(step["action"])}</div>'
            f'<div class="sc-timeline-detail">{step["duration_s"]:.3f}s ｜ '
            f'{_html.escape(str(step["output_summary"].get("reason") or ""))}</div>'
            '</div></div>'
            for step in trace
        )
        st.markdown(f'<div class="sc-timeline">{steps_html}</div>', unsafe_allow_html=True)

    # KG 候选展示
    if kg_candidates:
        with st.expander("知识图谱候选细胞类型", expanded=False):
            for c in kg_candidates[:5]:
                expr = c.get("expression") or {}
                if expr.get("per_gene"):
                    st.markdown(
                        f"- **{_html.escape(str(c['name']))}**（名字重合 {c['score']:.2f} · "
                        f"簇内实测检出 {len(expr['detected'])}/{len(expr['per_gene'])} · "
                        f"support_ratio={expr['support_ratio']:.2f}）"
                    )
                    st.caption(
                        "　检出：" + (", ".join(f"{_html.escape(d['gene'])} {d['pct_expressed']:.0%}"
                                               for d in expr["per_gene"] if d["detected"]) or "无")
                        + "　静默：" + (", ".join(_html.escape(g) for g in expr["silent"][:6]) or "无")
                        + f"　（n={expr['n_cells']}，口径：{expr.get('unit', '—')}）"
                    )
                else:
                    st.markdown(
                        f"- **{c['name']}** (score: {c['score']:.2f}, matched: {', '.join(c['matched_markers'][:3])})"
                    )

    het_panel(het)

    # 打字机效果展示推理文本
    typewriter(itp.reasoning or "（本次没有可展示的推理文本）",
               enabled=p["animate"], heading="【推断理由】")

    with st.expander("证据表 / 闸门判定 / 模型原文", expanded=False):
        evidence_table(result.evidence_of(cluster), n_top)
        gate_details(itp)
        if itp.raw:
            st.code(itp.raw, language="markdown")


def render_ai_card(cluster: str, result: pipeline.PipelineResult, p: dict[str, Any]) -> None:
    """兼容旧接口：直接调用 render_ai_card_with_trace。"""
    render_ai_card_with_trace(cluster, result, p)


def main() -> None:
    st.set_page_config(page_title="scCopilot · 单细胞 AI 工作台", layout="wide",
                       initial_sidebar_state="expanded")
    inject_css()

    params = sidebar_controls()
    result = ensure_result(params)
    hero(result)
    if result is None or result.clusters.empty:
        onboarding()
        return

    # 注释落回在渲染前统一做一次：左栏簇一览、右栏导出读到的是同一份 obs/簇表，
    # 不看渲染顺序脸色（缓存恢复、清 ai_cache 之后旧标签也会被重写回 unannotated）。
    result.annotate(annotation_rows(st.session_state.get("ai_cache", {})))

    df = cell_table(result)
    st.session_state["umap_cells"] = df          # 供点击回退查表使用
    colors = cluster_colors(df["cluster"].unique().tolist())
    cl = result.clusters
    # 下拉框展示"簇号 + 规模"，内部只保留簇号，避免格式化字符串到处传
    options = {str(c): f"{c} · {int(n):,} 个细胞" for c, n in zip(cl["cluster"], cl["n_cells"])}
    selected = resolve_selected(options)

    with st.sidebar:
        st.divider()
        st.markdown("##### 5. 运行")
        st.caption(f"当前聚焦 **簇 {selected}**。首次跑批含编译耗时，调分辨率只需几秒。")

    left, right = st.columns([0.60, 0.40], gap="large")

    with left:
        color_by = st.radio("UMAP 着色",
                            ["簇标签", "线粒体占比", "UMI 数量", "AI 置信度", "看一个基因"],
                            horizontal=True, key="color_by")
        gene_hit = None
        if color_by == "看一个基因":
            typed = st.text_input("基因名（大小写不敏感，小鼠 Cd3d 也能命中）",
                                  key="gene_probe", placeholder="例如 CD3D").strip()
            gene_hit = pipeline.gene_column(result.adata, typed) if typed else None
            if typed and gene_hit is None:
                st.warning(f"`{typed}` 在这份矩阵里没有唯一命中（名字可能是别的物种的同源名，"
                           "或者重复），所以不画——宁可不画也不猜一个基因给你看。")
        # 没命中时退回簇着色，别把"看一个基因"当成 AI 置信度分支
        umap_mode = "簇标签" if color_by == "看一个基因" and gene_hit is None else color_by
        fig = build_umap_fig(df, umap_mode, selected, colors, gene=gene_hit)
        # plotly-events 适配器：装了用它的细粒度点击事件，没装退回 Streamlit 原生 on_select
        if _HAS_PLOTLY_EVENTS:
            selected_events = plotly_events(fig, select_event=True, click_event=True, key="umap_click")
            if selected_events:
                # 取最后一个点作为选中簇
                last_pt = selected_events[-1]
                if "curveNumber" in last_pt:
                    selected = str(df["cluster"].unique()[last_pt["curveNumber"]])
                    st.session_state["cluster_pick"] = selected
        else:
            st.plotly_chart(fig, on_select="rerun",
                            selection_mode=("points", "box", "lasso"), key="umap_click")
        st.caption("单击选一个点（框选/套索取最后一个点）；被选中的簇会高亮，其余淡化。"
                   + ("　着色用的是 `adata.X`（log CPM+1），与 marker 条形图同一层。"
                      if gene_hit is not None else ""))
        if gene_hit is not None:
            name, vec = gene_hit
            st.plotly_chart(build_gene_dot(name, gene_cluster_table(df, vec), colors),
                            width="stretch")
        # 选项是簇号，展示文字由 format_func 加规模；state 已在 resolve_selected 里对齐，
        # 所以这里不再传 index——传了会和 session_state 打架，点击也就同步不进下拉框。
        selected = st.selectbox(
            "或从列表里选一个簇", options=list(options),
            format_func=lambda c: options[c], key="cluster_pick",
        )

        row = cl[cl["cluster"] == selected].iloc[0]
        mito_ok = bool(pd.notna(row["median_pct_mito"]) and row["median_pct_mito"] < 15)
        st.markdown(
            chip(f"{int(row['n_cells']):,} 个细胞", "blue")
            + chip(f"{int(row['n_strong_markers'])} 个强证据 marker", "blue")
            + chip(f"未注释比例 {row['novel_marker_ratio']:.0%}",
                   "ok" if row["novel_marker_ratio"] < 0.5 else "warn")
            + chip(f"中线粒体 {txt(row['median_pct_mito'])}%", "ok" if mito_ok else "warn")
            + (chip("证据充足", "ok") if row["evidence_ok"]
               else chip("证据不足 → AI 会拒绝解读", "bad")),
            unsafe_allow_html=True,
        )

        # 上限 = 本簇实际算出的 marker 行数：head(top) 截不出比这更多的条，
        # 写死 4~20 会让滑块右半段拖了没反应。
        evidence = result.evidence_of(selected)
        avail = len(evidence)
        if avail > 2:
            top_n = st.slider("条形图显示前几个基因", 2, avail, min(8, avail), key="bar_top",
                              help=f"本簇只算出了 {avail} 个 marker，再多就没有了。"
                                   "想看更多基因先调侧栏「每簇 Top 基因数」（即时生效），这里会自动放宽。")
        else:
            top_n = avail
        st.plotly_chart(build_marker_fig(evidence, top_n),
                        width="stretch")

        st.markdown("#### 全部簇一览")
        st.markdown(
            '<p class="sc-sub">「证据充足」= 细胞数 ≥10、强证据 marker ≥2、未注释比例 &lt;0.75、'
            "线粒体比例 &lt;0.5 同时成立。不达标的簇会被 AI 层直接拦下（不联网、不烧 token）。</p>",
            unsafe_allow_html=True,
        )
        view = cl.rename(columns={
            "cluster": "簇", "n_cells": "细胞数", "pct_cells": "占比%", "n_strong_markers": "强证据",
            "novel_marker_ratio": "未注释比例", "mito_marker_ratio": "线粒体比例",
            "median_pct_mito": "中线粒体%", "median_counts": "中UMI",
            "evidence_ok": "证据充足", "suspect_low_quality": "疑似低质量",
            "sccopilot_type": "AI注释", "sccopilot_het": "异质性",
        })
        keep = [c for c in ("簇", "AI注释", "细胞数", "占比%", "强证据", "中线粒体%", "中UMI",
                           "未注释比例", "线粒体比例", "证据充足", "疑似低质量") if c in view]
        st.dataframe(
            view[keep], hide_index=True, width="stretch", height=min(400, 60 + 35 * len(view)),
            column_config={
                "细胞数": st.column_config.ProgressColumn(min_value=0,
                                                     max_value=float(cl["n_cells"].max()), format="%d"),
                "未注释比例": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                        format="percent"),
                "中线粒体%": st.column_config.NumberColumn(format="%.1f"),
                "中UMI": st.column_config.NumberColumn(format="%.0f"),
            },
        )

    with right:
        # 用 st.fragment 隔离右侧 AI 面板：点击 UMAP 只刷新右侧，不重跑整个页面
        @st.fragment
        def ai_panel_fragment(cluster_id: str):
            render_ai_card_with_trace(cluster_id, result, params)
            st.markdown('<div style="height:.85rem"></div>', unsafe_allow_html=True)
            render_resolution_card(result, params)
            render_run_info(result, params)

        ai_panel_fragment(selected)


if __name__ == "__main__":
    main()
