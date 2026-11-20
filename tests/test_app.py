# -*- coding: utf-8 -*-
"""app/main.py 的回归测试：纯函数 + Streamlit 官方 AppTest 无头整页运行。

跑法：
    python tests/test_app.py          # 内置迷你 runner，无需 pytest
    python -m pytest tests/test_app.py -q

为什么值得跑 AppTest：它真的把「参数 → core 流水线 → Plotly 图 → AI 卡片」整条链路
走一遍，UI 里任何一处 import 名写错、scanpy/streamlit API 用错都会当场暴露。
代价是它会把流水线完整跑一次（本机约 1 分钟，含 numba 首次编译）。

不依赖浏览器：AppTest 直接驱动 ScriptRunner，所以 CI/评审机都能跑。
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# 开发机上装了 .env 也不能让无头测试真去敲大模型网关：走项目自带的总开关，
# 断言的是"没网也必须给出结论并标注来源"这条本地兜底路径。
os.environ["SCCOPILOT_DISABLE_AI"] = "1"

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402

app = __import__("app.main", fromlist=["main"])   # noqa: E402  命名空间包，无需 __init__.py


# --------------------------------------------------------------------------- #
# 辅助：构造与 Streamlit PlotlyState 同形的假选中状态
# --------------------------------------------------------------------------- #
def fake_state(points: list) -> SimpleNamespace:
    """外层支持属性访问（PlotlyState 就是 dict-like + 属性双访问），内层是普通 dict。"""
    return SimpleNamespace(selection=SimpleNamespace(points=points, box=[], lasso=[]))


# --------------------------------------------------------------------------- #
# 1. 小工具：数字与文本渲染不能出现 nan，HTML 必须转义
# --------------------------------------------------------------------------- #
def test_rgba_helper_produces_css_color():
    assert app.rgba("#2563eb", 0.5) == "rgba(37, 99, 235, 0.50)", app.rgba("#2563eb", 0.5)


def test_txt_helper_hides_nan():
    assert app.txt(float("nan")) == "—", "缺失值不能显示成 nan"
    assert app.txt(None) == "—"
    assert app.txt(12.34, "{:.1f}") == "12.3"
    assert app.txt("abc") == "—"
    assert app.txt(1234.0, "{:,.0f}") == "1,234"


def test_chip_escapes_untrusted_text():
    # 基因名来自用户上传的文件，属于不可信输入：不能原样拼进 DOM
    out = app.chip('<img src=x onerror="alert(1)">', "gene")
    assert "<img" not in out and "&lt;img" in out, out
    assert "sc-chip--gene" in out


def test_style_css_can_be_wrapped_in_one_style_tag():
    """style.css 会被整份塞进一个 style 标签注入页面。

    文件里只要再出现成对的标签字面量（哪怕是注释里的示例代码），外层标签就会被提前闭合，
    剩下的 CSS 会当正文显示出来——实测踩过，所以钉住这条约束。
    """
    css = (ROOT / "app" / "style.css").read_text(encoding="utf-8")
    assert "</style" not in css and "<style" not in css, "CSS 里不能出现 style 标签字面量"
    assert css.count("{") == css.count("}"), "花括号不配对，注入后规则会整段失效"
    assert "--sc-primary" in css, "设计令牌缺失说明样式没被真正读进来"


# --------------------------------------------------------------------------- #
# 2. 点击回填：这是"点 UMAP 就解读那个簇"的地基
# --------------------------------------------------------------------------- #
def test_clicked_cluster_handles_both_customdata_shapes():
    both = [
        fake_state([{"curve_number": 0, "point_index": 2, "customdata": ["7", 120]}]),
        fake_state([{"curve_number": 0, "point_index": 2, "customdata": "7"}]),
    ]
    for st_obj in both:
        assert app.clicked_cluster(st_obj) == "7", st_obj
    assert app.clicked_cluster(None) is None
    assert app.clicked_cluster(fake_state([])) is None
    assert app.clicked_cluster({"unexpected": "object"}) is None, "形状不认识时应安静返回 None"


def test_selection_signature_only_changes_on_a_new_click():
    a = fake_state([{"curve_number": 0, "point_index": 12}])
    b = fake_state([{"curve_number": 0, "point_index": 12}])
    c = fake_state([{"curve_number": 0, "point_index": 99}])
    assert app.selection_signature(a) == app.selection_signature(b)
    assert app.selection_signature(a) != app.selection_signature(c)
    assert app.selection_signature(None) == ""
    # 第一个点的 point_index 合法值就是 0，不能因为"假值"退回 point_number / None
    zero = fake_state([{"curve_number": 0, "point_index": 0, "point_number": 4711}])
    assert app.selection_signature(zero) == "0:0", app.selection_signature(zero)
    # 签名相同 => main 里不会用旧点击覆盖用户在下拉框里的新选择
    sig = app.selection_signature(a)
    assert sig and isinstance(sig, str)


def test_umap_figure_keeps_row_index_aligned_with_cluster():
    """单 trace 的代价是"点索引 = 行号"这条不变量，一旦破坏点击就会选错簇。"""
    df = pd.DataFrame({
        "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        "cluster": ["0", "0", "0", "1", "1", "2"],
        "pct_mito": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "n_counts": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
    })
    colors = app.cluster_colors(df["cluster"].unique().tolist())
    fig = app.build_umap_fig(df, "簇标签", "1", colors)
    assert len(fig.data) == 2, "第一条是细胞散点，第二条是簇质心"
    cells = fig.data[0]
    assert [cd[0] for cd in cells.customdata] == df["cluster"].tolist(), "行号与簇必须一一对应"
    assert cells.mode == "markers"
    # 选中簇不透明、其余淡出：肉眼要能立刻看出"我现在看的是哪一群"
    alpha = lambda s: float(s.rsplit(",", 1)[1].strip().rstrip(")"))  # noqa: E731
    alphas = [alpha(c) for c in cells.marker.color]
    assert alphas == [0.14, 0.14, 0.14, 0.95, 0.95, 0.14], alphas
    assert app.build_umap_fig(df, "线粒体占比", "1", colors).data[0].marker.colorbar is not None


def test_umap_continuous_colors_use_the_matching_column():
    """「UMI 数量」一度读的是 pct_mito 列：图照样画得出来、也不报错，
    但颜色表达的是另一件事——这类取错列的 bug 只能靠断言钉住。"""
    df = pd.DataFrame({
        "x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 1.0], "cluster": ["0", "0", "1"],
        "pct_mito": [99.0, 98.0, 97.0], "n_counts": [1000.0, 2000.0, 3000.0],
    })
    colors = app.cluster_colors(["0", "1"])
    umi = app.build_umap_fig(df, "UMI 数量", "0", colors).data[0].marker.color
    assert [round(v, 4) for v in umi] == [3.0, 3.301, 3.4771], umi
    mito = app.build_umap_fig(df, "线粒体占比", "0", colors).data[0].marker.color
    assert [float(v) for v in mito] == [99.0, 98.0, 97.0], mito


def test_marker_figure_color_codes_gene_classes():
    sub = pd.DataFrame({
        "rank": [1, 2, 3, 4],
        "gene": ["CD3D", "MT-CO1", "RPL13A", "LOC780542"],
        "mean_expr": [2.1, 3.0, 1.2, 0.4],
        "pct_expressed": [0.8, 0.95, 0.9, 0.1],
        "scores": [12.0, 9.0, 5.0, 2.0],
        "logfoldchanges": [1.5, 0.8, 0.3, 0.1],
        "is_mito": [False, True, False, False],
        "is_ribo": [False, False, True, False],
        "is_novel": [False, False, False, True],
    })
    fig = app.build_marker_fig(sub, 4)
    colors = fig.data[0].marker.color
    assert "#2563eb" in colors and "#f59e0b" in colors and "#e11d48" in colors, colors
    assert len(fig.data[0].y) == 4


def test_marker_hover_shows_log2fc_and_adjusted_p_without_inventing_them():
    """条数少的右列放不下所有统计量，但 hover 必须给全；缺列时宁可不给。"""
    sub = pd.DataFrame({
        "rank": [1, 2, 3, 4],
        "gene": ["CD3D", "MT-CO1", "RPL13A", "LOC780542"],
        "mean_expr": [2.1, 3.0, 1.2, 0.4],
        "pct_expressed": [0.8, 0.95, 0.9, 0.1],
        "scores": [12.0, 9.0, 5.0, 2.0],
        "logfoldchanges": [1.5, 0.8, 0.3, 0.1],
        "pvals_adj": [0.0, 1.2e-8, 0.44, 0.9],
        "is_mito": [False, True, False, False],
        "is_ribo": [False, False, True, False],
        "is_novel": [False, False, False, True],
    })
    stats = [row[3] for row in app.build_marker_fig(sub, 4).data[0].customdata]
    assert all("log2FC" in s and "校正 p" in s for s in stats), stats
    assert any("e-08" in s for s in stats), stats        # 小 p 用科学计数法，别显示成 0.000
    lean = sub.drop(columns=["logfoldchanges", "pvals_adj"])
    assert [r[3] for r in app.build_marker_fig(lean, 4).data[0].customdata] == [""] * 4


def test_gene_probe_colors_umap_and_summarizes_per_cluster():
    """「看一个基因」：feature plot 用同一行序的向量着色，点图给逐簇均值与检出率。"""
    df = pd.DataFrame({
        "x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 2.0, 3.0],
        "cluster": ["0", "0", "1", "1"],
        "pct_mito": [1.0, 2.0, 3.0, 4.0], "n_counts": [100.0, 200.0, 300.0, 400.0],
    })
    colors = {"0": "#1f77b4", "1": "#ff7f0e"}
    vec = np.array([0.0, 1.5, 4.0, 4.5], dtype=np.float32)

    fig = app.build_umap_fig(df, "看一个基因", "0", colors, gene=("CD3D", vec))
    assert [float(v) for v in fig.data[0].marker.color] == [0.0, 1.5, 4.0, 4.5]
    assert "log CPM+1" in str(fig.data[0].marker.colorbar.title)
    assert "CD3D %{customdata[2]:.3f}" in fig.data[0].hovertemplate
    # 没命中时按簇着色，绝不能把"看一个基因"当成 AI 置信度分支
    plain = app.build_umap_fig(df, "簇标签", "0", colors)
    assert "CD3D" not in plain.data[0].hovertemplate

    tab = app.gene_cluster_table(df, vec)
    assert tab["cluster"].tolist() == ["0", "1"]
    assert tab["pct_expressed"].tolist() == [0.5, 1.0]
    assert tab["mean_expr"].tolist() == [0.75, 4.25]

    dot = app.build_gene_dot("CD3D", tab, colors)
    assert dot.data[0].y.tolist() == ["1", "0"], "点图应把表达最高的簇放在上面"
    assert float(dot.data[0].x[0]) > float(dot.data[0].x[-1])


def test_typewriter_html_escapes_text_as_js_not_html():
    """两条硬约束：模板不能用 str.format（CSS 的 "{ margin: 0 }" 会被当占位符，真崩过）；
    正文必须作为 JS 字符串字面量进模板，不能拼成 HTML——模型输出和基因名都不可信。"""
    out = app.typewriter_html('CD3D"; alert(1); <b>x</b>', interval_ms=24)
    assert "@@TEXT@@" not in out and "@@INTERVAL@@" not in out
    assert "setInterval(tick, 24)" in out
    assert "{ margin: 0; padding: 0; background: transparent; }" in out, "样式块必须原样保留"
    assert "<b>" not in out, "正文不能拼成 HTML"
    assert 'var src = "CD3D\\"; alert(1); \\u003cb\\u003ex\\u003c/b\\u003e";' in out, out
    assert "T 细胞" in app.typewriter_html("T 细胞（CD3D+）")   # 中文不该被转成 \\uXXXX

    # 最关键的注入用例：正文里带 </script> 也不能提前闭合脚本标签
    evil = app.typewriter_html("</script><script>alert(1)</script>")
    assert evil.count("<script>") == 1, evil.count("<script>")
    assert evil.count("</script>") == 1, evil.count("</script>")

    # 小标题走同一通道：它来自代码，但同样不该被拼成可执行 HTML
    head = app.typewriter_html("正文", heading='【推断理由】<img src=x onerror=alert(1)>')
    assert "@@HEADING@@" not in head
    assert "<img" not in head, "标题里的标签必须被转义"
    assert '\\u003cimg src=x onerror=alert(1)\\u003e";' in head, head
    assert "【推断理由】" in head, "中文标题不该被转成 \\uXXXX"
    assert head.count("<div") == 2, "标题和正文各一个 div，不该多出可注入的位置"


# --------------------------------------------------------------------------- #
# 3. 参数编排：哪些改动要重跑全流水线，哪些只要重聚类
# --------------------------------------------------------------------------- #
def test_fingerprints_separate_full_run_from_recluster():
    base = {
        "src_id": "demo:x", "min_genes": 200, "min_counts": 500, "max_pct_mito": 20.0,
        "max_genes": 6000, "min_cells": 3, "filter_doublets": False, "n_top_genes": 2000,
        "hvg_flavor": "seurat", "n_pcs": 50, "n_neighbors": 15, "marker_method": "wilcoxon",
        "random_state": 0, "resolution": 1.0, "n_markers": 8,
    }
    keys = app.FULL_KEYS
    fp = lambda p: (app.fingerprint(p, keys), app.fingerprint(p, ("resolution", "n_markers", "src_id")))
    a, b = fp(base), fp({**base, "resolution": 1.6})
    assert a[0] == b[0] and a[1] != b[1], "只改分辨率：全量指纹不变，重聚类指纹变"
    c = fp({**base, "min_counts": 800})
    assert c[0] != a[0], "改 QC 阈值必须触发整条流水线重跑"
    d = fp({**base, "n_markers": 15})
    assert d[0] == a[0] and d[1] != a[1], "改 Top 基因数走快路径"


def test_params_to_config_maps_every_widget():
    p = {
        "min_genes": 200, "min_counts": 500, "max_pct_mito": 20.0, "max_genes": None,
        "min_cells": 3, "filter_doublets": True, "n_top_genes": 2000, "hvg_flavor": "seurat",
        "n_pcs": 50, "n_neighbors": 15, "resolution": 1.25, "n_markers": 10,
        "marker_method": "wilcoxon", "random_state": 0,
    }
    cfg = app.params_to_config(p)
    assert cfg.resolution == 1.25 and cfg.n_markers == 10 and cfg.max_genes is None
    assert cfg.filter_doublets is True
    cfg.validate()


def test_stage_upload_sanitizes_file_name(tmp_name: str = "evil/../../../windows.ini") -> None:
    """路径穿越的文件名必须被清洗，且相同内容复用同一个落盘文件。"""
    blob = b"not-a-real-h5ad-but-fine-for-this-test"
    p1, id1 = app.stage_upload(tmp_name, len(blob), blob)
    p2, id2 = app.stage_upload(tmp_name, len(blob), blob)
    try:
        assert p1 == p2 and id1 == id2, "同一份内容不该重复写盘"
        name = Path(p1).name
        assert ".." not in name and "/" not in name and "\\" not in name, name
        assert Path(p1).exists()
    finally:
        Path(p1).unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# 4. 整页无头运行（慢测试，但它是唯一能证明"UI 真的能跑"的东西）
# --------------------------------------------------------------------------- #
def test_app_renders_and_reclusters_headless():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "main.py"), default_timeout=900)
    at.run()
    assert not at.exception, [e.value for e in at.exception]

    # 演示数据默认选中 → 流水线应已跑完并留在 session_state 里
    res = at.session_state["result"]
    assert res is not None and len(res.clusters) >= 2, "没有跑完流水线"
    assert res.adata.n_obs > 100 and "X_umap" in res.adata.obsm
    assert len(at.session_state["umap_cells"]) == res.adata.n_obs, "点击回填依赖这张表的行号"

    md_all = " ".join(str(m.value) for m in at.markdown)
    assert "【预测细胞类型】" in md_all, "AI 卡片没渲染出来"
    assert at.get("plotly_chart"), "UMAP / 条形图至少要有两张图"
    assert at.get("iframe"), "打字机是 st.iframe 组件，元素类型是 iframe 而不是 html"
    # 【推断理由】的标题跟着正文进了 iframe：iframe 是独立 DOM，标题留在卡片里会和正文分家
    srcdoc = " ".join(str(el.proto.srcdoc) for el in at.get("iframe"))
    assert "【推断理由】" in srcdoc, "推理标题应和正文一起出现在打字机里"
    assert "【推断理由】" not in md_all, "卡片里不该再留一个没有正文相伴的标题"

    # 异质性判定详情跟着卡片渲染：参照细胞数与判型分中位数要能在页面上对外交代。
    # 这些口径数字走 st.caption——AppTest 把 caption 单独归类，不在 at.markdown 里。
    assert "簇内异质性判定" in md_all, "异质性判定详情没渲染出来"
    caps = " ".join(str(c.value) for c in at.get("caption"))
    assert "参照细胞" in caps, "判定没写明参照是谁"
    assert "判别基因" in caps, "判型分没说明打分口径"

    # 没有 LLM_API_KEY 时也必须给出结论，且明确标注是本地规则推断
    cache = at.session_state["ai_cache"]
    assert cache, "auto_ai 打开时应自动解读一次"
    # 缓存里是整个多智能体工作流（结论 + 批评 + KG 候选 + trace），结论只是其中一项
    itp = next(iter(cache.values()))["interpretation"]
    assert itp.predicted_type and itp.source in {"local_kb", "llm", "gate"}, itp.source
    if itp.source == "local_kb":
        assert "未调用大模型" in " ".join(itp.notes), itp.notes

    # AI 结论必须落回 obs/簇表/uns：只活在页面上的注释，用户带回自己的流程就用不上了
    obs = res.adata.obs
    for col in ("sccopilot_type", "sccopilot_conf", "sccopilot_source", "sccopilot_het"):
        assert col in obs.columns, col
    for cl, wf in cache.items():
        want = str(wf["interpretation"].predicted_type)
        got = set(obs.loc[obs[res.cluster_key].astype(str) == str(cl), "sccopilot_type"])
        assert got and got <= {want, "low_confidence"}, (cl, got, want)
    assert {"sccopilot_type", "sccopilot_conf"} <= set(res.clusters.columns)
    assert "sccopilot_annotation" in res.adata.uns
    assert "AI 注释落回" in md_all, "导出区没说清注释落到哪几列"

    n_before = len(res.clusters)
    n_cells_before = res.adata.n_obs

    # 拖 Resolution：应走快路径（邻居图复用），且簇数发生变化
    at.sidebar.slider(key="resolution").set_value(1.8)
    t0 = time.perf_counter()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    res2 = at.session_state["result"]
    assert len(res2.clusters) > n_before, \
        f"分辨率 1.0→1.8 后簇数应增多：{n_before} → {len(res2.clusters)}"
    assert res2.adata.n_obs == n_cells_before, "调分辨率不该重新过滤细胞"
    assert at.session_state["prev_labels"] is not None, "对比用的旧标签没留住"
    assert len(at.session_state["prev_labels"]) == res2.adata.n_obs
    assert abs(float(res2.config["resolution"]) - 1.8) < 1e-9, res2.config["resolution"]
    assert time.perf_counter() - t0 < 240, "调分辨率应走快路径，不该重跑邻居图"

    # 换簇：下拉框写 state → 重新渲染 → AI 缓存里应出现该簇的新结论
    labels = [str(c) for c in res2.clusters["cluster"]]
    target = labels[-1] if len(labels) > 1 else labels[0]
    at.selectbox(key="cluster_pick").set_value(target)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert target in at.session_state["ai_cache"], sorted(at.session_state["ai_cache"])

    # 关掉自动解读：切簇后不该新增调用，页面上给出按钮
    at.sidebar.checkbox(key="auto_ai").set_value(False)
    other = next((c for c in labels if c not in at.session_state["ai_cache"]), None)
    if other:
        at.selectbox(key="cluster_pick").set_value(other)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        assert other not in at.session_state["ai_cache"], "关掉自动解读后仍调用了模型"


def test_review_panel_records_decisions_headless():
    """人工复核整链路：表单 → append-only jsonl → obs 终稿列 → 导出区口径。

    钉住两条口径：终稿绝不覆盖 AI 首稿（否则改动率无法复算），第二次决定只追加一行、
    历史不消失。日志写进 SCCOPILOT_REVIEW_DIR 指向的隔离目录，不污染 data/reviews。
    """
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="sccopilot_reviews_"))
    os.environ["SCCOPILOT_REVIEW_DIR"] = str(tmp)
    try:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(ROOT / "app" / "main.py"), default_timeout=900)
        at.run()
        assert not at.exception, [e.value for e in at.exception]

        res = at.session_state["result"]
        cid = str(res.clusters["cluster"].iloc[0])
        ckey = res.cluster_key
        wf = at.session_state["ai_cache"][cid]
        itp = wf["interpretation"]
        blocked = bool(itp.refused) or (wf.get("critique") or {}).get("status") == "escalate"
        want_label = "low_confidence" if blocked else str(itp.predicted_type)
        draft = str(res.adata.obs.loc[res.adata.obs[ckey].astype(str) == cid,
                                      "sccopilot_type"].iloc[0])
        # 首稿落回必须已经跟上本轮的 AI 结论：曾经踩过 annotate 早于卡片执行，
        # 于是第一次复核把 ai_label 记成 unannotated。
        assert draft == want_label, (draft, want_label)
        assert draft != "unannotated", "annotate 没跟上本轮解读，复核记录会写错首稿"
        assert "人工复核" in " ".join(str(m.value) for m in at.markdown), "复核面板没渲染"

        at.radio(key=f"review_action_{cid}").set_value("edit")
        at.text_input(key=f"review_type_{cid}").set_value("NK 细胞")
        at.text_input(key=f"review_note_{cid}").set_value("CD3D 阴性，改判 NK")
        at.text_input(key=f"review_reviewer_{cid}").set_value("测试员")
        at.form_submit_button(key=f"review_save_{cid}").click().run()
        assert not at.exception, [e.value for e in at.exception]

        res2 = at.session_state["result"]
        sub = res2.adata.obs[res2.adata.obs[ckey].astype(str) == cid]
        assert set(sub["sccopilot_review"]) == {"edit"}, set(sub["sccopilot_review"])
        assert set(sub["sccopilot_final_type"]) == {"NK 细胞"}
        assert set(sub["sccopilot_type"]) == {draft}, "终稿不许覆盖 AI 首稿"
        assert "sccopilot_final_type" in res2.clusters.columns

        logs = list(tmp.glob("*.jsonl"))
        assert len(logs) == 1, logs
        rows = app.review.read_decisions(logs[0])
        assert len(rows) == 1 and rows[0]["final_type"] == "NK 细胞"
        assert rows[0]["ai_label"] == draft and rows[0]["note"] == "CD3D 阴性，改判 NK"
        assert rows[0]["run_id"], "决定要绑定运行指纹，换分辨率后不能沿用旧终稿"

        md_all = " ".join(str(m.value) for m in at.markdown)
        caps = " ".join(str(c.value) for c in at.get("caption"))
        assert "人工复核落回" in md_all, "导出区没说清终稿落到哪几列"
        assert "AI 首稿人工改动率" in caps, "改动率口径走 caption，别去 markdown 里找"

        # 同一簇再判一次「拒绝」：日志追加成两行，生效决定换成 human_rejected
        at.radio(key=f"review_action_{cid}").set_value("reject")
        at.form_submit_button(key=f"review_save_{cid}").click().run()
        assert not at.exception, [e.value for e in at.exception]
        rows = app.review.read_decisions(logs[0])
        assert [r["action"] for r in rows] == ["edit", "reject"], "append-only 被破坏了"
        sub2 = at.session_state["result"].adata.obs
        sub2 = sub2[sub2[ckey].astype(str) == cid]
        assert set(sub2["sccopilot_final_type"]) == {"human_rejected"}
        caps = " ".join(str(c.value) for c in at.get("caption"))
        assert "决定历史 2 条" in caps, caps[:300]

        # 空终稿不许入库：选了「修改类型」却没填名字，页面报错、日志仍只有两行
        at.radio(key=f"review_action_{cid}").set_value("edit")
        at.text_input(key=f"review_type_{cid}").set_value("")
        at.form_submit_button(key=f"review_save_{cid}").click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert len(app.review.read_decisions(logs[0])) == 2, "留空的修改不该写进日志"
        assert any("必须给出最终细胞类型" in str(e.value) for e in at.error), [e.value for e in at.error]
    finally:
        os.environ.pop("SCCOPILOT_REVIEW_DIR", None)
        shutil.rmtree(tmp, ignore_errors=True)


def test_app_survives_switching_to_missing_source():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "main.py"), default_timeout=300)
    at.run()                                   # 先跑一次，AppTest 才有 sidebar 这棵树
    assert not at.exception, [e.value for e in at.exception]
    at.sidebar.radio(key="src_kind").set_value("上传 .h5ad 文件")   # 此时并没有上传文件
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.get("plotly_chart"), "没有数据源时不该继续渲染上一份结果"
    md_all = " ".join(str(m.value) for m in at.markdown)
    # 引导文案按"下一步做什么"写，且跟着当前选的数据来源变（踩过：一律展示 pip 命令，
    # 用户以为要自己去装东西）。断言跟着文案走，别把它钉死成某句话。
    assert "开始使用" in md_all, "没数据时应展示上手引导而不是报错"
    assert "上传" in md_all, "选的是上传 .h5ad，引导却没提上传框"
    assert "Traceback" not in md_all and "pip install" not in md_all, \
        "默认视图把命令行步骤当成主要信息了，应该收在折叠区里"


# --------------------------------------------------------------------------- #
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
