# -*- coding: utf-8 -*-
"""单文件 HTML 存证包：一次分析的输入指纹、参数、逐簇证据与模型原文，打包成可离线核对的报告。

企业里的真实场景是三个月后 QA、审计或审稿人问「这个结论怎么来的」：对方不装 Python
也能打开这份文件逐条核对。因此这里只做「搬运运行记录」，不二次计算任何统计量；
所有不可信文本（基因名、模型回复、notes）一律转义，且不引用任何外部资源。
"""
from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from core import review

# prompt 是否真的发出去过，取决于 source：本地兜底与闸门拦截都没有产生 API 调用。
PROMPT_STATE = {
    "llm": "已发送的 prompt 原文",
    "local_kb": "按同一输入构造、但未发送的 prompt（本次未调用大模型）",
    "gate": "未构造：证据闸门在构造 prompt 之前就拦截了",
}

_STYLE = """
body{font:14px/1.65 -apple-system,'Segoe UI','Microsoft YaHei',sans-serif;margin:0;
     background:#f6f7fb;color:#1c2333}
main{max-width:1080px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:34px 0 10px;
   border-left:4px solid #3b6df0;padding-left:10px} h3{font-size:15px;margin:20px 0 8px}
p.sub{color:#5a6478;margin:0 0 18px}
table{border-collapse:collapse;width:100%;background:#fff;margin:8px 0 14px;font-size:13px}
th,td{border:1px solid #dfe3ee;padding:5px 9px;text-align:left;vertical-align:top}
th{background:#eef1fa;font-weight:600;white-space:nowrap}
td.num{text-align:right;font-variant-numeric:tabular-nums}
pre{background:#fff;border:1px solid #dfe3ee;border-left:4px solid #9aa7c7;padding:10px 12px;
    white-space:pre-wrap;word-break:break-word;font:12px/1.6 Consolas,'Courier New',monospace;
    margin:6px 0 14px}
.card{background:#fff;border:1px solid #dfe3ee;border-radius:8px;padding:14px 16px;margin:14px 0}
.tag{display:inline-block;border-radius:4px;padding:1px 8px;font-size:12px;margin-right:6px;
     background:#eef1fa;color:#33406b}
.tag.bad{background:#fdecec;color:#a12626} .tag.ok{background:#e8f6ec;color:#1d6b34}
code{background:#eef1fa;border-radius:3px;padding:0 4px;font-size:12px}
footer{color:#8a93a8;font-size:12px;margin-top:40px}
"""


def _e(v: Any) -> str:
    return html.escape("" if v is None else str(v), quote=True)


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _table(headers: list[str], rows: list[list[Any]], numeric: set[int] = frozenset()) -> str:
    out = ["<table><thead><tr>"]
    out += [f"<th>{_e(h)}</th>" for h in headers]
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>")
        for i, cell in enumerate(r):
            cls = ' class="num"' if i in numeric else ""
            out.append(f"<td{cls}>{_e(_fmt(cell))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _kv_table(d: Mapping[str, Any]) -> str:
    return _table(["项", "值"], [[k, v] for k, v in d.items()])


def _pre(text: str) -> str:
    return f"<pre>{_e(text)}</pre>"


def _obs_writeback(result: Any, cluster: str) -> str:
    obs = result.adata.obs
    cols = [c for c in ("sccopilot_type", "sccopilot_conf", "sccopilot_source", "sccopilot_het",
                        "sccopilot_review", "sccopilot_final_type")
            if c in obs.columns]
    if not cols:
        return "<p>本次运行尚未把 AI 结论写回 <code>obs</code>。</p>"
    sub = obs[obs[result.cluster_key].astype(str) == str(cluster)]
    rows = []
    for col in cols:
        for val, n in sub[col].value_counts().items():
            rows.append([col, val, int(n)])
    return _table(["obs 列", "取值", "细胞数"], rows, numeric={2})


def _review_block(row: Mapping[str, Any] | None) -> str:
    if not row:
        return ('<p>人工复核：<span class="tag">未复核</span>'
                "终稿留空，不以 AI 首稿冒充人工定稿。</p>")
    action = str(row.get("action", ""))
    changed = str(row.get("final_type", "")) != str(row.get("ai_label", ""))
    tag = "bad" if action == "reject" else ("ok" if not changed else "")
    return (
        f'<p>人工复核：<span class="tag {tag}">'
        f'{_e(review.ACTION_LABELS.get(action, action))}</span>'
        f'首稿 <code>{_e(row.get("ai_label"))}</code> → 终稿 <code>{_e(row.get("final_type"))}</code>'
        f'{"（标签有改动）" if changed else "（标签未变）"}</p>'
        + _kv_table({"复核人": row.get("reviewer") or "未署名",
                     "备注": row.get("note") or "无",
                     "决定时间 (UTC)": row.get("ts")})
    )


def _review_section(result: Any, decisions: Iterable[Mapping[str, Any]]) -> str:
    log = list(decisions)
    cids = result.clusters["cluster"].astype(str).tolist()
    parts = ["<h2>五、人工复核记录</h2>"]
    if not log:
        parts.append("<p>本次运行没有人工复核记录：所有簇的 <code>sccopilot_final_type</code> 留空，"
                     "<code>sccopilot_review</code> 为 unreviewed。</p>")
        return "".join(parts)
    s = review.review_stats(log, cids)
    parts.append("<p>口径：改动率 = 最终标签与首稿 obs 标签不同的簇数 ÷ <strong>已复核簇数</strong>；"
                 "覆盖率 = 已复核簇数 ÷ 全部簇数。日志只追加，同一簇后写的决定生效，历史行全部保留。</p>")
    parts.append(_kv_table({
        "日志条数（含历史）": s["n_decisions"],
        "已复核簇数": s["n_reviewed"],
        "全部簇数": s["n_clusters"],
        "复核覆盖率": f"{s['coverage']:.1%}",
        "采纳首稿": s["n_accept"],
        "修改类型": s["n_edit"],
        "否决首稿": s["n_reject"],
        "标签实际改动簇数": s["n_changed"],
        "AI 首稿人工改动率": f"{s['change_rate']:.1%}",
    }))
    headers = ["簇", "AI 首稿（页面）", "首稿 obs 标签", "动作", "最终标签",
               "备注", "复核人", "来源", "置信度", "时间 (UTC)"]

    def rows_of(items: Iterable[Mapping[str, Any]]) -> list[list[Any]]:
        return [[r.get("cluster"), r.get("ai_type"), r.get("ai_label"),
                 review.ACTION_LABELS.get(str(r.get("action", "")), r.get("action")),
                 r.get("final_type"), r.get("note") or "无", r.get("reviewer") or "未署名",
                 r.get("source"), r.get("confidence"), r.get("ts")] for r in items]

    latest = review.latest_by_cluster(log)
    order = sorted(latest, key=lambda c: (len(c), c))
    parts.append("<p>当前生效的决定：</p>")
    parts.append(_table(headers, rows_of(latest[c] for c in order), numeric={8}))
    if s["unreviewed_clusters"]:
        parts.append("<p>未复核簇（终稿留空）：" + _e("、".join(s["unreviewed_clusters"])) + "</p>")
    else:
        parts.append("<p>全部簇均已复核。</p>")
    if len(log) > len(latest):
        parts.append("<p>决策历史（逐行照搬 jsonl，含被后续决定覆盖的行）：</p>")
        parts.append(_table(headers, rows_of(log), numeric={8}))
    return "".join(parts)


def _cluster_section(result: Any, cid: str, wf: Mapping[str, Any] | None,
                     review_row: Mapping[str, Any] | None = None) -> str:
    cl_row = result.clusters[result.clusters["cluster"].astype(str) == str(cid)]
    parts = [f'<div class="card"><h3>簇 {_e(cid)}</h3>']
    if len(cl_row):
        parts.append(_kv_table(cl_row.iloc[0].to_dict()))

    itp = (wf or {}).get("interpretation")
    if itp is None:
        parts.append("<p>本簇未解读：没有 AI 结论，也没有 prompt / 回复记录。</p>")
        parts.append(_review_block(review_row))
        parts.append(_obs_writeback(result, cid))
        parts.append("</div>")
        return "".join(parts)

    src = getattr(itp, "source", "")
    tags = [f'<span class="tag">{_e(src)}</span>',
            f'<span class="tag">置信度 {_e(itp.confidence_label)}</span>']
    if getattr(itp, "refused", False):
        tags.append('<span class="tag bad">拒答</span>')
    if getattr(itp, "cache_hit", False):
        tags.append(f'<span class="tag">缓存命中 · 首次请求 {_e(itp.cached_at)}</span>')
    parts.append("<p>" + "".join(tags) + "</p>")
    parts.append(_kv_table({
        "预测类型": itp.predicted_type,
        "置信度分数": itp.confidence_score,
        "模型": itp.model or "未调用大模型",
        "耗时 s": itp.latency_s,
        "prompt 字符数": itp.prompt_chars,
    }))

    gate = getattr(itp, "gate", {}) or {}
    if gate:
        ok = gate.get("ok")
        parts.append(f'<p>证据闸门：<span class="tag {"ok" if ok else "bad"}">'
                     f'{"证据充足" if ok else "证据不足"}</span></p>')
        for key, label in (("reasons", "拦截原因"), ("suggestions", "改进建议"),
                           ("warnings", "其他提示")):
            vals = gate.get(key) or []
            if vals:
                parts.append(f"<p>{label}：</p><ul>"
                             + "".join(f"<li>{_e(v)}</li>" for v in vals) + "</ul>")

    ev = result.evidence_of(cid)
    mk = result.markers[result.markers["cluster"].astype(str) == str(cid)]
    mk = mk.set_index("gene") if len(mk) else mk
    rows = []
    for _, r in ev.iterrows():
        g = str(r["gene"])
        m = mk.loc[g] if g in mk.index else {}
        cat = "线粒体" if r.get("is_mito") else ("未注释" if r.get("is_novel") else "已知")
        rows.append([g, r.get("mean_expr"), r.get("pct_expressed"),
                     m.get("logfoldchanges"), m.get("scores"), m.get("pvals_adj"), cat])
    parts.append("<p>证据表（one-vs-rest Wilcoxon，p 为 BH 校正后）：</p>")
    parts.append(_table(["基因", "平均表达", "簇内检出率", "log2FC", "差异打分", "校正 p", "类别"],
                        rows, numeric={1, 2, 3, 4, 5}))

    state = PROMPT_STATE.get(src, "prompt 原文")
    if src == "llm" and getattr(itp, "cache_hit", False):
        state = "首次请求时已发送的 prompt 原文（本次命中缓存，未重新发送）"
    parts.append(f"<p>{_e(state)}：</p>")
    parts.append(_pre(getattr(itp, "prompt", "") or "（无记录）"))
    raw = getattr(itp, "raw", "")
    if raw:
        parts.append("<p>模型回复原文（逐字，未改写）：</p>")
        parts.append(_pre(raw))
    else:
        parts.append("<p>无模型回复：本次结论由本地规则产生。</p>")
    parts.append("<p>推断理由（页面展示版）：</p>")
    parts.append(_pre(itp.reasoning))

    for key, label in (("cited_genes", "引用到的本次 marker"),
                       ("fabricated_genes", "审计判定为编造的基因"),
                       ("violations", "审计告警"), ("notes", "说明")):
        vals = getattr(itp, key, []) or []
        if vals:
            parts.append(f"<p>{label}：</p><ul>"
                         + "".join(f"<li>{_e(v)}</li>" for v in vals) + "</ul>")

    kg = (wf or {}).get("kg_candidates") or []
    if kg:
        parts.append("<p>知识图谱候选：</p>")
        parts.append(_table(["候选", "得分"],
                            [[c.get("name", c.get("cell_type", "?")), c.get("score")] for c in kg],
                            numeric={1}))
    het = (wf or {}).get("heterogeneity") or {}
    if het:
        parts.append("<p>簇内异质性判定：</p>")
        parts.append(_kv_table({k: het.get(k) for k in
                                ("dominant", "dominant_prop", "runner_up_type",
                                 "normalized_entropy", "assigned_prop")}))
    crit = (wf or {}).get("critique") or {}
    if crit:
        parts.append(f'<p>Critic：<span class="tag '
                     f'{"ok" if crit.get("status") == "approve" else "bad"}">'
                     f'{_e(crit.get("status"))}</span> {_e(crit.get("reason") or "")}</p>')
    trace = (wf or {}).get("trace") or []
    if trace:
        parts.append("<p>Agent 轨迹：</p>")
        parts.append(_table(["agent", "action", "耗时 s", "时间"],
                            [[t.get("agent_name"), t.get("action"), t.get("duration_s"),
                              t.get("timestamp")] for t in trace], numeric={2}))
    parts.append(_review_block(review_row))
    parts.append("<p>结论写回 <code>obs</code> 的取值分布：</p>")
    parts.append(_obs_writeback(result, cid))
    parts.append("</div>")
    return "".join(parts)


def build_audit_html(result: Any, workflows: Mapping[str, Mapping[str, Any]],
                     meta: Mapping[str, Any] | None = None,
                     reviews: Iterable[Mapping[str, Any]] | None = None) -> str:
    """把一次运行打包成自包含 HTML。meta 缺省项写「未记录」，不猜。

    ``reviews`` 是复核日志的原始行（append-only jsonl 读回来的列表），不是生效决定：
    历史行也要进报告，改动率才可由读者自己复算。
    """
    meta = dict(meta or {})
    log = list(reviews or [])
    latest = review.latest_by_cluster(log)
    meta.setdefault("generated_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    for k in ("data_source", "data_sha256", "code_commit"):
        meta.setdefault(k, "未记录")

    head = [
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<meta name=\"generator\" content=\"scCopilot audit bundle\">",
        f"<title>scCopilot 存证包 · {_e(meta['generated_at'])}</title>",
        f"<style>{_STYLE}</style></head><body><main>",
        "<h1>scCopilot 分析存证包</h1>",
        '<p class="sub">本文件自包含、可离线打开。所有数字与原文均搬运自本次运行记录，'
        "报告本身不做二次计算；核对方法见文末。</p>",
        "<h2>一、运行指纹</h2>",
        _kv_table({
            "生成时间 (UTC)": meta["generated_at"],
            "数据来源": meta["data_source"],
            "输入数据 sha256": meta["data_sha256"],
            "代码 commit": meta["code_commit"],
            "细胞数 / 基因数（QC 后）": f"{result.adata.n_obs} / {result.adata.n_vars}",
            "聚类键": result.cluster_key,
        }),
        "<h2>二、流水线参数</h2>",
        _kv_table(result.config),
        "<h2>三、QC 漏斗</h2>",
        _kv_table(result.qc),
        "<h2>四、逐簇证据与结论</h2>",
    ]
    order = sorted(workflows, key=lambda c: (len(str(c)), str(c)))
    seen = set()
    for cid in order:
        seen.add(str(cid))
        head.append(_cluster_section(result, str(cid), workflows[cid], latest.get(str(cid))))
    for cid in result.clusters["cluster"].astype(str):
        if cid not in seen:
            head.append(_cluster_section(result, cid, None, latest.get(cid)))

    head += [
        _review_section(result, log),
        "<h2>六、如何核对</h2>",
        "<ul>",
        "<li>输入数据指纹：<code>sha256sum &lt;file.h5ad&gt;</code>（Windows 用 "
        "<code>certutil -hashfile &lt;file.h5ad&gt; SHA256</code>），与上表逐字比对。</li>",
        "<li>生物学数字可复现：<code>python -X utf8 scripts/release_check.py --reexport</code> "
        "会重跑导出并逐值比对（忽略耗时/时间戳）。</li>",
        "<li>全量回归：<code>python -X utf8 scripts/run_tests.py</code>。</li>",
        "<li>prompt 状态标签区分「已发送 / 构造未发送 / 未构造」；带「缓存命中」标签的结论"
        "取自本地响应缓存，未产生新的 API 调用。</li>",
        "<li>拒答簇只写 <code>low_confidence</code>，不写具体类型；未解读簇写 "
        "<code>unannotated</code>。</li>",
        "<li>复核记录：导出区的「复核记录 JSONL」与第五节逐行对应（append-only，同簇后写生效）。"
        "改动率的分母是已复核簇数；未复核簇的 <code>sccopilot_final_type</code> 留空，"
        "AI 首稿始终保留在 <code>sccopilot_type</code>，两列可自行比对。</li>",
        "</ul>",
        '<footer>scCopilot 存证包 · 生成于 ' + _e(meta["generated_at"]) + "</footer>",
        "</main></body></html>",
    ]
    return "".join(head)