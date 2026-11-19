# -*- coding: utf-8 -*-
"""scCopilot 的 Prompt 契约层。

设计原则：**输出格式是接口，不是建议**。explainer._parse 按这里定义的三段标题解析，
所以任何修改都必须同时改解析器。三段标题（【预测细胞类型】/【置信度】/【推断理由】）
刻意用中文方括号，模型偏离概率远低于要求它输出 JSON 后再容错。

防幻觉的三处硬约束，都写在 System Prompt 里并在代码侧复核：
  1. 只准引用 GENES 列表内的基因   -> 代码侧 _audit_cited_genes() 复核；
  2. 低表达/管家基因不算证据        -> 代码侧先算 evidence gate，不合格根本不发请求；
  3. 证据不足必须写"证据不足，建议人工复核" -> 代码侧若模型仍给出类型名，强制改写并记违规。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "IMMUNOLOGIST_SYSTEM_PROMPT",
    "RESOLUTION_SYSTEM_PROMPT",
    "build_cluster_prompt",
    "build_resolution_prompt",
    "SECTION_TYPE",
    "SECTION_CONF",
    "SECTION_REASON",
]

SECTION_TYPE = "预测细胞类型"
SECTION_CONF = "置信度"
SECTION_REASON = "推断理由"

# --------------------------------------------------------------------------- #
# 细胞类型解读：免疫学家 + 反幻觉硬规则
# --------------------------------------------------------------------------- #
IMMUNOLOGIST_SYSTEM_PROMPT = f"""你是一位计算免疫学家（computational immunologist），长期从事流式细胞术、
mass cytometry 与单细胞转录组的细胞类型注释工作，熟悉人类外周血、淋巴组织与肿瘤微环境的
经典标志基因体系。你的用户是生信初学者，因此表述要专业但可懂。

【任务】
仅依据用户给出的某个 cluster 的 Top 差异表达基因列表（one-vs-rest Wilcoxon 排序）
及其客观表达证据，推断该细胞群最可能的细胞类型。

【硬性规则——违反任何一条即为失败回答】
1. 证据封闭性：只能引用 GENES 列表中出现的基因。禁止引用列表外的任何基因，
   禁止编造基因符号。若你想到的关键基因不在列表里，就写"本次输入未提供 X，无法验证"，
   而不是假装它高表达。
2. 表达量优先：mean_expr 为 log(CPM+1) 尺度，pct_expressed 为该簇中检出该基因的细胞比例。
   pct_expressed < 0.05 或 mean_expr < 0.1 的基因一律视为技术噪音，不得作为证据。
   log2FC 小且检出率低的基因即使排序靠前也不可信。
3. 无信息量基因：B2M、ACTB、GAPDH、MALAT1、FTL、FTH1、EEF1A1、核糖体蛋白（RPL*/RPS*）、
   线粒体基因（MT-*）以及应激即时早期基因（FOS/JUN/HSPA1A）属于全局高表达或应激信号，
   不能用于定义细胞类型。若某簇的 Top 基因基本由这类基因构成，
   结论必须是"证据不足，建议人工复核"，并指出这通常意味着低质量/濒死细胞或双细胞残留。
4. 不足即拒答：当满足规则 1-3 的有效 marker 少于 3 个，或剩余证据指向互相冲突的谱系
   （例如同时强指 B 细胞与髓系），【预测细胞类型】必须写"证据不足，建议人工复核"，
   并在【推断理由】中说明缺什么证据、建议怎么补（提高 n_markers、放宽检出率阈值、
   追加受体型 marker、或与流式/免疫组化结果交叉验证）。绝对禁止为了"给个答案"而猜测。
5. 允许主判 + 至多 2 个备选，但每一项都必须挂上它依据的具体基因名。
6. 置信度诚实且离散，只能取三档：
   高 = ≥3 个同一谱系的经典 marker 且无冲突证据；
   中 = 2 个经典 marker，或存在 1 项冲突/替代解释；
   低 = 仅 1 个经典 marker，或多数 marker 本地知识库未收录（不在你确定的标志基因体系内）。
   存在冲突证据时禁止给"高"。
7. 物种：基因名全大写（CD3D）按人类处理，首字母大写（Cd3d）按小鼠处理，并在理由中注明。
8. 先验不等于结论：若提供了「本地先验」，它只是本地规则的字面匹配结果，
   你可能同意也可能推翻它；推翻时必须在理由里说明依据。

【输出格式】严格三段，每段以中文方括号标题起头（方括号本身也要写），不要 Markdown 代码块，不要任何前后缀解释：
【{SECTION_TYPE}】：<中文名（英文缩写/谱系）>，或"证据不足，建议人工复核"
【{SECTION_CONF}】：<高|中|低>（0-1 数值，例：高（0.82））
【{SECTION_REASON}】：<不超过 200 字的分点理由，每点形如 "GENE：它在本簇中的意义">
"""


def build_cluster_prompt(
    *,
    cluster: str,
    rows: list[dict[str, Any]],
    cluster_stats: dict[str, Any] | None = None,
    species_hint: str = "human",
    local_prior: list[dict[str, Any]] | None = None,
    n_cells_dataset: int | None = None,
) -> str:
    """把"事实"排成紧凑表格喂给模型。只给事实，不给诱导性结论。

    rows 每行需含：gene / scores（或 score）/ pct_expressed / mean_expr；
    logfoldchanges、pvals_adj 有则给，整列缺失时该列不出现。
    """
    stats = cluster_stats or {}
    head = [f"CLUSTER {cluster}"]
    if stats.get("n_cells") is not None:
        piece = f"{stats['n_cells']} cells"
        if stats.get("pct_cells") is not None:
            piece += f" ({stats['pct_cells']}% of dataset)"
        elif n_cells_dataset:
            piece += f" / {n_cells_dataset}"
        head.append(piece)
    for key, label, fmt in (
        ("median_counts", "median UMI", "{:.0f}"),
        ("median_pct_mito", "median %mito", "{:.2f}%"),
    ):
        val = stats.get(key)
        if isinstance(val, (int, float)) and val == val:  # 过滤 NaN
            head.append(f"{label} " + fmt.format(val))
    head.append(f"species≈{species_hint}")

    lines = ["\n".join(head), ""]

    def _num(row: dict[str, Any], *keys: str) -> float | None:
        # core 的 marker 表列名是 scores，手写/测试数据常用 score，两个键都认
        v = next((row[k] for k in keys if isinstance(row.get(k), (int, float))), None)
        if v is None or v != v:          # v != v 过滤 NaN
            return None
        return float(v)

    # 最后一列是 mandatory：整列没值也照旧输出（缺了就是真缺，不该被静默丢掉）。
    columns = [
        ("score", ("score", "scores"), "{:.2f}", "score=统计量", True),
        ("lfc", ("logfoldchanges",), "{:.2f}", "lfc=log2FC", False),
        ("padj", ("pvals_adj",), "{:.2e}", "padj=Benjamini-Hochberg 校正 p", False),
        ("pct", ("pct_expressed",), "{:.2f}", "pct=簇内检出细胞比例", True),
        ("mean", ("mean_expr",), "{:.2f}", "mean=log(CPM+1)", True),
    ]
    # lfc/padj 由 rank_genes_groups 给出，不是每份证据表都带。整列没值时干脆不放这一列：
    # 一竖排 "NA" 既浪费 token，也会让模型把"没算这个统计量"读成"统计量不显著"。
    shown = [c for c in columns if c[4] or any(_num(r, *c[1]) is not None for r in rows)]
    lines.append("Top markers (one-vs-rest Wilcoxon; " + ", ".join(c[3] for c in shown) + "):")
    lines.append("\t".join(["rank", "gene"] + [c[0] for c in shown]))
    for i, r in enumerate(rows, start=1):
        cells = [c[2].format(v) if (v := _num(r, *c[1])) is not None else "NA" for c in shown]
        lines.append("\t".join([str(i), str(r.get("gene"))] + cells))

    # 显式白名单：审计阶段据此判定"模型是否引用了不存在的证据"
    genes = [str(r.get("gene")) for r in rows if r.get("gene")]
    lines += ["", f"GENES = {', '.join(genes)}"]

    unknown = [str(r.get("gene")) for r in rows if r.get("is_novel")]
    if unknown:
        lines.append(
            "本地未收录命名（知识库无法确认其命名，慎用作证据） = "
            + ", ".join(unknown)
        )

    if local_prior:
        lines.append("")
        lines.append("本地先验（本地字面匹配，仅供参考，不是结论）：")
        for p in local_prior:
            hits = ", ".join(list(p.get("canonical_hits", []))[:6]) or "-"
            lines.append(
                f"- {p.get('cell_type')} | score={p.get('score')} | 命中经典marker: {hits}"
            )

    lines += ["", "请按三段格式输出该 cluster 的解读。"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 分辨率变化解释
# --------------------------------------------------------------------------- #
RESOLUTION_SYSTEM_PROMPT = """你是一位单细胞分析教练，正在向初学者解释 Leiden 聚类分辨率变化带来的影响。

【背景原理（你可以使用，但不得编造具体数字）】
Leiden 通过一个分辨率参数控制社区发现的粒度：resolution 越大，算法越倾向把一个区域切成
多个小块（亚群被拆分，簇数上升、平均簇规模下降）；越小则越容易把转录组相近的群体并成一块
（簇被合并，簇数下降）。分辨率本身不引入新的生物学信息——它改变的是"在同一张邻居图上
切多细"，因此 UMAP 坐标不变，只是同一批细胞被重新贴上不同的簇标签。
分辨率调高后新拆出的簇，可能是真实亚群（如 CD8 T 分出 GZMK+ 与 GZMB+ 效应状态），
也可能只是连续状态（细胞周期、应激梯度）被人为切断，或是低质量细胞被单独立簇。
判据是：拆分出的簇是否有各自的、稳健且特异的标志基因。

【硬性规则】
1. 只允许引用用户给出的 CLUSTER_TABLE 中真实存在的数字（簇数、细胞数、marker 命中数）。
   禁止编造未提供的数值、基因名或细胞类型。
2. 若数据不足以判断（例如没有前后对比表），只讲通用原理，并明确写出
   "本次未获得前后聚类对比数据，以下为一般性说明"。
3. 不得把"分辨率更高"等同于"分辨率更好"；必须提示需要结合 marker 特异性与
   生物学可解释性人工确认，并给出下一步操作建议。
4. 语言：简体中文，不超过 180 字，2-4 句，不使用 Markdown 标题。

【输出格式】
【变化摘要】<一句话，引用真实数字说明簇数/规模如何变化>
【为什么会这样】<机制解释，可引用 resolution 数值>
【下一步建议】<可执行的验证动作>
"""


def build_resolution_prompt(
    *,
    resolution: float,
    prev_resolution: float | None = None,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    table_rows: list[dict[str, Any]] | None = None,
    split_pairs: list[str] | None = None,
) -> str:
    """组装分辨率解读请求。before/after 是"数值事实"，模型只能引用不能发挥。"""
    lines = [f"resolution = {resolution}"]
    if prev_resolution is not None:
        lines.append(f"previous_resolution = {prev_resolution}")
    if before:
        lines.append(
            "BEFORE: n_clusters={n_clusters}, median_cluster_size={median_size}, "
            "max_cluster_size={max_size}".format(**before)
        )
    if after:
        lines.append(
            "AFTER : n_clusters={n_clusters}, median_cluster_size={median_size}, "
            "max_cluster_size={max_size}".format(**after)
        )
    if split_pairs:
        lines.append("SPLIT_OR_MERGED_HINTS = " + "; ".join(split_pairs[:8]))

    if table_rows:
        lines.append("")
        lines.append("CLUSTER_TABLE (after clustering, per cluster):")
        lines.append("cluster\tn_cells\tn_strong_markers\tnovel_marker_ratio\tevidence_ok")
        for r in table_rows:
            lines.append(
                f"{r.get('cluster')}\t{r.get('n_cells')}\t{r.get('n_strong_markers')}\t"
                f"{r.get('novel_marker_ratio')}\t{r.get('evidence_ok')}"
            )
    else:
        lines.append("")
        lines.append("CLUSTER_TABLE = <未提供>")

    lines.append("")
    lines.append("请按三段格式解释这次分辨率调整的后果。")
    return "\n".join(lines)
