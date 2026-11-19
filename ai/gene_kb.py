# -*- coding: utf-8 -*-
"""本地基因知识库：AI 防幻觉的第一道与最后一道闸门。

为什么需要这个文件：大模型对"没见过的基因名"会礼貌地编一个答案。要拦住它，
程序侧必须自己知道两件事——
  1) 哪些 symbol 是真实存在且**可用于定义细胞类型**的（MARKER_SETS / KNOWN_SYMBOLS）；
  2) 哪些基因虽然是真名，但对分群毫无信息量（UNINFORMATIVE：管家基因、核糖体、
     线粒体、应激基因）。真实项目里最骗人的簇就是"整簇 Top 基因全是 MT-* + B2M"。

三道用途：
  - 调用 LLM 前：算 recognizable_ratio，低于阈值直接拒绝请求（省钱 + 杜绝瞎编）；
  - 调用 LLM 后：审计模型引用的基因，列表外的一律标为"疑似编造"；
  - 无 API Key / 断网时：用 canonical_markers 加权投票给出**明确标注为规则推断**的兜底答案。

标注约定：本表按人类（human）常用命名整理；小鼠数据由大小写风格自动识别后
统一转大写比对（见 normalize_symbol）。
"""

from __future__ import annotations

import re
from collections.abc import Iterable

__all__ = [
    "CELL_TYPES",
    "HETEROGENEITY_REFERENCE",
    "KNOWN_SYMBOLS",
    "UNINFORMATIVE",
    "MITO_RE",
    "RIBO_RE",
    "is_valid_symbol",
    "normalize_symbol",
    "looks_mouse",
    "is_uninformative",
    "recognize",
    "vote_cell_types",
    "type_panels",
    "heterogeneity_panels",
]

# --------------------------------------------------------------------------- #
# 合法基因名字符集（同时也是 prompt 注入的输入清洗规则）
# --------------------------------------------------------------------------- #
_SYMBOL_STRICT = re.compile(r"^[A-Za-z0-9\-_\.]{1,30}$")
_MITO_RE = re.compile(r"^(MT-|mt-|chrM|ChrM)")
# RPLP0 / RPSA 也是核糖体基因：只写 RP[LS]\d 会漏掉它们，于是"哪都是"的读数被当成可用证据
_RIBO_RE = re.compile(r"^(RP[LS]|Rp[ls])[0-9A-Za-z]|^FAU$")
# 真实但"读不出生物学含义"的注释名：这些出现越多，越说明注释表没做好
# Ensembl 稳定 ID 的物种前缀不止人类（ENSG / ENSMUSG / ENSRRNO / ENSDARG…）
_UNANNOTATED_RE = re.compile(
    r"^(ENS[A-Z]{0,3}G\d|LOC\d+|LINC\d|MIR\d+P?$|AC\d{5,}|AF\d{5,}|U\d{5,}|SOX\d+-ps|"
    r".*-AS\d$|.*-ps\d$|SNOR|sno|intergenic|Unassigned)"
)

MITO_RE, RIBO_RE = _MITO_RE, _RIBO_RE


def is_valid_symbol(name: object) -> bool:
    """是否长得像一个基因 symbol。同时充当注入防护：非法字符直接不进 prompt。"""
    return isinstance(name, str) and bool(_SYMBOL_STRICT.match(name.strip()))


def looks_unannotated(name: str) -> bool:
    return bool(_UNANNOTATED_RE.match(name))


def normalize_symbol(name: str) -> str:
    """统一成大写比对形式（小鼠 Cd3d → CD3D）。"""
    return name.strip().upper()


def looks_mouse(genes: list[str]) -> bool:
    """小鼠命名惯例：首字母大写其余小写（Cd3d）；人类全大写（CD3D）。"""
    # 只统计“像个 symbol”的 token：带空格/分号的注入文本会首字母大写，混进来会把物种判错
    plain = [g for g in (str(x).strip() for x in genes) if g and _SYMBOL_STRICT.match(g)]
    if not plain:
        return False
    mixed = sum(1 for g in plain if g[:1].isupper() and g[1:2].islower())
    return mixed / len(plain) > 0.5


# --------------------------------------------------------------------------- #
# 对分群无信息量的基因：不能作为细胞类型证据
# --------------------------------------------------------------------------- #
UNINFORMATIVE: set[str] = {
    # 管家 / 结构
    "B2M", "ACTB", "GAPDH", "EEF1A1", "EEF2", "RPL13A", "MALAT1", "FTL", "FTH1",
    "TMSB4X", "TMSB10", "PGBP1", "ATP5F1B", "MT-CO1", "MT-CO2", "MT-ND4", "MT-CYB",
    "MT-ATP6", "MT-CO3", "MT-ND1", "MT-ND2", "MT-CO1-ps1",
    # 常见 "哪都是" 的应激/线粒体读数
    "NDUFA1", "NDUFB5", "COX7C", "ATP5MC1", "SRM", "ODC1", "JUN", "FOS", "FOSB",
    "EGR1", "HSPA1A", "HSPA1B", "DNAJB1", "HMOX1", "ZBTB16",
}


def is_uninformative(name: str) -> bool:
    """管家/核糖体/线粒体/应激基因，或无注释占位名——都不配当 marker。"""
    s = normalize_symbol(name)
    return (
        s in UNINFORMATIVE
        or bool(_MITO_RE.match(name))
        or bool(_RIBO_RE.match(name))
        or looks_unannotated(s)
    )


# --------------------------------------------------------------------------- #
# 细胞类型 → 标志基因
#   canonical：单独出现即有较强指示性；supportive：需要与 canonical 组合才成立
# --------------------------------------------------------------------------- #
CELL_TYPES: dict[str, dict[str, object]] = {
    "CD4+ T 细胞": {
        "canonical": ["CD3D", "CD3E", "CD3G", "CD2", "TRAC", "TRBC2", "CD4"],
        "supportive": ["IL7R", "CD28", "ITK", "SKAP1", "LCK"],
        "lineage": "T 淋巴细胞 / 适应性免疫",
    },
    "初始 CD4+ T 细胞 (naive)": {
        "canonical": ["CCR7", "SELL", "TCF7", "LEF1"],
        "supportive": ["IL7R", "CD4", "NELL2", "SATB1"],
        "lineage": "T 淋巴细胞 / 未活化",
    },
    "细胞毒性 CD8+ T 细胞": {
        "canonical": ["CD8A", "CD8B", "GZMK", "GZMA", "CCL5"],
        "supportive": ["GNLY", "PRF1", "NKG7", "KLRG1", "CX3CR1", "CD3D"],
        "lineage": "T 淋巴细胞 / 细胞毒",
    },
    "组织驻留/适应性 CD8+ T 细胞": {
        "canonical": ["ZNF683", "CXCR4", "IKZF2"],
        "supportive": ["CD8B", "FCGR3A", "GZMK", "KLRD1"],
        "lineage": "T 淋巴细胞 / 效应记忆",
    },
    "γδ T 细胞": {
        "canonical": ["TRDC", "TRGC1", "TRGC2", "KLRB1"],
        "supportive": ["CD3D", "SELL", "GZMK"],
        "lineage": "T 淋巴细胞 / 固有样",
    },
    "调节性 T 细胞 (Treg)": {
        "canonical": ["FOXP3", "IL2RA", "IKZF2", "CTLA4"],
        "supportive": ["TNFRSF18", "LRRC32", "FOXP3", "TIGIT"],
        "lineage": "T 淋巴细胞 / 免疫抑制",
    },
    "Th17 / 黏膜相关 T 细胞": {
        "canonical": ["RORC", "CCR6", "IL23R"],
        "supportive": ["IL17A", "KLRB1", "CCL20"],
        "lineage": "T 淋巴细胞 / 辅助",
    },
    "初始 B 细胞 (naive B)": {
        "canonical": ["MS4A1", "CD79A", "CD79B", "IGHD", "CD19", "BANK1"],
        "supportive": ["TNFRSF13C", "IL32", "PAX5", "TCL1A", "FCRL1"],
        "lineage": "B 淋巴细胞 / 适应性免疫",
    },
    "记忆 B 细胞 / 浆母": {
        "canonical": ["CD27", "TNFRSF13B"],
        "supportive": ["MS4A1", "CD79A", "IGHG1", "SDC1"],
        "lineage": "B 淋巴细胞 / 抗原经历",
    },
    "浆细胞 (plasma cell)": {
        "canonical": ["JCHAIN", "MZB1", "DERL3", "XBP1", "PRDM1"],
        "supportive": ["SDC1", "IGHG1", "TNFRSF17", "IRF4"],
        "lineage": "B 淋巴细胞终末分化 / 抗体分泌",
    },
    "NK 细胞": {
        "canonical": ["GNLY", "NKG7", "KLRD1", "KLRF1", "PRF1", "GZMB"],
        "supportive": ["NCAM1", "KIR2DL3", "KLRC1", "TYROBP", "FCGR3A"],
        "lineage": "固有淋巴细胞 / 细胞毒",
    },
    "经典单核细胞 (CD14+)": {
        "canonical": ["CD14", "LYZ", "S100A8", "S100A9", "FCN1", "VCAN", "CSF3R"],
        "supportive": ["CD36", "FN1", "MNF1", "ADAMDEC1"],
        "lineage": "髓系 / 单核",
    },
    "非经典单核细胞 (CD16+)": {
        "canonical": ["FCGR3A", "MS4A7", "LILRB2", "SERPINA1"],
        "supportive": ["CD14", "CYBB", "CSF1R", "TCF7L2"],
        "lineage": "髓系 / 单核",
    },
    "树突状细胞 (cDC2/DC)": {
        "canonical": ["FCER1A", "CD1C", "CLEC10A"],
        "supportive": ["HLA-DRA", "CST3", "TLR4", "FLT3"],
        "lineage": "髓系 / 抗原递呈",
    },
    "浆样树突状细胞 (pDC)": {
        "canonical": ["LILRA4", "CLEC4C", "IL3RA", "TCF4"],
        "supportive": ["GZMB", "SERPINF1", "IRF7", "RUNX2"],
        "lineage": "髓系 / 病毒感应 I 型干扰素",
    },
    "巨噬细胞": {
        "canonical": ["CD68", "MRC1", "MARCO", "C1QA", "C1QB"],
        "supportive": ["APOE", "SPP1", "LYVE1", "MAF", "ITGAX"],
        "lineage": "髓系 / 组织驻留吞噬",
    },
    "中性粒细胞": {
        "canonical": ["FCGR3B", "CSF3R", "MPO", "ELANE", "CEACAM8"],
        "supportive": ["CXCR2", "SELL", "MMP8", "ITGAM"],
        "lineage": "髓系 / 固有炎症",
    },
    "肥大细胞 / 嗜碱粒细胞": {
        "canonical": ["KIT", "CPA3", "MS4A2", "HDC", "TPSAB1"],
        "supportive": ["GATA2", "CMA1", "ENPP3"],
        "lineage": "髓系 / 过敏与抗寄生虫",
    },
    "血小板": {
        "canonical": ["PPBP", "PF4", "TUBB1", "NRGN"],
        "supportive": ["GP9", "ITGA2B", "PF4V1", "THBS1"],
        "lineage": "巨核细胞碎片 / 常为技术噪音",
    },
    "造血干/祖细胞 (HSPC)": {
        "canonical": ["CD34", "KIT", "MLLT3", "PROM1"],
        "supportive": ["GATA2", "MYB", "AVP", "CRHBP"],
        "lineage": "造血前体",
    },
    "红系细胞": {
        "canonical": ["HBB", "HBA1", "HBA2", "SLC4A1", "ALAS2"],
        "supportive": ["AHSP", "GATA1", "KLF1"],
        "lineage": "红细胞系 / 常为溶血残留",
    },
    "内皮细胞": {
        "canonical": ["PECAM1", "CDH5", "VWF", "KDR"],
        "supportive": ["CLDN5", "TEK", "EGFL7"],
        "lineage": "非造血 / 组织来源污染",
    },
    "上皮细胞": {
        "canonical": ["EPCAM", "KRT8", "KRT18", "CDH1"],
        "supportive": ["KRT7", "CLDN4", "GRP78"],
        "lineage": "非造血 / 组织来源",
    },
}

# 预编译：全量已知 symbol（含 supportive），供 recognizable_ratio 与输出审计使用
KNOWN_SYMBOLS: set[str] = {normalize_symbol(g) for v in CELL_TYPES.values() for g in (v["canonical"] + v["supportive"])}  # type: ignore[index]
KNOWN_SYMBOLS |= UNINFORMATIVE
# 常见但不用于定类型的免疫学genes：加入以免被判成"编造"
KNOWN_SYMBOLS |= {
    "HLA-A", "HLA-B", "HLA-C", "HLA-DRA", "HLA-DRB1", "CD45", "PTPRC", "CD40LG",
    "IFNG", "TNF", "IL6", "IL1B", "IL10", "CXCL8", "CCL2", "CCL3", "CCL4",
    "MKI67", "TOP2A", "CCND1", "GZMH", "GZMB", "NKG7", "S100A12", "SELL",
    "ADGRE1", "CSF1R", "MERTK", "CD52", "LTB", "COTL1", "NAMPT", "GPR65",
    "VPREB3", "PEG10", "GPX1", "CA2", "S100A4", "LGALS1", "IFITM3", "LYAR",
}

# 惯用名 → 基因符号。数据里给的是 symbol（MS4A1），模型和文献爱说 CD 编号（CD20）
# 或 NKG/KIR 这类受体俗名（NKG2D）。两者是同一个分子；审计层不认这层别名，就会把正确
# 答案判成"编造基因"并降级。这里只收血液免疫学常用、命名无争议的对应（拿不准的一律不写）。
GENE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "CD2": ("CD2",), "CD3": ("CD3D", "CD3E", "CD3G", "CD247"), "CD4": ("CD4",),
    "CD5": ("CD5",), "CD7": ("CD7",), "CD8": ("CD8A", "CD8B"), "CD9": ("CD9",),
    "CD10": ("MME",), "CD11B": ("ITGAM",), "CD11C": ("ITGAX",), "CD14": ("CD14",),
    "CD15": ("FUT4",), "CD16": ("FCGR3A", "FCGR3B"), "CD19": ("CD19",),
    "CD20": ("MS4A1",), "CD21": ("CR2",), "CD22": ("CD22",), "CD23": ("FCER2",),
    "CD25": ("IL2RA",), "CD26": ("DPP4",), "CD27": ("CD27",), "CD28": ("CD28",),
    "CD31": ("PECAM1",), "CD33": ("CD33",), "CD34": ("CD34",), "CD36": ("CD36",),
    "CD38": ("CD38",), "CD40": ("CD40",), "CD41": ("ITGA2B",), "CD44": ("CD44",),
    "CD45": ("PTPRC",), "CD49B": ("ITGA2",), "CD49D": ("ITGA4",), "CD54": ("ICAM1",),
    "CD56": ("NCAM1",), "CD61": ("ITGB3",), "CD62L": ("SELL",), "CD64": ("FCGR1A",),
    "CD69": ("CD69",), "CD71": ("TFRC",), "CD79": ("CD79A", "CD79B"),
    "CD80": ("CD80",), "CD86": ("CD86",), "CD94": ("KLRD1",), "CD95": ("FAS",),
    "CD98": ("SLC3A2",), "CD103": ("ITGAE",), "CD115": ("CSF1R",),
    "CD122": ("IL2RB",), "CD123": ("IL3RA",), "CD127": ("IL7R",),
    "CD134": ("TNFRSF4",), "CD137": ("TNFRSF9",), "CD152": ("CTLA4",),
    "CD154": ("CD40LG",), "CD158": ("KIR2DL1", "KIR2DL2", "KIR2DL3", "KIR2DL4"),
    "CD161": ("KLRB1",), "CD183": ("CXCR3",), "CD184": ("CXCR4",),
    "CD195": ("CCR5",), "CD244": ("CD244",), "CD274": ("CD274",), "CD279": ("PDCD1",),
    "CD278": ("ICOS",), "CD357": ("TNFSF13B",),
    "NKG2": ("KLRC1", "KLRC2", "KLRD1"), "NKG2A": ("KLRC1",), "NKG2B": ("KLRC1",),
    "NKG2C": ("KLRC2",), "NKG2D": ("KLRK1",), "NKG2E": ("KLRC3",),
}


def alias_targets(name: str) -> tuple[str, ...]:
    """惯用名（CD 编号 / NKG 俗名）对应的基因符号；不是别名就返回空。"""
    return GENE_SYNONYMS.get(normalize_symbol(name), ())


def recognize(genes: list[str]) -> dict[str, object]:
    """把基因列表按"能不能当证据"分箱，给出可解释的计数。

    informative   —— 已知 symbol 且不是管家/应激类，可直接作证据
    uninformative —— 真名但无分类价值（B2M、MT-*、RPL*…）
    unknown       —— 命名合法但不在本知识库（非免疫组织常见），可提示模型但需降级置信度
    unannotated   —— LOC/LINC/ENSG/AC… 等注释缺失名，属于"数据没注释好"而非生物学信号
    invalid       —— 含非法字符（潜在 prompt 注入文本），不进 prompt
    """
    buckets: dict[str, list[str]] = {"informative": [], "uninformative": [], "unknown": [],
                                    "unannotated": [], "invalid": []}
    seen: set[str] = set()
    for raw in genes:
        if not is_valid_symbol(raw):
            buckets["invalid"].append(str(raw)[:20])
            continue
        g = normalize_symbol(raw)
        if g in seen:
            continue
        seen.add(g)
        if _MITO_RE.match(raw) or _RIBO_RE.match(raw) or g in UNINFORMATIVE:
            buckets["uninformative"].append(g)
        elif looks_unannotated(g):
            buckets["unannotated"].append(g)
        elif g in KNOWN_SYMBOLS:
            buckets["informative"].append(g)
        else:
            buckets["unknown"].append(g)
    total = sum(len(v) for v in buckets.values()) or 1
    return {
        **buckets,
        "n_total": total,
        "informative_ratio": round(len(buckets["informative"]) / total, 3),
        # 注释缺失 + 非法名合并：这是"证据不可用"的主判据
        "unannotated_ratio": round(
            (len(buckets["unannotated"]) + len(buckets["invalid"])) / total, 3
        ),
        "unknown_ratio": round(len(buckets["unknown"]) / total, 3),
    }


def vote_cell_types(informative: list[str], top: int = 3) -> list[dict[str, object]]:
    """本地加权投票：canonical 记 2 分、supportive 记 1 分。

    这是断网/无 Key 时的兜底，也是给 LLM 的"先验提示"（但绝不代表结论）。
    分数只反映"匹配了几个已知标志基因"，不等于真实置信度，故输出里必须标注来源。
    """
    genes = {normalize_symbol(g) for g in informative}
    scored: list[dict[str, object]] = []
    for name, spec in CELL_TYPES.items():
        canonical = [g for g in spec["canonical"] if g in genes]       # type: ignore[index]
        supportive = [g for g in spec["supportive"] if g in genes]     # type: ignore[index]
        score = 2 * len(canonical) + len(supportive)
        if score:
            scored.append({
                "cell_type": name,
                "score": score,
                "canonical_hits": canonical,
                "supportive_hits": supportive,
                "lineage": spec["lineage"],                            # type: ignore[arg-type]
            })
    scored.sort(key=lambda d: (-d["score"], d["cell_type"]))
    return scored[:top]


# --------------------------------------------------------------------------- #
# 异质性判定的候选类型面板
# --------------------------------------------------------------------------- #
# 参照集：PBMC/外周免疫里最常共存的类型。为什么不能只用本簇 Top 基因投票出来的那几类——
# 一个"90% T + 10% 单核"的混合簇，Top 差异基因几乎全是 T 的，投票列表里根本没有单核，
# 于是那 10% 的细胞没有任何面板可以去匹配，混合就被读成了"判读不出"而不是"混了第二种细胞"。
# 参照集固定写死（不是模型生成的），换数据不变、重跑必得同一组面板。
HETEROGENEITY_REFERENCE: tuple[str, ...] = (
    "CD4+ T 细胞",
    "细胞毒性 CD8+ T 细胞",
    "初始 B 细胞 (naive B)",
    "NK 细胞",
    "经典单核细胞 (CD14+)",
    "树突状细胞 (cDC2/DC)",
    "浆细胞 (plasma cell)",
    "血小板",
)


def type_panels(names: Iterable[str]) -> dict[str, list[str]]:
    """细胞类型名 → 判型面板基因（canonical + supportive，剔除无信息量基因）。

    面板只用于给**单个细胞**打分，不当作证据列表，因此不改变任何已有的结论口径。
    """
    out: dict[str, list[str]] = {}
    for name in names:
        spec = CELL_TYPES.get(str(name))
        if not spec:
            continue
        genes = [g for g in list(spec["canonical"]) + list(spec["supportive"])   # type: ignore[operator]
                 if is_valid_symbol(g) and not is_uninformative(g)]
        out[str(name)] = list(dict.fromkeys(normalize_symbol(g) for g in genes))
    return out


def heterogeneity_panels(informative: list[str], *, top_k: int = 5) -> dict[str, list[str]]:
    """参照集 ∪ 本簇投票前列，构成异质性判定的候选类型集合。"""
    votes = [str(v["cell_type"]) for v in vote_cell_types(informative, top=top_k)]  # type: ignore[arg-type]
    return type_panels(list(dict.fromkeys(votes + list(HETEROGENEITY_REFERENCE))))

