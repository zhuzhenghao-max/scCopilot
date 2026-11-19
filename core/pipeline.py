# -*- coding: utf-8 -*-
"""scCopilot 生信核心流水线（Scanpy 1.12.x 实测适配）。

职责边界：本模块只负责"算得准"——把 .h5ad 变成可解释的聚类结果 + 每个簇的
Top 差异基因 + **可供 AI 判断可信度的客观证据**（平均表达量/检出率/线粒体占比等）。
它不做任何生物学解释；解释由 ai/explainer.py 负责，保证 AI 拿到的永远是结构化事实。

内存策略（大数据集必看）：
1. 全程 float32 稀疏 CSR，绝不把 X 稠密化；
2. 用 ``layers['counts']`` 备份原始计数，代替 ``adata.raw``（后者会复制 var/obs 索引）；
3. HVG 用 ``mask_var`` 掩码参与 PCA，而不是 ``subset=True`` 真删基因——
   这样后续画 marker 条形图 / 基因表达着色时全基因集仍在；
4. 结束后 ``gc.collect()`` 回收中间对象。

注意：与 scanpy 惯例一致，``run_pipeline`` 会**原地修改**传入的 adata（写入 obs/var/obsm）。
"""

from __future__ import annotations

import gc
import importlib.util
import re
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from core.provenance import ProvenanceTracker

__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "load_h5ad",
    "run_pipeline",
    "recluster",
    "recluster_only",
    "extract_markers",
    "gene_column",
    "qc_metrics",
    "warmup",
    "QC_MIN_FIELDS",
]

# 进度回调：(阶段名, 0~1 完成度)。前端可在此抛异常实现"中断跑批"。
ProgressFn = Callable[[str, float], None]

# 基因名风格判定：真 symbol vs Ensembl/LOC/LINC 等人类与 AI 都读不懂的占位名
_SYMBOL_RE = re.compile(r"^[A-Za-z0-9\-\.]{2,20}$")
# Ensembl 稳定 ID 的物种前缀不止人类：ENSG、ENSMUSG/ENSRRNO（鼠）、ENSDARG（斑马鱼）…
_ENSEMBL_RE = re.compile(r"^ENS[A-Z]{0,3}G\d")
# 与 ai/gene_kb._UNANNOTATED_RE 同一口径：AC/AF/U*/SNOR 与 -AS/-ps 后缀也是"注释没做好"
# 的名字。漏掉它们会让「未注释比例」低于 ai 层实际判据，前端显示"证据充足"却被 AI 拒绝。
_NOVEL_RE = re.compile(
    r"^(ENS[A-Z]{0,3}G\d|LOC\d+|MIR\d|LINC\d|SOC\d|A[1-6]\d{3}\.\d|Unassigned|"
    r"intergenic_region|gene\d+|NA$|\?+$"
    r"|AC\d{5,}|AF\d{5,}|U\d{5,}|SNOR|sno|SOX\d+-ps|.*-AS\d$|.*-ps\d$)"
)
_MT_RE = re.compile(r"^(MT-|mt-|chrM|ChrM)")
# 与 ai/gene_kb._RIBO_RE 保持同一口径：RPLP0 / RPSA 也属于"哪都是"的核糖体读数
_RIBO_RE = re.compile(r"^(RP[LS]|Rp[ls])[0-9A-Za-z]|^FAU$")

# 结果里必须落盘的最小 QC 字段，供前端画散点/表格
QC_MIN_FIELDS = ("n_counts", "n_genes", "pct_mito", "pct_ribo")

_CFG_KEY = "sccopilot"          # adata.uns 下的元信息键
_WARN_KEY = "sccopilot_warnings"

# AI 结论落回 obs / 簇表的列（带 sccopilot_ 前缀，不撞用户自己的列）
ANNOTATE_COLS = ("sccopilot_type", "sccopilot_conf", "sccopilot_source", "sccopilot_het")
ANNOTATE_KEY = "sccopilot_annotation"        # adata.uns 下的落款
ANNOT_UNANNOTATED = "unannotated"            # 还没点过解读的簇
ANNOT_LOW_CONFIDENCE = "low_confidence"      # 拒答 / Critic 升级的簇：不写具体类型


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    """流水线全部可调参数。默认值面向 3' 体外显子（10x）人/鼠 PBMC 数据。"""

    # ---- QC：过滤阈值作用在 raw counts 上 ----
    min_genes: int = 200          # 每细胞最少检出基因数（去掉空液滴）
    max_genes: int | None = 6000  # 每细胞最多基因数（去掉双细胞/碎屑），None=不限
    min_counts: int = 500
    max_counts: int | None = None
    max_pct_mito: float = 20.0    # 线粒体基因占比上限（%），应激/死亡细胞主要靠它剔除
    min_cells: int = 3            # 每基因至少在 N 个细胞中检出
    filter_doublets: bool = False  # scrublet 双细胞检测（numba 首次编译较慢，默认关）

    # ---- 归一化与降维 ----
    target_sum: float = 1e4       # CPM 标准化目标
    n_top_genes: int = 2000
    hvg_flavor: str = "seurat"    # seurat | scanpy | cell_ranger | seurat_v3(需 counts)
    n_pcs: int = 50
    n_neighbors: int = 15

    # ---- 聚类与降维可视化 ----
    resolution: float = 1.0
    leiden_flavor: str = "auto"   # auto | igraph | leidenalg
    n_markers: int = 8            # 每个簇取 Top N 差异基因
    marker_method: str = "wilcoxon"

    # ---- 其它 ----
    random_state: int = 0
    keep_counts_layer: bool = True

    def resolved_leiden_flavor(self) -> str:
        """1.12 起 leiden 默认后端仍是 leidenalg，但官方声明未来会切到 igraph。

        igraph 快一个数量级，因此 auto 时优先 igraph；没装则回落 leidenalg。
        """
        if self.leiden_flavor != "auto":
            return self.leiden_flavor
        return "igraph" if importlib.util.find_spec("igraph") else "leidenalg"

    def validate(self) -> None:
        if self.resolution <= 0:
            raise ValueError("resolution 必须 > 0")
        if self.n_pcs < 2:
            raise ValueError("n_pcs 至少为 2")
        if not 0 < self.n_markers <= 50:
            raise ValueError("n_markers 需在 1~50 之间")
        if not 0 < self.max_pct_mito <= 100:
            raise ValueError("max_pct_mito 需在 (0, 100] 之间")


@dataclass
class PipelineResult:
    """一次跑批的完整产物，可直接交给 UI 与 AI 层。"""

    adata: ad.AnnData
    markers: pd.DataFrame          # 长表：cluster, rank, gene, score, mean_expr, pct_expressed…
    clusters: pd.DataFrame         # 每簇一行：规模、UMAP 质心、证据质量字段
    qc: dict[str, Any]             # 过滤漏斗统计
    timings: dict[str, float]
    config: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    @property
    def cluster_key(self) -> str:
        return str(self.adata.uns[_CFG_KEY]["cluster_key"])

    def genes_of(self, cluster: str, top: int | None = None) -> list[str]:
        """取某簇按 score 排序的 Top 基因名，供 ai/explainer 直接消费。"""
        sub = self.markers[self.markers["cluster"] == str(cluster)].sort_values("rank")
        genes = sub["gene"].tolist()
        return genes[:top] if top else genes

    def evidence_of(self, cluster: str) -> pd.DataFrame:
        """某簇 marker 的证据表（gene/mean_expr/pct_expressed/is_mito/is_novel）。"""
        return self.markers[self.markers["cluster"] == str(cluster)].copy()

    def evidence_records(self, cluster: str, top: int | None = None) -> list[dict[str, Any]]:
        """证据表转成 JSON 友好的记录列表，供 ai 层的证据闸门使用。

        只给基因名的话 assess_evidence 只能做名字层面判断，mean_expr / pct_expressed
        这两道硬阈值会整组失效——所以任何要把证据交给 AI 层的调用都走这里。
        """
        sub = self.evidence_of(cluster).sort_values("rank")
        if top:
            sub = sub.head(top)
        keep = [c for c in ("gene", "mean_expr", "pct_expressed", "logfoldchanges",
                           "scores", "pvals", "pvals_adj", "is_mito", "is_ribo", "is_novel")
                if c in sub.columns]
        return sub[keep].to_dict("records")

    def cluster_stats(self, cluster: str) -> dict[str, Any]:
        """某簇的规模与 QC 中位数，证据闸门的另一路输入。"""
        row = self.clusters[self.clusters["cluster"] == str(cluster)].iloc[0]
        return {
            "n_cells": int(row["n_cells"]),
            "median_counts": float(row["median_counts"]),
            "median_pct_mito": float(row["median_pct_mito"]),
        }

    def annotate(self, rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """把解读结论写回 ``obs``、簇表与 ``uns``，让注释能进用户自己的下游分析。

        ``rows`` 每项形如 ``{"type", "conf", "source", "het", "blocked"}``。
        ``blocked=True``（模型拒答或 Critic 升级）的簇写 :data:`ANNOT_LOW_CONFIDENCE`
        而不是具体类型——落到文件里的注释不能比页面上的结论更肯定。没解读过的簇写
        :data:`ANNOT_UNANNOTATED`，且每次调用都整列重写，所以换分辨率、清缓存之后
        不会留下上一轮的旧标签。
        """
        a = self.adata
        obs = a.obs[self.cluster_key].astype(str)
        lab = np.full(len(obs), ANNOT_UNANNOTATED, dtype=object)
        conf = np.full(len(obs), np.nan, dtype=float)
        src = np.full(len(obs), "", dtype=object)
        het = np.full(len(obs), "", dtype=object)
        labels: dict[str, str] = {}
        n_cells = 0
        for cl, row in sorted(rows.items(), key=lambda kv: str(kv[0])):
            idx = np.flatnonzero((obs == str(cl)).to_numpy())
            if idx.size == 0:
                continue
            blocked = bool(row.get("blocked"))
            name = ANNOT_LOW_CONFIDENCE if blocked else str(row.get("type") or "")
            name = name or ANNOT_UNANNOTATED
            try:
                c = float(row.get("conf"))
            except (TypeError, ValueError):
                c = float("nan")
            lab[idx] = name
            conf[idx] = c
            src[idx] = str(row.get("source") or "")
            het[idx] = str(row.get("het") or "")
            labels[str(cl)] = name
            n_cells += int(idx.size)

        for col, arr in zip(ANNOTATE_COLS, (lab, conf, src, het)):
            a.obs[col] = arr
        if "cluster" in self.clusters.columns:
            cl_str = self.clusters["cluster"].astype(str).tolist()

            def cell(c: str, key: str, default: Any) -> Any:
                if key == "type":
                    return labels.get(c, ANNOT_UNANNOTATED)
                val = (rows.get(c) or {}).get(key)
                if key == "conf":
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        return float("nan")
                return default if val is None else str(val)

            for col, key, default in ((ANNOTATE_COLS[0], "type", ""),
                                      (ANNOTATE_COLS[1], "conf", float("nan")),
                                      (ANNOTATE_COLS[2], "source", ""),
                                      (ANNOTATE_COLS[3], "het", "")):
                self.clusters[col] = [cell(c, key, default) for c in cl_str]
        a.uns[ANNOTATE_KEY] = {
            "tool": "scCopilot",
            "written_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "cluster_key": self.cluster_key,
            "rule": ("blocked(拒答/Critic 升级) → low_confidence；未解读 → unannotated；"
                     "其余用模型给出的类型名"),
            "n_clusters_annotated": len(labels),
            "labels": labels,
        }
        return {"n_clusters_annotated": len(labels), "n_cells_labeled": int(n_cells),
                "n_cells_total": int(len(obs)), "columns": list(ANNOTATE_COLS)}

    def save(self, out_dir: str | Path, stem: str = "run") -> dict[str, str]:
        """落盘：处理后的 h5ad + marker 表 + 簇表，便于复现与二次分析。"""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = {
            "h5ad": str(out / f"{stem}.h5ad"),
            "markers": str(out / f"{stem}_markers.csv"),
            "clusters": str(out / f"{stem}_clusters.csv"),
            "config": str(out / f"{stem}_config.json"),
        }
        self.adata.write_h5ad(paths["h5ad"], compression="gzip")
        self.markers.to_csv(paths["markers"], index=False)
        self.clusters.to_csv(paths["clusters"], index=False)
        Path(paths["config"]).write_text(
            pd.Series(self.config).to_json(force_ascii=False, indent=2), encoding="utf-8"
        )
        return paths


# --------------------------------------------------------------------------- #
# 数据读取
# --------------------------------------------------------------------------- #
def _is_sparse(x: Any) -> bool:
    return sp.issparse(x)


def _to_float32_csr(x: Any) -> sp.csr_matrix:
    """统一成 float32 CSR：整条流水线最重要的省内存手段。"""
    x = sp.csr_matrix(x) if not _is_sparse(x) else x.tocsr()
    return x if x.dtype == np.float32 else x.astype(np.float32)


def _dense_col(x: Any, rows: np.ndarray, col: int) -> np.ndarray:
    """取稀疏矩阵子集的单列并转一维，避免整块 todense()。"""
    sub = x[rows][:, [col]]
    return np.asarray(sub.todense() if _is_sparse(sub) else sub).ravel()


def looks_like_counts(x: Any, sample: int = 5000, max_rows: int = 64) -> bool:
    """抽样判断矩阵是否为原始 counts（非负 + 近似整数）。

    用户上传的 h5ad 常已 normalize+log1p 过，再归一化一次会毁掉数据，
    所以必须先探测再决定是否跑 QC/normalize。
    """
    n_obs = int(x.shape[0])
    if n_obs == 0:
        return False
    # 必须在全矩阵范围内均匀取行：10x raw 文件开头常有整段空液滴（CSR 里一行都不存），
    # 只看前 20 行会把这类 counts 误判成"已归一化"，进而跳过 normalize_total/log1p。
    idx = np.unique(np.linspace(0, n_obs - 1, min(n_obs, max_rows)).astype(int))
    if _is_sparse(x):
        data = np.asarray(x[idx].data, dtype=np.float64)[:sample]
    else:
        flat = np.asarray(x[idx], dtype=np.float64).ravel()
        data = flat[flat != 0][:sample]
    if data.size == 0:
        return False
    if np.any(data < 0):
        return False
    return bool(np.allclose(data, np.round(data), atol=1e-6))


def _prefer_gene_symbols(adata: ad.AnnData, notes: list[str]) -> ad.AnnData:
    """Ensembl ID 对 AI 与人类都不友好：若 var 里带 symbol 列则切换为 symbol。"""
    is_ensembl = np.fromiter(
        (_ENSEMBL_RE.match(str(s)) is not None for s in adata.var_names), bool, adata.n_vars
    )
    ensembl_frac = float(is_ensembl.mean())
    if ensembl_frac < 0.5:
        return adata
    for cand in ("gene_symbols", "symbols", "symbol", "feature_name", "external_gene_name"):
        if cand in adata.var.columns:
            vals = adata.var[cand].astype("string")
            if float(vals.notna().mean()) > 0.8:
                adata.var["ensembl_id"] = adata.var_names.astype(str)
                adata.var_names = pd.Index(vals.astype(str))
                adata.var_names_make_unique()
                notes.append(f"var_names 已由 Ensembl ID 切换为 '{cand}'（AI 解读更准）。")
                return adata
    notes.append("检测到 Ensembl ID 命名且无 symbol 注释：marker 可读性与 AI 解读质量会下降。")
    return adata


def load_h5ad(path: str | Path) -> tuple[ad.AnnData, list[str]]:
    """读取 .h5ad 并做健壮性预处理，返回 (adata, 提示信息列表)。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")
    notes: list[str] = []

    adata = sc.read_h5ad(str(path))
    if adata.isbacked:
        adata = adata.to_memory()
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("文件里没有细胞或没有基因，请检查 .h5ad。")

    adata.X = _to_float32_csr(adata.X)
    if adata.var_names.has_duplicates:
        before = adata.n_vars
        adata.var_names_make_unique()
        notes.append(f"基因名有重复，已自动去重（{before} → {adata.n_vars}）。")
    if adata.obs_names.has_duplicates:
        adata.obs_names_make_unique()

    adata = _prefer_gene_symbols(adata, notes)

    if not looks_like_counts(adata.X):
        notes.append(
            "X 看起来不是 raw counts（已归一化/取过 log）。QC 只做细胞层面过滤，"
            "并跳过 normalize+log1p，以免二次归一化扭曲表达量。"
        )
    if adata.n_obs < 200:
        notes.append(f"细胞数仅 {adata.n_obs}，统计功效不足，聚类与 marker 结果仅供参考。")

    return adata, notes


# --------------------------------------------------------------------------- #
# QC
# --------------------------------------------------------------------------- #
def flag_gene_classes(adata: ad.AnnData) -> ad.AnnData:
    """给基因打三类标签：线粒体 / 核糖体 / 名字未知，供 QC 与 AI 证据层复用。"""
    vn = adata.var_names.astype(str).to_series()
    adata.var["is_mito"] = vn.str.match(_MT_RE).to_numpy()
    adata.var["is_ribo"] = vn.str.match(_RIBO_RE).to_numpy()
    # 读不懂的名字 = 不符合 symbol 规范，或命中 LOC/MIR/ENSG 等占位名前缀
    unreadable = vn.str.match(_NOVEL_RE) | ~vn.str.match(_SYMBOL_RE)
    adata.var["is_novel"] = unreadable.to_numpy()
    return adata


def _ensure_gene_flags(adata: ad.AnnData) -> ad.AnnData:
    """幂等补齐 is_mito / is_ribo / is_novel 三列。

    这几个 QC 函数既可以被 run_pipeline 串起来调用，也可以被单独调用（例如前端只做
    QC 复算），所以每个入口都必须自己保证依赖存在，而不是假设上游跑过。
    """
    if not {"is_mito", "is_ribo", "is_novel"}.issubset(adata.var.columns):
        flag_gene_classes(adata)
    return adata


def qc_metrics(adata: ad.AnnData) -> ad.AnnData:
    """计算 per-cell QC 指标（写入 obs）。语义上应作用于 raw counts。"""
    _ensure_gene_flags(adata)
    X = adata.X
    if _is_sparse(X):
        counts = np.asarray(X.sum(axis=1)).ravel()
        detected = np.asarray((X > 0).sum(axis=1)).ravel()
    else:
        counts = np.asarray(X).sum(axis=1)
        detected = np.asarray((X > 0).sum(axis=1))

    adata.obs["n_counts"] = counts.astype(np.float32)
    adata.obs["total_counts"] = adata.obs["n_counts"]
    adata.obs["n_genes"] = detected.astype(np.int32)
    adata.obs["n_genes_by_counts"] = adata.obs["n_genes"]

    for col, flag in (("pct_mito", "is_mito"), ("pct_ribo", "is_ribo")):
        mask = adata.var[flag].to_numpy() if flag in adata.var else np.zeros(adata.n_vars, bool)
        if mask.any():
            part = np.asarray(X[:, mask].sum(axis=1)).ravel() if _is_sparse(X) else np.asarray(X)[:, mask].sum(axis=1)
            adata.obs[col] = np.divide(100.0 * part, np.maximum(counts, 1e-9)).astype(np.float32)
        else:
            adata.obs[col] = np.float32(0.0)
    return adata


def apply_qc(adata: ad.AnnData, cfg: PipelineConfig) -> tuple[ad.AnnData, dict[str, Any]]:
    """按阈值过滤基因与细胞，返回 (过滤后的 adata, 漏斗统计)。

    顺序：先收窄基因矩阵（后续所有按列求和都更快），再过滤低质量细胞。
    这里用布尔索引返回新对象，而不是调 AnnData 的私有 ``_inplace_subset_obs``。
    """
    fun: dict[str, Any] = {"n_cells_start": int(adata.n_obs), "n_genes_start": int(adata.n_vars)}
    is_counts = looks_like_counts(adata.X)

    # 基因层面
    sc.pp.filter_genes(adata, min_cells=cfg.min_cells)
    sc.pp.filter_genes(adata, min_counts=1)
    fun["n_genes_after_gene_filter"] = int(adata.n_vars)
    _ensure_gene_flags(adata)   # 收窄矩阵后再打标签，避免给已滤掉的基因白算

    # 细胞层面：counts / genes
    if is_counts:
        sc.pp.filter_cells(adata, min_counts=cfg.min_counts)
        sc.pp.filter_cells(adata, min_genes=cfg.min_genes)
        if cfg.max_genes:
            sc.pp.filter_cells(adata, max_genes=cfg.max_genes)
        if cfg.max_counts:
            sc.pp.filter_cells(adata, max_counts=cfg.max_counts)
    else:
        # 已归一化的数据 counts 语义不成立，只保守地按检出基因数过滤
        sc.pp.filter_cells(adata, min_genes=max(10, cfg.min_genes // 10))
    fun["n_cells_after_hard_filter"] = int(adata.n_obs)

    qc_metrics(adata)

    # 线粒体占比（只有识别到 MT 基因时才有意义）
    fun["n_mito_genes"] = int(adata.var["is_mito"].sum())
    if cfg.max_pct_mito is not None and fun["n_mito_genes"] > 0:
        keep = (adata.obs["pct_mito"] < cfg.max_pct_mito).to_numpy()
        # .copy() 不是多余的一步：布尔索引返回的是视图，会一直吊着整张原始矩阵；
        # 复制之后原对象可以释放，obs 也不再借用父对象的行（否则后续写 obs 会触发
        # anndata 的 ImplicitModificationWarning）。
        adata = adata[keep].copy()
    fun["n_cells_after_mito_filter"] = int(adata.n_obs)

    if cfg.filter_doublets:
        n_doublet = _scrublet(adata)
        fun["n_doublets_predicted"] = n_doublet
        if n_doublet >= 0 and "predicted_doublet" in adata.obs:
            adata = adata[(~adata.obs["predicted_doublet"].astype(bool)).to_numpy()].copy()

    # 实测 2~3 个细胞时 UMAP 的谱分解会在 scipy 里抛 "eigh ... k >= N"，
    # 用户看不懂；在还能解释的这一步就拦下来，并给出可操作的补救办法。
    if adata.n_obs < 4:
        raise ValueError(
            f"QC 后只剩 {adata.n_obs} 个细胞（少于 4 个，无法降维），"
            "请放宽 min_genes / min_counts / max_pct_mito 阈值。"
        )

    fun.update(n_cells_final=int(adata.n_obs), n_genes_final=int(adata.n_vars))
    fun["cells_removed_pct"] = round(
        100.0 * (fun["n_cells_start"] - fun["n_cells_final"]) / max(fun["n_cells_start"], 1), 2
    )
    if fun["n_mito_genes"] == 0:
        fun.setdefault("note", "未识别到线粒体基因，pct_mito 过滤已跳过。")
    # 留给下游：_cluster_summary 据此判断 n_counts 是不是真的 UMI
    adata.uns[_CFG_KEY] = {**adata.uns.get(_CFG_KEY, {}), "x_is_counts": bool(is_counts)}
    return adata, fun


def _scrublet(adata: ad.AnnData) -> int:
    """双细胞检测；失败（缺依赖/数据过小）时只记警告，不中断流水线。"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.pp.scrublet(adata, expected_doublet_rate=0.06, verbose=False)
        return int(adata.obs["predicted_doublet"].sum())
    except Exception as exc:  # pragma: no cover - 依赖缺失路径
        adata.uns.setdefault(_WARN_KEY, []).append(f"双细胞检测已跳过：{exc}")
        return -1


# --------------------------------------------------------------------------- #
# 归一化 / 降维 / 聚类
# --------------------------------------------------------------------------- #
def normalize(adata: ad.AnnData, cfg: PipelineConfig, notes: list[str]) -> None:
    """CPM 归一化 + log1p；counts 备份到 layers['counts']。"""
    if not looks_like_counts(adata.X):
        notes.append("X 非 raw counts，已跳过 normalize_total/log1p（直接复用输入表达量）。")
        if cfg.keep_counts_layer:
            adata.layers["counts"] = adata.X.copy()
        return

    qc_metrics(adata)  # 归一化前先留一份 counts 口径的指标
    if cfg.keep_counts_layer:
        adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=cfg.target_sum)
    sc.pp.log1p(adata)  # 稀疏原地运算，不会稠密化


def embed(adata: ad.AnnData, cfg: PipelineConfig, notes: list[str]) -> None:
    """HVG → PCA → 邻居图。用 mask_var 做 HVG 掩码 PCA，保留完整基因集。"""
    n_top = int(max(100, min(cfg.n_top_genes, adata.n_vars - 1)))
    flavor = cfg.hvg_flavor
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor=flavor, inplace=True)
    except Exception as exc:
        # seurat_v3 需要 counts、cell_ranger 假定特定分布；退回最稳的离散度法
        notes.append(f"HVG flavor='{flavor}' 失败（{exc}），已回退 flavor='scanpy'。")
        sc.pp.highly_variable_genes(adata, min_disp=0.5, flavor="scanpy", inplace=True)

    hvg = adata.var["highly_variable"].to_numpy(dtype=bool)
    if int(hvg.sum()) < 50:
        notes.append(f"高变基因仅 {int(hvg.sum())} 个，降维分辨率受限（改用全基因集）。")
        hvg = np.ones(adata.n_vars, dtype=bool)

    # n_comps 必须严格小于样本数与特征数，否则 sklearn 直接报错。
    # 下限只能是 1：写成 max(2, …) 会在"细胞数 ≤ 2"时把约束又顶回去，等于没防住。
    n_pcs = int(max(1, min(cfg.n_pcs, int(hvg.sum()) - 1, adata.n_obs - 1)))
    sc.pp.pca(
        adata,
        n_comps=n_pcs,
        mask_var=hvg,           # 关键：不删基因也能只对 HVG 做 PCA
        zero_center=True,
        random_state=cfg.random_state,
        dtype="float32",
    )
    adata.obsm["X_pca"] = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    adata.uns[_CFG_KEY] = {**adata.uns.get(_CFG_KEY, {}), "n_pcs_used": n_pcs}

    n_neighbors = int(max(2, min(cfg.n_neighbors, adata.n_obs - 1)))
    sc.pp.neighbors(
        adata, n_neighbors=n_neighbors, n_pcs=min(n_pcs, 30),
        random_state=cfg.random_state,
    )


def cluster(adata: ad.AnnData, cfg: PipelineConfig, key: str = "leiden") -> None:
    """Leiden 聚类 + UMAP。屏蔽两个后端（igraph / leidenalg）的参数差异。"""
    recluster_only(
        adata,
        resolution=cfg.resolution,
        key=key,
        flavor=cfg.resolved_leiden_flavor(),
        random_state=cfg.random_state,
    )
    sc.tl.umap(adata, random_state=cfg.random_state)
    adata.obsm["X_umap"] = np.asarray(adata.obsm["X_umap"], dtype=np.float32)


def recluster_only(
    adata: ad.AnnData,
    resolution: float,
    *,
    key: str = "leiden",
    flavor: str = "auto",
    random_state: int = 0,
) -> None:
    """在已有邻居图上重跑 Leiden（不重算 PCA/UMAP）。"""
    if "neighbors" not in adata.uns:
        raise ValueError("adata 尚未计算邻居图，请先执行 run_pipeline。")
    flavor = PipelineConfig(leiden_flavor=flavor).resolved_leiden_flavor()
    kw: dict[str, Any] = dict(
        resolution=float(resolution), random_state=random_state,
        key_added=key, flavor=flavor, use_weights=True,
    )
    if flavor == "igraph":
        # igraph 实现只支持无向图；(n_iterations=2, directed=False) 是官方给的"未来默认"组合
        kw.update(n_iterations=2, directed=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 屏蔽后端即将切换的 FutureWarning 噪声
        try:
            sc.tl.leiden(adata, **kw)
        except Exception as exc:
            if flavor != "igraph":
                raise
            adata.uns.setdefault(_WARN_KEY, []).append(f"igraph Leiden 失败（{exc}），改用 leidenalg。")
            kw.update(flavor="leidenalg", n_iterations=-1)
            kw.pop("directed", None)
            sc.tl.leiden(adata, **kw)

    adata.uns[_CFG_KEY] = {
        **adata.uns.get(_CFG_KEY, {}),
        "cluster_key": key,
        "resolution": float(resolution),
        "leiden_flavor": flavor,
    }


# --------------------------------------------------------------------------- #
# Marker 与"证据"提取
# --------------------------------------------------------------------------- #
def _gene_evidence(
    adata: ad.AnnData, cluster_key: str, cluster: str, genes: Iterable[str]
) -> pd.DataFrame:
    """统计某簇内这些基因的平均表达量与检出率——防幻觉层的事实来源。

    AI 只看到基因名时，容易把噪音基因/低表达基因脑补成某个细胞类型；
    给出 mean_expr 与 pct_expressed 后，ai 层可以用硬阈值判定"证据不足"。
    """
    col_idx = {g: _var_loc(adata, g) for g in dict.fromkeys(genes)}
    col_idx = {g: j for g, j in col_idx.items() if j is not None}
    if not col_idx:
        return pd.DataFrame(columns=["gene", "mean_expr", "pct_expressed"])

    obs = adata.obs[cluster_key].astype(str)
    rows = np.flatnonzero((obs == str(cluster)).to_numpy())
    X = adata.X
    out = []
    for gene, j in col_idx.items():
        vec = _dense_col(X, rows, j) if rows.size else np.zeros(0, dtype=np.float32)
        out.append(
            {
                "gene": gene,
                "mean_expr": round(float(vec.mean()), 5) if vec.size else 0.0,  # log 归一化尺度
                "pct_expressed": round(float((vec > 0).mean()), 5) if vec.size else 0.0,
            }
        )
    return pd.DataFrame(out)


def gene_column(adata: ad.AnnData, name: str) -> tuple[str, np.ndarray] | None:
    """按基因名取逐细胞表达向量（``adata.X``，即 log(CPM+1) 口径），供前端画 feature plot。

    返回 ``(命中的写法, 向量)``：命中的写法可能是 ``CD3D`` 而用户输入的是 ``cd3d``，
    前端要把真实基因名标到色条上，不能回显用户的手输。

    大小写不敏感：人类 ``CD3D`` 与小鼠 ``Cd3d`` 都能命中同一个写法。命中 0 个或多个
    （重名列不确定取哪一列）都返回 ``None``——宁可不画，也不猜一个基因给用户看。
    """
    target = str(name).strip().upper()
    if not target:
        return None
    idx = [i for i, v in enumerate(map(str, adata.var_names)) if v.strip().upper() == target]
    if len(idx) != 1:
        return None
    col = int(idx[0])
    vec = _dense_col(adata.X, np.arange(adata.n_obs), col)
    return str(adata.var_names[col]), np.asarray(vec, dtype=np.float32)


def _var_loc(adata: ad.AnnData, gene: str) -> int | None:
    """安全取基因列下标。get_loc 对重复名会返回 slice/布尔数组，这里统一降级为 None。"""
    try:
        loc = adata.var_names.get_loc(gene)
    except KeyError:
        return None
    if isinstance(loc, (int, np.integer)):  # 唯一命中
        return int(loc)
    return None  # 重复名或掩码结果：语义不明确，宁可跳过该基因


def extract_markers(
    adata: ad.AnnData,
    cluster_key: str = "leiden",
    *,
    n_markers: int = 8,
    method: str = "wilcoxon",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """提取每簇 Top-N 差异基因 + 簇级证据汇总。返回 (markers 长表, clusters 表)。"""
    if cluster_key not in adata.obs:
        raise KeyError(f"obs 中没有聚类列 '{cluster_key}'")

    obs = adata.obs[cluster_key].astype(str)
    sizes = obs.value_counts()
    groups = sorted(map(str, sizes.index), key=lambda s: (len(s), s))
    # 只有 1 个细胞的簇没法做统计检验，scanpy 碰到就直接抛 ValueError；
    # 这里显式把它们排除在检验之外，让它们以"无 marker"的形式留在簇表里，
    # 由 evidence_ok 判定为证据不足，而不是让整个流程挂掉。
    testable = [g for g in groups if int(sizes[g]) >= 2]
    skipped = [g for g in groups if g not in set(testable)]
    if skipped:
        adata.uns.setdefault(_WARN_KEY, []).append(
            f"{len(skipped)} 个簇只有 1 个细胞，已跳过差异检验（marker 为空）："
            f"{', '.join(skipped[:5])}{' 等' if len(skipped) > 5 else ''}。"
        )
    if testable:
        sc.tl.rank_genes_groups(
            adata,
            groupby=cluster_key,
            method=method,
            n_genes=int(n_markers),
            reference="rest",        # one-vs-rest：这才是"簇特异"基因
            tie_correct=True,
            groups=testable,
            # 不传 pts：检出率我们在 _gene_evidence 里按 Top 基因精确算，省一次全基因扫描
        )

    frames: list[pd.DataFrame] = []
    for grp in testable:
        # scanpy 返回的基因列名是 'names'，统一改成语义更清晰的 'gene'
        df = sc.get.rank_genes_groups_df(adata, group=grp).rename(columns={"names": "gene"})
        df = df.sort_values("scores", ascending=False).head(n_markers).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        df.insert(0, "cluster", grp)
        # 逐簇补证据，(cluster, gene) 联合键 merge，避免同一基因在多簇时笛卡尔膨胀
        ev = _gene_evidence(adata, cluster_key, grp, df["gene"])
        df = df.merge(ev, on="gene", how="left")
        frames.append(df)
    markers = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=["cluster", "rank", "gene", "scores", "pvals", "pvals_adj",
                     "logfoldchanges", "mean_expr", "pct_expressed"]
        )
    )

    if "mean_expr" in markers:
        markers["mean_expr"] = markers["mean_expr"].fillna(0.0)
        markers["pct_expressed"] = markers["pct_expressed"].fillna(0.0)

    flags = {c: adata.var[c].to_numpy() if c in adata.var else None
             for c in ("is_mito", "is_ribo", "is_novel")}
    for col, arr in flags.items():
        if arr is None:
            flag_gene_classes(adata)
            arr = adata.var[col].to_numpy()
        locs = [_var_loc(adata, g) for g in markers["gene"]]
        markers[col] = [bool(arr[j]) if j is not None else False for j in locs]

    clusters = _cluster_summary(adata, cluster_key, markers)
    adata.uns[_CFG_KEY] = {
        **adata.uns.get(_CFG_KEY, {}),
        "cluster_key": cluster_key,
        "n_clusters": int(clusters.shape[0]),
        "n_markers": int(n_markers),
    }
    return markers, clusters


def _cluster_summary(
    adata: ad.AnnData, cluster_key: str, markers: pd.DataFrame
) -> pd.DataFrame:
    """每簇一行：规模、QC 中位数、UMAP 质心、marker 质量。质量字段是 AI 拒绝解读的依据。"""
    obs = adata.obs[cluster_key].astype(str)
    umap = adata.obsm.get("X_umap")
    # apply_qc 留下的判定；没跑过 QC 的调用（例如直接对构造好的 adata 取 marker）沿用旧口径
    counts_are_umis = bool(adata.uns.get(_CFG_KEY, {}).get("x_is_counts", True))
    rows: list[dict[str, Any]] = []

    for grp in sorted(obs.unique(), key=lambda s: (len(s), s)):
        mask = (obs == grp).to_numpy()
        size = int(mask.sum())
        mk = markers[markers["cluster"] == grp]
        weak = float((mk["pct_expressed"] < 0.05).mean()) if len(mk) else 1.0
        novel = float(mk["is_novel"].mean()) if len(mk) else 1.0
        mito = float(mk["is_mito"].mean()) if len(mk) else 0.0
        # "硬证据" marker：既有检出又有表达量
        strong = int(((mk["pct_expressed"] > 0.2) & (mk["mean_expr"] > 0.3)).sum()) if len(mk) else 0

        row: dict[str, Any] = {
            "cluster": grp,
            "n_cells": size,
            "pct_cells": round(100.0 * size / max(int(adata.n_obs), 1), 2),
            "n_markers": int(len(mk)),
            "n_strong_markers": strong,
            "weak_marker_ratio": round(weak, 3),
            "novel_marker_ratio": round(novel, 3),
            "mito_marker_ratio": round(mito, 3),
        }
        for src, dst in (("n_counts", "median_counts"), ("pct_mito", "median_pct_mito")):
            row[dst] = float(np.median(adata.obs[src].to_numpy()[mask])) if src in adata.obs and size else np.nan
        if not counts_are_umis:
            # 输入不是 counts 时 obs['n_counts'] 是各基因表达量之和，没有 UMI 语义；
            # 报成数字会让 UI 的「中UMI」列和 prompt 里的 median UMI 变成假的测序深度。
            row["median_counts"] = float("nan")
        if umap is not None and size:
            cen = np.asarray(umap, dtype=np.float64)[mask].mean(axis=0)
            row.update(umap_x=float(cen[0]), umap_y=float(cen[1]))
        rows.append(row)

    cl = pd.DataFrame(rows)
    if cl.empty:
        return cl
    # 证据是否足以让 AI 做细胞类型推断——前端与 ai 层共用同一判据
    cl["evidence_ok"] = (
        (cl["n_cells"] >= 10)
        & (cl["n_strong_markers"] >= 2)
        & (cl["novel_marker_ratio"] < 0.75)
        & (cl["mito_marker_ratio"] < 0.5)
    )
    # median_counts 可能是 NaN（非 counts 输入），NaN 参与比较恒为 False，
    # 于是"深度不足"这一半自动退出判断，只剩线粒体占比。
    cl["suspect_low_quality"] = (cl["median_pct_mito"] > 25) | (cl["median_counts"] < 500)
    return cl.sort_values("n_cells", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def run_pipeline(
    adata: ad.AnnData,
    config: PipelineConfig | None = None,
    *,
    cluster_key: str = "leiden",
    progress: ProgressFn | None = None,
) -> PipelineResult:
    """标准单细胞流程：QC → 归一化 → HVG → PCA → 邻居图 → Leiden → UMAP → Marker。

    Parameters
    ----------
    adata : AnnData
        ``load_h5ad`` 得到的对象；X 建议是 raw counts（非 counts 会自动探测并跳过归一化）。
    config : PipelineConfig, optional
        缺省使用默认阈值。
    progress : callable(str, float), optional
        阶段回调，Streamlit 用它驱动进度条；在回调里抛异常即可中断跑批。

    Returns
    -------
    PipelineResult
        含 adata / markers / clusters / qc 漏斗 / 各阶段耗时 / 告警列表。
    """
    cfg = config or PipelineConfig()
    cfg.validate()
    emit: ProgressFn = progress or (lambda *_: None)
    timings: dict[str, float] = {}
    notes: list[str] = []
    t_all = time.perf_counter()

    if adata.isbacked:
        adata = adata.to_memory()
    flag_gene_classes(adata)

    def _step(name: str, pct: float, fn: Callable[[], Any]) -> Any:
        emit(name, pct)
        t0 = time.perf_counter()
        out = fn()
        timings[name] = round(time.perf_counter() - t0, 2)
        return out

    adata, qc_fun = _step("QC 过滤", 0.10, lambda: apply_qc(adata, cfg))
    _step("归一化 log1p", 0.25, lambda: normalize(adata, cfg, notes))
    _step("高变基因 + PCA", 0.50, lambda: embed(adata, cfg, notes))
    _step("Leiden 聚类 + UMAP", 0.78, lambda: cluster(adata, cfg, key=cluster_key))
    markers, clusters = _step("差异基因排序", 0.95,
                             lambda: extract_markers(adata, cluster_key,
                                                     n_markers=cfg.n_markers,
                                                     method=cfg.marker_method))

    gc.collect()
    timings["总计"] = round(time.perf_counter() - t_all, 2)
    emit("完成", 1.0)

    notes.extend(adata.uns.pop(_WARN_KEY, []))
    return PipelineResult(
        adata=adata,
        markers=markers,
        clusters=clusters,
        qc=qc_fun,
        timings=timings,
        config=asdict(cfg),
        warnings=list(dict.fromkeys(notes)),
    )


def warmup(*, n_cells: int = 80, n_genes: int = 400, seed: int = 0) -> float:
    """预热 scanpy/numba 的即时编译，返回耗时（秒）。

    为什么要这个函数：scanpy 的 njit 内核没有开启 cache，每个进程首次调用都要现编译。
    本机实测 ``normalize_total`` 首次 10.4s、第二次 0.005s（约 2000 倍），
    ``neighbors``/``umap``/``wilcoxon`` 各有数秒编译开销，合计约一分钟。
    把这笔一次性成本放到应用启动阶段跑（前端用 ``st.cache_resource`` 调一次），
    用户点"开始分析"时就不会看到莫名其妙的长卡顿。
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(
        rng.poisson(1.2, size=(n_cells, n_genes)).astype(np.float32)
    )
    a = ad.AnnData(X=X)
    a.var_names = [f"W{i}" for i in range(n_genes)]
    a.obs_names = [f"c{i}" for i in range(n_cells)]
    cfg = PipelineConfig(
        min_genes=1, min_counts=1, min_cells=1, max_genes=None, max_counts=None,
        max_pct_mito=100.0, n_top_genes=200, n_pcs=5, n_neighbors=5,
        resolution=1.0, n_markers=3, filter_doublets=False, keep_counts_layer=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_pipeline(a, cfg)
    return round(time.perf_counter() - t0, 2)


def recluster(
    adata: ad.AnnData,
    resolution: float,
    *,
    cluster_key: str = "leiden",
    n_markers: int = 8,
    method: str = "wilcoxon",
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """只重跑 Leiden + Marker，复用已有邻居图。

    前端 Resolution 滑块走这条路径：PCA 与邻居图不变，耗时从分钟级降到秒级；
    固定 random_state 保证同分辨率下结果可复现。UMAP 坐标保持不变，
    因此簇会"在原来的图上被切开/合起来"，便于向用户解释分辨率的作用。
    """
    recluster_only(
        adata, resolution, key=cluster_key, flavor="auto", random_state=random_state
    )
    return extract_markers(adata, cluster_key, n_markers=n_markers, method=method)


# --------------------------------------------------------------------------- #
# 状态感知流水线（工业级封装）
# --------------------------------------------------------------------------- #
class StateError(Exception):
    """状态机非法转移时抛出。"""
    pass


class StateAwarePipeline:
    """面向对象的流水线封装：维护 AnnData 的状态属性，提供分步执行接口。

    状态流转：
        raw → qc_passed → embedded → clustered

    设计意图：
    - 废弃原有的单次执行函数（run_pipeline 仍保留作为内部实现）；
    - 为 AnnData 维护 self.state（如 raw, qc_passed, clustered）；
    - 提供独立方法：run_qc(), run_clustering(resolution)；
    - 如果用户在 raw 状态直接调用聚类，抛出 StateError；
    - 集成 ProvenanceTracker：每次状态流转自动记一笔；
    - 集成 kg_retriever：聚类完成后可直接调用 retrieve(marker_genes)。

    典型用法：
        pipe = StateAwarePipeline(adata)
        pipe.run_qc()
        pipe.run_clustering(resolution=1.0)
        report = pipe.provenance.export_json()
    """

    def __init__(self, adata: ad.AnnData, *, config: PipelineConfig | None = None) -> None:
        self.adata = adata.copy()  # 避免污染原始输入
        self.config = config or PipelineConfig()
        self.state = "raw"
        self.provenance = ProvenanceTracker()
        self._result: PipelineResult | None = None

    def run_qc(self) -> None:
        """执行 QC 过滤（状态：raw → qc_passed）。"""
        if self.state != "raw":
            raise StateError(f"当前状态为 {self.state}，无法执行 QC（期望 raw）。")
        t0 = time.perf_counter()
        self.adata, qc = apply_qc(self.adata, self.config)
        duration = time.perf_counter() - t0
        self.provenance.log_step(
            "apply_qc",
            self.config.__dict__,
            {"n_cells": qc["n_cells_final"], "n_genes": qc["n_genes_final"]},
            duration_s=duration,
        )
        self.state = "qc_passed"

    def run_embedding(self) -> None:
        """执行降维（HVG + PCA + 邻居图）（状态：qc_passed → embedded）。"""
        if self.state != "qc_passed":
            raise StateError(f"当前状态为 {self.state}，无法执行降维（期望 qc_passed）。")
        t0 = time.perf_counter()
        normalize(self.adata, self.config, notes=[])
        embed(self.adata, self.config, notes=[])
        duration = time.perf_counter() - t0
        self.provenance.log_step(
            "embed",
            {"n_top_genes": self.config.n_top_genes, "n_pcs": self.config.n_pcs},
            {"status": "done"},
            duration_s=duration,
        )
        self.state = "embedded"

    def run_clustering(self, resolution: float | None = None) -> PipelineResult:
        """执行聚类 + Marker 提取（状态：embedded → clustered）。

        Parameters
        ----------
        resolution : float, optional
            Leiden 分辨率；若不传则使用 self.config.resolution。

        Returns
        -------
        PipelineResult
            包含 adata / markers / clusters / qc / timings / warnings。
        """
        if self.state != "embedded":
            raise StateError(f"当前状态为 {self.state}，无法执行聚类（期望 embedded）。")
        cfg = self.config
        if resolution is not None:
            cfg = PipelineConfig(**{**asdict(cfg), "resolution": resolution})
        t0 = time.perf_counter()
        cluster(self.adata, cfg)
        markers, clusters = extract_markers(self.adata, n_markers=cfg.n_markers, method=cfg.marker_method)
        duration = time.perf_counter() - t0
        self.provenance.log_step(
            "cluster",
            {"resolution": cfg.resolution},
            {"n_clusters": int(clusters.shape[0])},
            duration_s=duration,
        )
        self.state = "clustered"
        self._result = PipelineResult(
            adata=self.adata,
            markers=markers,
            clusters=clusters,
            qc={},
            timings={"total": duration},
            config=asdict(cfg),
            warnings=[],
        )
        return self._result

    def run_full(self) -> PipelineResult:
        """一键跑完整流水线（raw → clustered）。"""
        if self.state != "raw":
            raise StateError(f"当前状态为 {self.state}，无法执行完整流水线（期望 raw）。")
        self.provenance.start_session()
        self.run_qc()
        self.run_embedding()
        return self.run_clustering()

    def get_result(self) -> PipelineResult | None:
        """返回最近一次聚类的结果（若未执行过则返回 None）。"""
        return self._result

    def retrieve_cell_types(self, cluster: str, *, top_k: int = 5) -> list[dict[str, Any]]:
        """根据某簇的 Top marker 基因，调用 kg_retriever 检索候选细胞类型。

        前提：self.state == "clustered"。
        """
        if self.state != "clustered" or self._result is None:
            raise StateError("必须先执行 run_clustering() 才能检索细胞类型。")
        genes = self._result.genes_of(cluster, top=self.config.n_markers)
        from core.kg_retriever import retrieve
        return retrieve(genes, top_k=top_k)


# 导出新符号
__all__.extend(["StateAwarePipeline", "StateError"])
