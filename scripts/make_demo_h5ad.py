# -*- coding: utf-8 -*-
"""生成一份带"已知答案"的模拟单细胞数据，用于演示与自测。

为什么不用 pbmc3k：真实数据集要联网下载，而且无法验证"AI 是否把簇认对了"。
这里用 4 类免疫细胞 + 1 类低质量簇手工构造 marker，基因名取真实 symbol，
因此可以断言：pipeline 提取的 Top 基因应出现 CD3D / MS4A1 / LYZ / GNLY，
且低质量簇必须被 evidence_ok=False 拦下（这正是 ai 层拒绝解读的输入条件）。

测序深度按 10x 真实量级设定（~3000 UMI / 1500 genes per cell），
否则默认 QC 阈值（min_counts=500 / min_genes=200）会把细胞几乎全滤掉。

用法：
    python scripts/make_demo_h5ad.py [--seed 0] [--out data/raw/demo_pbmc_sim.h5ad]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

# (细胞类型, 细胞数, 特异 marker 基因)
POPULATIONS: list[tuple[str, int, list[str]]] = [
    ("T_cell", 900, ["CD3D", "CD3E", "CD3G", "LCK", "IL7R", "TRAC", "TRBC2", "CD2"]),
    ("B_cell", 450, ["MS4A1", "CD79A", "CD79B", "BANK1", "TNFRSF13C", "IGKC", "CD19"]),
    ("Monocyte", 500, ["LYZ", "S100A8", "S100A9", "CD14", "FCN1", "VCAN", "CSF3R"]),
    ("NK", 250, ["GNLY", "NKG7", "KLRD1", "PRF1", "GZMB", "KLRF1", "KIR2DL3"]),
    # 低质量簇：无特异基因 + 线粒体偏高，期望 evidence_ok=False
    ("LowQuality", 140, []),
]

# 公共高表达基因：真实数据里到处都有，不该成为某个簇的 marker
UBIQUITOUS = ["B2M", "ACTB", "GAPDH", "EEF1A1", "FTL", "FTH1", "MALAT1", "PGBP1", "TMSB4X", "TMSB10"]
RIBO = ["RPL13A", "RPS6", "RPS27", "RPL23", "MRPS18B", "RPLP0", "RPS19"]
MT_GENES = ["MT-CO1", "MT-CO2", "MT-ND4", "MT-CYB", "MT-ATP6", "MT-CO3"]
# 真实数据里大量存在的"读不懂"的名字：pipeline 的 is_novel 判定就是为它们准备的
JUNK_PREFIXES = ("LOC", "LINC", "MIR", "AC", "ENSG")


def _junk_genes(n: int, rng: np.random.Generator) -> list[str]:
    """构造 LOC/LINC/MIR/AC/ENSG 风格的无意义可注释名。"""
    out = []
    for i in range(n):
        p = JUNK_PREFIXES[i % len(JUNK_PREFIXES)]
        if p == "ENSG":
            out.append(f"ENSG{int(rng.integers(10**8, 10**9 - 1))}")
        elif p == "AC":
            out.append(f"AC{int(rng.integers(10**5, 10**6))}.{int(rng.integers(1, 9))}")
        else:
            out.append(f"{p}{int(rng.integers(1000, 999999))}")
    return out


def build(seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    markers = [g for _, _, mk in POPULATIONS for g in mk]
    genes = list(dict.fromkeys(UBIQUITOUS + RIBO + MT_GENES + markers + _junk_genes(2400, rng)))
    gidx = {g: i for i, g in enumerate(genes)}
    n_genes = len(genes)

    base = rng.gamma(1.5, 0.9, size=n_genes)          # 基因平均丰度，目标 ~3000 UMI/cell
    mt_cols = [gidx[g] for g in MT_GENES if g in gidx]

    rows, labels = [], []
    for name, n, mk in POPULATIONS:
        lam = np.tile(base, (n, 1))
        if mk:
            cols = [gidx[g] for g in mk]
            lam = lam.copy()
            lam[:, cols] += rng.uniform(2.5, 5.5, size=(n, len(cols)))  # 簇特异高表达
        if name == "LowQuality":
            # 捕获量低 + 线粒体占比拉到 ~15%：低于 max_pct_mito=20 所以能活过 QC，
            # 但应当被 marker 证据层识别为"不值得让 AI 解读"
            lam = lam * 0.35
            lam[:, mt_cols] += 30.0
        rows.append(rng.poisson(lam).astype(np.float32))
        labels += [name] * n

    X = sp.csr_matrix(np.vstack(rows))
    adata = ad.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(X.shape[0])]
    adata.var_names = [str(g) for g in genes]
    adata.obs["cell_type_true"] = np.array(labels)   # 保留真值，用于事后评估纯度
    return adata


def main() -> None:
    # Windows 控制台默认 GBK，中文与符号输出会抛 UnicodeEncodeError，统一切到 UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="generate a labelled demo .h5ad for scCopilot")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "data" / "raw" / "demo_pbmc_sim.h5ad"),
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    adata = build(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out, compression="gzip")

    counts = np.asarray(adata.X.sum(axis=1)).ravel()
    detected = np.asarray((adata.X > 0).sum(axis=1)).ravel()
    print(f"[ok] 已生成模拟数据：{out}")
    print(f"     {adata.n_obs} cells x {adata.n_vars} genes | "
          f"median {np.median(counts):.0f} UMI, {np.median(detected):.0f} genes/cell")
    print(adata.obs["cell_type_true"].value_counts().to_string())


if __name__ == "__main__":
    main()
