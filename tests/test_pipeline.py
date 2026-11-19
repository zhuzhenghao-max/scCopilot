# -*- coding: utf-8 -*-
"""core/pipeline.py 的回归测试。

两种跑法都支持（不依赖 pytest 是否安装）：
    python -m pytest tests/test_pipeline.py -q      # 若已装 pytest
    python tests/test_pipeline.py                   # 内置迷你 runner

注意：首次调用 scanpy 会触发 numba JIT 编译（本机约 60s），所以全套测试偏慢；
真实项目里这笔成本由 app 启动时的 warmup 承担。
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import anndata as ad          # noqa: E402
import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402
import scipy.sparse as sp     # noqa: E402

from core import pipeline as pl  # noqa: E402
from make_demo_h5ad import POPULATIONS, build  # noqa: E402

warnings.simplefilter("ignore")


# --------------------------------------------------------------------------- #
# 共享数据（只跑一次流水线，避免重复付 JIT 成本）
# --------------------------------------------------------------------------- #
_CACHE: dict[str, object] = {}


def full_result() -> pl.PipelineResult:
    if "res" not in _CACHE:
        adata = build(seed=0)
        _CACHE["res"] = pl.run_pipeline(adata, pl.PipelineConfig(resolution=1.0))
    return _CACHE["res"]  # type: ignore[return-value]


def tiny_adata(n_obs: int = 120, n_vars: int = 400, seed: int = 1) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    a = ad.AnnData(X=sp.csr_matrix(rng.poisson(1.5, (n_obs, n_vars)).astype(np.float32)))
    a.var_names = [f"G{i}" for i in range(n_vars)]
    a.obs_names = [f"c{i}" for i in range(n_obs)]
    return a


# --------------------------------------------------------------------------- #
# 单元测试：输入探测与基因标注
# --------------------------------------------------------------------------- #
def test_looks_like_counts():
    a = tiny_adata()
    assert pl.looks_like_counts(a.X) is True, "整数计数应判定为 counts"

    # 第一次：counts → 应做归一化；第二次：已是 log 值 → 应跳过，绝不二次归一化
    pl.normalize(a, pl.PipelineConfig(keep_counts_layer=False), notes1 := [])
    assert not any("跳过" in n for n in notes1), f"counts 输入不应跳过归一化：{notes1}"
    pl.normalize(a, pl.PipelineConfig(keep_counts_layer=False), notes2 := [])
    assert any("跳过" in n for n in notes2), f"log 归一化后应跳过：{notes2}"
    assert pl.looks_like_counts(a.X) is False

    neg = a.X.copy()
    neg.data = neg.data - 1.0          # 出现负值（如已 scale 的数据）
    assert pl.looks_like_counts(neg) is False


def test_gene_class_flags():
    a = tiny_adata(n_vars=11)
    a.var_names = ["CD3D", "MT-CO1", "RPL13A", "LOC100130427", "LINC00909",
                   "ENSG00000258947", "MS4A1", "mt-ND4", "RPLP0", "RPSA", "MRPL12"]
    pl.flag_gene_classes(a)
    assert a.var["is_mito"].tolist() == [False, True, False, False, False, False, False, True,
                                         False, False, False]
    # RPLP0 / RPSA 与 RPL13A 同类；MRPL*（线粒体核糖体）不在这一档，维持原口径
    assert a.var["is_ribo"].tolist() == [False, False, True, False, False, False, False, False,
                                         True, True, False]
    assert a.var["is_novel"].tolist() == [False, False, False, True, True, True, False, False,
                                         False, False, False]


def test_gene_column_returns_name_then_vector():
    """前端的契约是 gene=(名字, 向量)，色条标题用第 0 项、着色用第 1 项。
    顺序反了不会报错，只会在画图时 float('CD3D') 崩掉——单测里手搓元组测不出来，
    所以直接在 core 的返回值上钉住顺序。"""
    a = tiny_adata(n_obs=25, n_vars=4)
    a.var_names = ["CD3D", "LYZ", "GNLY", "MS4A1"]
    out = pl.gene_column(a, "  cd3d ")
    assert out is not None
    name, vec = out
    assert name == "CD3D", name
    assert len(vec) == a.n_obs and vec.dtype == np.float32, (len(vec), vec.dtype)
    assert np.allclose(vec, np.asarray(a.X.toarray()[:, 0], dtype=np.float32))

    dup = tiny_adata(n_obs=6, n_vars=3)
    dup.var_names = ["CD3D", "cd3d", "LYZ"]
    assert pl.gene_column(dup, "CD3D") is None, "重名列不能猜取哪一列"
    assert pl.gene_column(a, "") is None and pl.gene_column(a, "NOPE") is None


def test_ensembl_switches_to_symbols():
    a = tiny_adata(n_obs=30, n_vars=4)
    a.var_names = [f"ENSG{10**8 + i}" for i in range(4)]
    a.var["gene_symbols"] = ["CD3D", "MS4A1", "LYZ", "GNLY"]
    notes: list[str] = []
    out = pl._prefer_gene_symbols(a, notes)
    assert out.var_names.tolist() == ["CD3D", "MS4A1", "LYZ", "GNLY"]
    assert "ensembl_id" in out.var and any("switch" in n.lower() or "symbol" in n for n in notes)


def test_mouse_ensembl_switches_to_symbols_too():
    """Ensembl ID 的物种前缀不止 ENSG：小鼠是 ENSMUSG。只认人类前缀时，
    读不懂的占位名会被当成合法 symbol——既不切 symbol，也不计入未注释比例。"""
    a = tiny_adata(n_obs=30, n_vars=4)
    a.var_names = [f"ENSMUSG{10**8 + i}" for i in range(4)]
    a.var["gene_symbols"] = ["Cd3d", "Ms4a1", "Lyz", "Gnly"]
    notes: list[str] = []
    out = pl._prefer_gene_symbols(a, notes)
    assert out.var_names.tolist() == ["Cd3d", "Ms4a1", "Lyz", "Gnly"], out.var_names.tolist()
    assert any("symbol" in n for n in notes), notes

    b = tiny_adata(n_obs=6, n_vars=3)
    b.var_names = ["ENSMUSG00000100000", "CD3D", "MT-CO1"]
    pl.flag_gene_classes(b)
    assert b.var["is_novel"].tolist() == [True, False, False], b.var["is_novel"].tolist()


def test_counts_detected_despite_leading_empty_droplets():
    """raw 10x 文件开头常排着一整段空液滴（CSR 里整行不存任何元素）。
    抽样只看前几行会把这类 counts 误判成"已归一化"，于是跳过 normalize_total/log1p，
    后面 PCA/HVG 全在原始计数上跑——结果错但一声不吭。"""
    X = sp.vstack([sp.csr_matrix((25, 50), dtype=np.float32),
                   sp.csr_matrix(np.random.default_rng(0).poisson(3.0, (60, 50)).astype(np.float32))
                   ]).tocsr()
    a = ad.AnnData(X=X)
    a.var_names = [f"G{i}" for i in range(50)]
    a.obs_names = [f"c{i}" for i in range(X.shape[0])]
    assert a.X[:20].nnz == 0, "构造样本的前 20 行必须真的是空行，否则这条测试测不到东西"
    assert pl.looks_like_counts(a.X) is True

    notes: list[str] = []
    pl.normalize(a, pl.PipelineConfig(keep_counts_layer=False), notes)
    assert not any("跳过" in n for n in notes), f"counts 输入不该跳过归一化：{notes}"


# --------------------------------------------------------------------------- #
# QC 行为：必须真的能滤掉细胞
# --------------------------------------------------------------------------- #
def test_qc_removes_bad_cells():
    n_bg, n_mt, n_normal = 294, 6, 20
    mt = ["MT-CO1", "MT-CO2", "MT-ND4", "MT-CYB", "MT-ATP6", "MT-CO3"]
    rng = np.random.default_rng(7)
    X = rng.poisson(1.5, size=(60, n_bg + n_mt)).astype(np.float32)
    X[:20] = 0.0                      # 空液滴：只有 3 个基因有计数
    X[:20, :3] = 1.0
    X[20:40, :n_bg] = 0.0             # 线粒体爆表：只有 MT 基因有计数
    X[20:40, n_bg:] = 400.0
    a = ad.AnnData(X=sp.csr_matrix(X))
    a.var_names = [f"G{i}" for i in range(n_bg)] + mt
    a.obs_names = [f"c{i}" for i in range(60)]

    out, fun = pl.apply_qc(a, pl.PipelineConfig(min_genes=50, min_counts=50, max_pct_mito=20.0))
    assert fun["n_cells_final"] <= n_normal + 2, (
        f"空液滴与高线粒体细胞应被滤除，实际剩 {fun['n_cells_final']}"
    )
    assert fun["n_cells_start"] == 60
    assert (out.obs["pct_mito"] < 20.0).all()
    assert set(pl.QC_MIN_FIELDS).issubset(out.obs.columns)


def test_qc_raises_when_too_strict():
    a = tiny_adata(n_obs=40)
    try:
        pl.apply_qc(a, pl.PipelineConfig(min_genes=99999, min_counts=1, max_pct_mito=100.0))
    except ValueError as exc:
        assert "QC" in str(exc)
    else:
        raise AssertionError("过严阈值应抛出 ValueError 而不是静默返回空数据")


# --------------------------------------------------------------------------- #
# 端到端：结果必须与"已知答案"一致
# --------------------------------------------------------------------------- #
def test_pipeline_recovers_known_cell_types():
    res = full_result()
    assert res.qc["n_cells_final"] > 2000, f"QC 不应滤掉大部分正常细胞：{res.qc}"
    assert res.adata.n_obs == res.qc["n_cells_final"]

    a = res.adata
    ct = pd.crosstab(a.obs["leiden"].astype(str), a.obs["cell_type_true"])
    purity = ct.max(axis=1).sum() / ct.values.sum()
    assert purity > 0.9, f"聚类纯度过低：{purity:.3f}"

    # 每个真实群体应至少成为某个簇的多数成员
    covered = {g for g in ct.columns if (ct[g] > 0).any()}
    for name, _, _ in POPULATIONS:
        assert name in covered, f"没有任何簇以 {name} 为主"


def test_markers_match_ground_truth():
    res = full_result()
    ct = pd.crosstab(res.adata.obs["leiden"].astype(str), res.adata.obs["cell_type_true"])
    majority = ct.idxmax(axis=1)  # 每个簇的真实身份
    expects = {
        "T_cell": {"CD3D", "CD3E", "CD3G", "LCK", "TRAC"},
        "B_cell": {"MS4A1", "CD79A", "CD79B", "BANK1"},
        "Monocyte": {"LYZ", "S100A8", "S100A9", "CD14", "FCN1"},
        "NK": {"GNLY", "NKG7", "KLRD1", "PRF1"},
    }
    hits = 0
    for cluster, truth in majority.items():
        genes = set(res.genes_of(cluster, 8))
        if truth in expects and genes & expects[truth]:
            hits += 1
    assert hits >= 4, f"Top8 marker 命中真实身份基因（期望>=4，实际{hits}）"


def test_low_quality_cluster_is_not_trusted():
    """AI 防幻觉的前提：核心层必须把"证据不足"算成结构化字段。"""
    res = full_result()
    ct = pd.crosstab(res.adata.obs["leiden"].astype(str), res.adata.obs["cell_type_true"])
    lq_clusters = [c for c in ct.index if ct.loc[c].idxmax() == "LowQuality"]
    assert lq_clusters, "数据里应有 LowQuality 主导的簇"
    row = res.clusters.set_index("cluster").loc[lq_clusters[0]]
    assert bool(row["evidence_ok"]) is False, "低质量簇必须被标记为证据不足"
    assert row["mito_marker_ratio"] >= 0.5 or row["n_strong_markers"] < 4, row
    mk = res.evidence_of(lq_clusters[0])
    assert mk["mean_expr"].notna().all() and mk["pct_expressed"].notna().all()


def test_evidence_columns_are_filled():
    res = full_result()
    need = {"cluster", "rank", "gene", "scores", "pvals", "pvals_adj",
            "logfoldchanges", "mean_expr", "pct_expressed", "is_mito", "is_ribo", "is_novel"}
    assert need.issubset(res.markers.columns), need - set(res.markers.columns)
    assert (res.markers["rank"] <= res.config["n_markers"]).all()
    # 同一基因在多簇出现时，证据必须按簇独立，而不是 merge 出笛卡尔积
    dup = res.markers[res.markers.duplicated(["cluster", "gene"], keep=False)]
    assert dup.empty, f"(cluster, gene) 不应重复，发现 {len(dup)} 行"
    assert len(res.markers) <= res.config["n_markers"] * res.clusters.shape[0]


def test_novel_names_cover_ac_and_pseudogenes():
    """AC/AF/U* 占位名与 -AS/-ps 后缀在 10x 注释表里很常见。core 漏判时「未注释比例」
    会低于 ai 层的实际判据，前端显示"证据充足"却被 AI 拒绝。"""
    names = ["AC123456.7", "AF234567.1", "U32857", "SNOR25", "HBA1-AS1",
             "NTRK2-ps1", "CD3D", "LYZ", "GNLY", "LOC100130427"]
    a = tiny_adata(n_vars=len(names))
    a.var_names = names
    pl.flag_gene_classes(a)
    assert a.var["is_novel"].tolist() == [True, True, True, True, True, True,
                                         False, False, False, True], a.var["is_novel"].tolist()
    # 与 ai 层判据同口径：core 说"读不懂"的，gene_kb 也必须说读不懂
    from ai.gene_kb import looks_unannotated
    both = list(zip(names, a.var["is_novel"].tolist(), [looks_unannotated(g) for g in names]))
    assert all(core == kb for _, core, kb in both), both


def test_singleton_cluster_is_skipped_not_fatal():
    """高分辨率会切出单细胞簇；scanpy 对它们做检验是直接抛异常的，流程必须撑住。"""
    a = tiny_adata(n_obs=80, n_vars=300)
    pl.flag_gene_classes(a)
    a.obs["leiden"] = pd.Categorical(["0"] * 40 + ["1"] * 39 + ["2"])

    markers, clusters = pl.extract_markers(a, "leiden", n_markers=5)

    assert set(clusters["cluster"]) == {"0", "1", "2"}, "簇表不能因为没做检验就丢簇"
    assert set(markers["cluster"]) == {"0", "1"}, "单细胞簇不应出现在 marker 表里"
    assert not bool(clusters.set_index("cluster").loc["2", "evidence_ok"]), "应判为证据不足"
    assert any("只有 1 个细胞" in w for w in a.uns.get(pl._WARN_KEY, [])), a.uns.get(pl._WARN_KEY)


def test_tiny_dataset_fails_with_actionable_message():
    """2~3 个细胞会在 UMAP 的谱分解里抛 scipy 技术错误，要提前拦成看得懂的话。"""
    cfg = pl.PipelineConfig(min_genes=1, min_counts=1, min_cells=1,
                            max_pct_mito=100.0, max_genes=None, n_top_genes=200)
    for n in (2, 3):
        try:
            pl.run_pipeline(tiny_adata(n_obs=n, n_vars=300), cfg)
        except ValueError as exc:
            assert "QC" in str(exc) and "4" in str(exc), exc
        else:
            raise AssertionError(f"{n} 个细胞应抛出可读的 ValueError")

    # 4 个细胞起就能跑通：PCA 维度下限被顶成 2 时曾让 sklearn 直接报错
    res = pl.run_pipeline(tiny_adata(n_obs=4, n_vars=300), cfg)
    assert len(res.clusters) >= 1


def test_median_umi_is_blanked_for_normalized_input():
    """已归一化的输入里 n_counts 是表达量之和，当成 UMI 报出去就是假的测序深度。"""
    import scanpy as sc

    a = build(seed=2)
    a = a[:800].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    res = pl.run_pipeline(a, pl.PipelineConfig(resolution=1.0))

    assert res.clusters["median_counts"].isna().all(), res.clusters["median_counts"]
    assert (res.clusters["median_pct_mito"] >= 0).all(), "占比类指标与 counts 无关，应照常给出"
    # 对照：raw counts 输入仍报得出中位 UMI
    assert full_result().clusters["median_counts"].notna().all()


# --------------------------------------------------------------------------- #
# 分辨率滑块路径
# --------------------------------------------------------------------------- #
def test_recluster_reuses_graph_and_is_deterministic():
    # 用副本测试：recluster 会改写 obs['leiden']，不能污染共享的 full_result()
    a = full_result().adata.copy()
    umap_before = np.asarray(a.obsm["X_umap"]).copy()
    n_before = a.obs["leiden"].nunique()

    mk_lo, cl_lo = pl.recluster(a, 0.3, n_markers=8)
    assert a.obs["leiden"].nunique() <= n_before, "降低分辨率应减少簇数"
    mk_hi, cl_hi = pl.recluster(a, 2.0, n_markers=8)
    assert a.obs["leiden"].nunique() > n_before, "提高分辨率应拆分出更多亚群"
    assert np.allclose(np.asarray(a.obsm["X_umap"]), umap_before), "重聚类不应改变 UMAP 坐标"
    assert set(cl_hi["cluster"]) == set(mk_hi["cluster"]), "marker 表与簇表应覆盖同一批簇"

    # 同分辨率重跑必须完全一致（可复现性）
    mk_a, cl_a = pl.recluster(a, 1.0)
    mk_b, cl_b = pl.recluster(a, 1.0)
    assert cl_a["cluster"].tolist() == cl_b["cluster"].tolist()
    assert mk_a["gene"].tolist() == mk_b["gene"].tolist()


def test_recluster_without_graph_fails_loudly():
    try:
        pl.recluster_only(tiny_adata(n_obs=20), 1.0)
    except ValueError as exc:
        assert "邻居图" in str(exc)
    else:
        raise AssertionError("缺少邻居图时应报错")


# --------------------------------------------------------------------------- #
# 配置校验 / 落盘
# --------------------------------------------------------------------------- #
def test_config_validation():
    for bad in (dict(resolution=0.0), dict(n_pcs=1), dict(n_markers=0), dict(max_pct_mito=150.0)):
        try:
            pl.PipelineConfig(**bad).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} 应被 validate() 拒绝")
    pl.PipelineConfig().validate()


def test_save_and_reload():
    """落盘可复现性：h5ad + marker 表 + 簇表 + 配置，重新读回形状一致。"""
    res = full_result()
    out = Path(__file__).resolve().parents[1] / "data" / "cache" / "pytest_out"
    paths = res.save(out, stem="pytest_smoke")
    assert Path(paths["h5ad"]).stat().st_size > 1000
    reloaded = ad.read_h5ad(paths["h5ad"])
    assert reloaded.n_obs == res.adata.n_obs
    assert "leiden" in reloaded.obs and reloaded.obs["leiden"].notna().all()
    assert Path(paths["markers"]).stat().st_size > 0
    csv = pd.read_csv(paths["clusters"])
    assert {"cluster", "evidence_ok", "n_strong_markers"}.issubset(csv.columns)
    cfg_txt = Path(paths["config"]).read_text(encoding="utf-8")
    assert '"resolution"' in cfg_txt


def test_annotate_writes_conclusions_to_obs_and_survives_save():
    """AI 结论落回：obs 四列 + 簇表 + uns 落款，被拦下的簇只能写 low_confidence。"""
    a = ad.AnnData(sp.csr_matrix(np.zeros((6, 3), dtype="float32")),
                   obs=pd.DataFrame({"leiden": ["0", "0", "0", "1", "1", "2"]},
                                    index=[str(i) for i in range(6)]))
    a.uns[pl._CFG_KEY] = {"cluster_key": "leiden"}
    res = pl.PipelineResult(
        adata=a, markers=pd.DataFrame(columns=["cluster", "rank", "gene"]),
        clusters=pd.DataFrame({"cluster": ["0", "1", "2"], "n_cells": [3, 2, 1]}),
        qc={}, timings={}, config={})

    out = res.annotate({
        "0": {"type": "T cell", "conf": 0.83, "source": "local_kb", "het": "homogeneous"},
        "1": {"type": "Mono", "conf": 0.4, "source": "llm", "het": "heterogeneous",
              "blocked": True},
    })
    assert a.obs["sccopilot_type"].tolist() == [
        "T cell", "T cell", "T cell", "low_confidence", "low_confidence", "unannotated"]
    assert a.obs["sccopilot_conf"].iloc[0] == 0.83
    assert a.obs["sccopilot_conf"].iloc[5] != a.obs["sccopilot_conf"].iloc[5]  # NaN：没解读过
    assert res.clusters["sccopilot_type"].tolist() == ["T cell", "low_confidence", "unannotated"]
    assert res.clusters["sccopilot_het"].tolist() == ["homogeneous", "heterogeneous", ""]
    assert out["n_clusters_annotated"] == 2 and out["n_cells_labeled"] == 5
    assert a.uns[pl.ANNOTATE_KEY]["labels"] == {"0": "T cell", "1": "low_confidence"}

    # 换分辨率/清缓存后重写：不能留下上一轮的旧标签
    res.annotate({})
    assert set(a.obs["sccopilot_type"]) == {"unannotated"}

    out_dir = Path(__file__).resolve().parents[1] / "data" / "cache" / "pytest_annotate"
    res.annotate({"0": {"type": "T cell", "conf": 0.83, "source": "local_kb",
                        "het": "homogeneous"}})
    paths = res.save(out_dir, stem="annotate")
    reloaded = ad.read_h5ad(paths["h5ad"])
    assert "sccopilot_type" in reloaded.obs
    assert reloaded.obs["sccopilot_type"].iloc[0] == "T cell"
    assert reloaded.uns[pl.ANNOTATE_KEY]["labels"]["0"] == "T cell"


# --------------------------------------------------------------------------- #
# 迷你 runner（无 pytest 时可 python tests/test_pipeline.py）
# --------------------------------------------------------------------------- #
def _main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"[PASS] {fn.__name__:<45s} {time.perf_counter() - t0:6.2f}s")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"[FAIL] {fn.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
