# -*- coding: utf-8 -*-
"""Tests for core/review.py — 人工复核记录。

复核的价值全在「口径不许撒谎」：日志只追加、后写生效但历史可查、改动率的分母是
已复核簇数、首稿列不被终稿覆盖、未复核的簇终稿必须留空。这里逐条钉住。
"""

import json
import sys
import pathlib
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import anndata as ad
import numpy as np
import pandas as pd

from core import review


def _result():
    obs = pd.DataFrame({"leiden": ["0", "0", "1", "1", "2", "2"]},
                       index=[f"cell{i}" for i in range(6)])
    obs["sccopilot_type"] = ["CD4+ T 细胞", "CD4+ T 细胞", "B 细胞", "B 细胞",
                             "low_confidence", "low_confidence"]
    a = ad.AnnData(X=np.zeros((6, 3), dtype=np.float32), obs=obs)
    clusters = pd.DataFrame({"cluster": ["0", "1", "2"], "n_cells": [2, 2, 2],
                             "sccopilot_type": ["CD4+ T 细胞", "B 细胞", "low_confidence"]})
    return SimpleNamespace(adata=a, clusters=clusters, cluster_key="leiden")


def _dec(cluster="0", action="accept", **kw):
    base = dict(cluster=cluster, ai_type="CD4+ T 细胞", ai_label="CD4+ T 细胞",
                action=action, source="local_kb", confidence=0.72, run_id="run_x")
    base.update(kw)
    return review.Decision(**base)


def test_final_type_is_derived_from_action():
    assert _dec(action="accept").final_type == "CD4+ T 细胞"
    assert _dec(action="accept", ai_label="low_confidence").final_type == "low_confidence"
    assert _dec(action="reject").final_type == review.REJECTED
    assert _dec(action="edit", final_type="  NK 细胞  ").final_type == "NK 细胞"
    assert _dec().ts and "+00:00" in _dec().ts, "时间戳必须是带时区的 UTC"


def test_invalid_action_or_empty_edit_is_rejected():
    for bad in ("", "approve", "ACCEPT"):
        try:
            _dec(action=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"未知动作 {bad!r} 应被拒绝")
    try:
        _dec(action="edit", final_type="   ")
    except ValueError:
        pass
    else:
        raise AssertionError("修改类型留空应被拒绝")


def test_append_and_read_roundtrip_keeps_unicode():
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "reviews" / "run_x.jsonl"
        review.append_decision(p, _dec(note="CD3D 检出率偏低，先采纳"))
        review.append_decision(p, _dec(cluster="1", action="edit", ai_type="B 细胞",
                                       ai_label="B 细胞", final_type="浆细胞"))
        rows = review.read_decisions(p)
        assert len(rows) == 2 and rows[0]["note"] == "CD3D 检出率偏低，先采纳"
        assert rows[1]["final_type"] == "浆细胞"
        text = p.read_text(encoding="utf-8")
        assert "浆细胞" in text and "\\u" not in text, "中文不该被转成 \\uXXXX"
        assert len(text.strip().splitlines()) == 2


def test_append_only_history_survives_a_second_decision():
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "r.jsonl"
        review.append_decision(p, _dec(action="accept"))
        review.append_decision(p, _dec(action="edit", final_type="NK 细胞"))
        rows = review.read_decisions(p)
        assert [r["action"] for r in rows] == ["accept", "edit"], "历史行不能被改写"
        latest = review.latest_by_cluster(rows)
        assert latest["0"]["action"] == "edit" and latest["0"]["final_type"] == "NK 细胞"


def test_missing_log_reads_as_empty_and_corrupt_line_raises():
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "nope.jsonl"
        assert review.read_decisions(p) == []
        p.write_text('{"cluster": "0"}\n{坏行}\n', encoding="utf-8")
        try:
            review.read_decisions(p)
        except ValueError as exc:
            assert "第 2 行" in str(exc), exc
        else:
            raise AssertionError("坏行必须报错，不能静默丢弃——少一行就会算错改动率")


def test_change_rate_denominator_is_reviewed_not_all_clusters():
    log = [_dec(cluster="0", action="accept").as_dict(),
           _dec(cluster="1", action="edit", ai_type="B 细胞", ai_label="B 细胞",
                final_type="浆细胞").as_dict()]
    s = review.review_stats(log, ["0", "1", "2"])
    assert s["n_reviewed"] == 2 and s["n_clusters"] == 3
    assert abs(s["coverage"] - 2 / 3) < 1e-9
    assert s["n_changed"] == 1 and abs(s["change_rate"] - 0.5) < 1e-9
    assert s["unreviewed_clusters"] == ["2"]
    assert s["n_accept"] == 1 and s["n_edit"] == 1 and s["n_reject"] == 0


def test_edit_back_to_the_same_label_is_not_counted_as_a_change():
    log = [_dec(action="edit", final_type="CD4+ T 细胞").as_dict()]
    s = review.review_stats(log, ["0", "1", "2"])
    assert s["n_edit"] == 1 and s["n_changed"] == 0 and s["change_rate"] == 0.0


def test_reject_counts_as_a_change_and_history_is_counted_separately():
    log = [_dec(action="accept").as_dict(), _dec(action="reject").as_dict()]
    s = review.review_stats(log, ["0", "1"])
    assert s["n_decisions"] == 2 and s["n_reviewed"] == 1
    assert s["n_reject"] == 1 and s["n_changed"] == 1 and s["change_rate"] == 1.0


def test_no_reviews_means_zero_rate_not_a_fake_hundred_percent():
    s = review.review_stats([], ["0", "1", "2"])
    assert s["n_reviewed"] == 0 and s["change_rate"] == 0.0 and s["coverage"] == 0.0
    assert s["unreviewed_clusters"] == ["0", "1", "2"]


def test_apply_reviews_writes_obs_and_cluster_table():
    r = _result()
    out = review.apply_reviews(r, [_dec(cluster="0", action="accept").as_dict(),
                                   _dec(cluster="2", action="reject",
                                        ai_type="证据不足", ai_label="low_confidence").as_dict()])
    obs = r.adata.obs
    assert obs["sccopilot_review"].tolist() == ["accept", "accept", "unreviewed",
                                                "unreviewed", "reject", "reject"]
    assert obs["sccopilot_final_type"].tolist() == ["CD4+ T 细胞", "CD4+ T 细胞", "", "",
                                                    review.REJECTED, review.REJECTED]
    assert r.clusters["sccopilot_review"].tolist() == ["accept", "unreviewed", "reject"]
    assert r.clusters["sccopilot_final_type"].tolist() == ["CD4+ T 细胞", "", review.REJECTED]
    assert out["n_reviewed"] == 2 and out["n_cells_reviewed"] == 4 and out["n_cells_total"] == 6
    assert out["columns"] == list(review.REVIEW_COLS)


def test_apply_reviews_never_overwrites_the_ai_first_draft():
    r = _result()
    before = r.adata.obs["sccopilot_type"].tolist()
    before_cl = r.clusters["sccopilot_type"].tolist()
    review.apply_reviews(r, [_dec(cluster="0", action="edit", final_type="NK 细胞").as_dict()])
    assert r.adata.obs["sccopilot_type"].tolist() == before, "首稿列必须原样保留"
    assert r.clusters["sccopilot_type"].tolist() == before_cl
    assert r.adata.obs["sccopilot_final_type"].iloc[0] == "NK 细胞"


def test_apply_reviews_rewrites_columns_so_stale_labels_cannot_survive():
    r = _result()
    review.apply_reviews(r, [_dec(cluster="1", action="edit", ai_type="B 细胞",
                                  ai_label="B 细胞", final_type="浆细胞").as_dict()])
    assert r.adata.obs["sccopilot_final_type"].iloc[2] == "浆细胞"
    review.apply_reviews(r, [])
    assert set(r.adata.obs["sccopilot_review"]) == {review.UNREVIEWED}
    assert set(r.adata.obs["sccopilot_final_type"]) == {""}
    assert set(r.clusters["sccopilot_final_type"]) == {""}


def test_uns_summary_records_the_rule_and_the_rate():
    r = _result()
    review.apply_reviews(r, [_dec(cluster="0", action="reject").as_dict()])
    meta = r.adata.uns[review.REVIEW_KEY]
    assert meta["n_reviewed"] == 1 and meta["change_rate"] == 1.0
    assert "sccopilot_type" in meta["rule"] and "留空" in meta["rule"]
    assert meta["final_labels"] == {"0": review.REJECTED}


def test_log_for_an_unknown_cluster_does_not_invent_cells():
    r = _result()
    out = review.apply_reviews(r, [_dec(cluster="99").as_dict()])
    assert out["n_cells_reviewed"] == 0
    assert set(r.adata.obs["sccopilot_review"]) == {review.UNREVIEWED}


def test_jsonl_bytes_roundtrip():
    rows = [_dec(note="备注 <b>x</b>").as_dict()]
    data = review.to_jsonl_bytes(rows)
    back = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    assert back == rows
    assert review.to_jsonl_bytes([]) == b""


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
                passed += 1
            except Exception as e:
                import traceback
                print(f"[FAIL] {name}  {e}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(1 if failed else 0)