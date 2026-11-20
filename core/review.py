# -*- coding: utf-8 -*-
"""人工复核：把「AI 首稿 → 人工定稿」这一步变成可追溯、可统计的记录。

科研者真正的痛点不是 AI 给不出名字，而是给完之后没法交代「谁在什么时候改了什么」。
所以这里的设计只有三条硬规则：

1. **append-only**：每次复核往 jsonl 追加一行，不改写历史；同一簇后写的决定生效，
   但先写的仍留在日志里，改动率与决策次数都能复算。
2. **首稿与终稿分列**：``sccopilot_type`` 永远是 AI 首稿，人工定稿写
   ``sccopilot_final_type``。未复核的簇终稿留空——不把 AI 首稿冒充人工定稿。
3. **口径写在脸上**：改动率的分母是「已复核簇数」，不是全部簇数；覆盖率单独给。

本模块不重算任何生物学数字，只搬运决策记录。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

REVIEW_ACTIONS = ("accept", "edit", "reject")
ACTION_LABELS = {"accept": "采纳首稿", "edit": "修改类型", "reject": "否决首稿"}

REVIEW_COLS = ("sccopilot_review", "sccopilot_final_type")
REVIEW_KEY = "sccopilot_review"          # adata.uns 下的落款
UNREVIEWED = "unreviewed"                # 还没人复核过的簇
REJECTED = "human_rejected"              # 人工否决且未给替代标签


@dataclass
class Decision:
    """一次复核决定。``final_type`` 由动作推导，避免调用方写出自相矛盾的组合。"""

    cluster: str
    ai_type: str                         # 页面展示的 AI 首稿（拒答簇就是拒答文案）
    ai_label: str                        # 首稿落到 obs 的值：类型 / low_confidence
    action: str
    final_type: str = ""
    note: str = ""
    reviewer: str = ""
    source: str = ""
    confidence: float | None = None
    run_id: str = ""
    ts: str = ""

    def __post_init__(self) -> None:
        self.cluster = str(self.cluster)
        self.action = str(self.action)
        if self.action not in REVIEW_ACTIONS:
            raise ValueError(f"未知复核动作：{self.action!r}（只接受 {REVIEW_ACTIONS}）")
        if not self.ts:
            self.ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if self.action == "accept":
            self.final_type = str(self.ai_label)
        elif self.action == "reject":
            self.final_type = REJECTED
        else:
            self.final_type = str(self.final_type).strip()
            if not self.final_type:
                raise ValueError("选择「修改类型」必须给出最终标签，不能留空")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def append_decision(path: str | Path, decision: Decision | Mapping[str, Any]) -> Path:
    """追加一行 JSON。文件不存在就建目录；历史行永不改写。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = decision.as_dict() if isinstance(decision, Decision) else dict(decision)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def read_decisions(path: str | Path) -> list[dict[str, Any]]:
    """按写入顺序读回日志。坏行直接报错，不静默丢弃——少一行就会算错改动率。"""
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"复核日志第 {i} 行不是合法 JSON（{p}）：{exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"复核日志第 {i} 行不是对象（{p}）")
        out.append(row)
    return out


def latest_by_cluster(decisions: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """同一簇后写覆盖先写；返回的是「当前生效」的决定，不是历史。"""
    latest: dict[str, dict[str, Any]] = {}
    for row in decisions:
        cid = str(row.get("cluster", ""))
        if cid:
            latest[cid] = dict(row)
    return latest


def review_stats(decisions: Iterable[Mapping[str, Any]],
                 clusters: Iterable[str]) -> dict[str, Any]:
    """复核口径统计。

    ``n_changed`` 数的是「最终标签 != 首稿 obs 标签」的簇：选了「修改」却填回同一个
    名字，动作计数里算 edit，但不算标签改动——改动率不能靠动作名虚高。
    """
    log = list(decisions)
    latest = latest_by_cluster(log)
    all_clusters = [str(c) for c in clusters]
    n_reviewed = len(latest)
    counts = {a: 0 for a in REVIEW_ACTIONS}
    n_changed = 0
    for row in latest.values():
        action = str(row.get("action", ""))
        if action in counts:
            counts[action] += 1
        if str(row.get("final_type", "")) != str(row.get("ai_label", "")):
            n_changed += 1
    reviewed = set(latest)
    return {
        "n_decisions": len(log),
        "n_reviewed": n_reviewed,
        "n_clusters": len(all_clusters),
        "coverage": (n_reviewed / len(all_clusters)) if all_clusters else 0.0,
        "n_accept": counts["accept"],
        "n_edit": counts["edit"],
        "n_reject": counts["reject"],
        "n_changed": n_changed,
        "change_rate": (n_changed / n_reviewed) if n_reviewed else 0.0,
        "unreviewed_clusters": [c for c in all_clusters if c not in reviewed],
    }


def apply_reviews(result: Any, decisions: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """把当前生效的复核决定写回 ``obs`` / 簇表 / ``uns``。

    与 ``PipelineResult.annotate`` 一样每次整列重写：撤销复核或换分辨率之后，
    不会留下上一轮的旧终稿。首稿列 ``sccopilot_type`` 一个字符都不动。
    """
    latest = latest_by_cluster(decisions)
    a = result.adata
    obs = a.obs[result.cluster_key].astype(str)
    action = np.full(len(obs), UNREVIEWED, dtype=object)
    final = np.full(len(obs), "", dtype=object)
    n_cells = 0
    for cid, row in latest.items():
        idx = np.flatnonzero((obs == cid).to_numpy())
        if idx.size == 0:
            continue
        action[idx] = str(row.get("action") or UNREVIEWED)
        final[idx] = str(row.get("final_type") or "")
        n_cells += int(idx.size)
    for col, arr in zip(REVIEW_COLS, (action, final)):
        a.obs[col] = arr

    if "cluster" in result.clusters.columns:
        cl_str = result.clusters["cluster"].astype(str).tolist()
        result.clusters[REVIEW_COLS[0]] = [
            str((latest.get(c) or {}).get("action") or UNREVIEWED) for c in cl_str]
        result.clusters[REVIEW_COLS[1]] = [
            str((latest.get(c) or {}).get("final_type") or "") for c in cl_str]

    stats = review_stats(latest.values(), result.clusters["cluster"].astype(str).tolist())
    a.uns[REVIEW_KEY] = {
        "tool": "scCopilot",
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cluster_key": result.cluster_key,
        "rule": ("accept → 终稿=首稿 obs 标签；edit → 终稿=人工填写；reject → human_rejected；"
                 "未复核 → 终稿留空。首稿始终保留在 sccopilot_type，不被覆盖。"),
        "n_reviewed": stats["n_reviewed"],
        "n_changed": stats["n_changed"],
        "change_rate": stats["change_rate"],
        "final_labels": {c: str(r.get("final_type") or "") for c, r in latest.items()},
    }
    return {"n_reviewed": stats["n_reviewed"], "n_cells_reviewed": int(n_cells),
            "n_cells_total": int(len(obs)), "columns": list(REVIEW_COLS), **stats}


def to_jsonl_bytes(decisions: Iterable[Mapping[str, Any]]) -> bytes:
    """导出用的字节串。调用方在渲染期取快照——下载回调线程里读不到 session_state。"""
    lines = [json.dumps(dict(r), ensure_ascii=False) for r in decisions]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")