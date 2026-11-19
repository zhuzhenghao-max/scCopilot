# -*- coding: utf-8 -*-
"""分析溯源（Provenance）追踪器。

设计意图：
- 记录每一步操作的工具名、参数、结果摘要、耗时；
- 提供 export_json() 导出完整的溯源报告（供前端展示或落盘审计）；
- 与 StateAwarePipeline 集成：每次状态流转自动记一笔。

典型用法：
    tracker = ProvenanceTracker()
    tracker.log_step("apply_qc", {"min_genes": 200}, {"n_cells": 2240})
    tracker.log_step("normalize", {"target_sum": 10000}, {"status": "done"})
    report = tracker.export_json()  # → dict
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StepRecord:
    """一条操作记录。"""
    tool_name: str
    params: dict[str, Any]
    result_summary: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "params": self.params,
            "result_summary": self.result_summary,
            "timestamp": self.timestamp,
            "duration_s": round(self.duration_s, 3),
        }


class ProvenanceTracker:
    """分析溯源追踪器：记录每一步操作，支持导出 JSON 报告。"""

    def __init__(self) -> None:
        self._steps: list[StepRecord] = []
        self._start_time: float | None = None

    def start_session(self) -> None:
        """标记一次分析会话的开始（用于计算总耗时）。"""
        self._start_time = time.time()

    def log_step(
        self,
        tool_name: str,
        params: dict[str, Any],
        result_summary: dict[str, Any],
        *,
        duration_s: float = 0.0,
    ) -> None:
        """记录一步操作。

        Parameters
        ----------
        tool_name : str
            工具/函数名（如 "apply_qc", "run_clustering"）。
        params : dict
            调用参数（可序列化为 JSON 的任意结构）。
        result_summary : dict
            结果摘要（如 {"n_cells": 2240, "n_genes": 18000}）。
        duration_s : float
            该步骤耗时（秒）。
        """
        record = StepRecord(
            tool_name=tool_name,
            params=params,
            result_summary=result_summary,
            duration_s=duration_s,
        )
        self._steps.append(record)

    def get_steps(self) -> list[dict[str, Any]]:
        """返回所有步骤的字典列表（按时间顺序）。"""
        return [s.to_dict() for s in self._steps]

    def export_json(self) -> dict[str, Any]:
        """导出完整的溯源报告（JSON 可序列化）。

        Returns
        -------
        dict
            {
                "session_start": float | None,
                "session_duration_s": float | None,
                "total_steps": int,
                "steps": [
                    {"tool_name": ..., "params": ..., "result_summary": ..., "timestamp": ..., "duration_s": ...},
                    ...
                ]
            }
        """
        total_duration = None
        if self._start_time is not None:
            total_duration = round(time.time() - self._start_time, 3)

        return {
            "session_start": self._start_time,
            "session_duration_s": total_duration,
            "total_steps": len(self._steps),
            "steps": self.get_steps(),
        }

    def save_json(self, path: str | Path) -> None:
        """将溯源报告落盘为 JSON 文件。"""
        report = self.export_json()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        """清空所有记录（用于重置会话）。"""
        self._steps.clear()
        self._start_time = None

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        return f"<ProvenanceTracker steps={len(self._steps)}>"
