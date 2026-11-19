# -*- coding: utf-8 -*-
"""scCopilot AI 层：防幻觉的细胞类型解读与分辨率解释。

典型用法：
    from ai.explainer import interpret_cluster, explain_resolution_change
    itp = interpret_cluster(res.genes_of("3"), res.evidence_of("3"), cluster="3")
    if itp.refused:
        ...  # 展示 itp.reasons（为什么拒绝）而不是硬编一个细胞类型
"""

from ai.explainer import (  # noqa: F401
    REFUSAL,
    Evidence,
    Interpretation,
    LLMConfig,
    ResolutionExplanation,
    assess_evidence,
    diff_clusters,
    explain_resolution_change,
    interpret_cluster,
    interpret_result_cluster,
)
from ai import gene_kb, prompts  # noqa: F401
