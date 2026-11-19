# -*- coding: utf-8 -*-
"""scCopilot AI 防幻觉解读层。

对外两个函数：
    interpret_cluster(marker_genes_list, ...)    -> 某簇的细胞类型推断
    explain_resolution_change(resolution, ...)   -> 分辨率调整的解释

为什么"防幻觉"不能只靠 prompt
------------------------------
把"请别瞎编"写进 System Prompt 只是第一层，模型仍可能礼貌地编一个答案。真正可交付的做法
是**程序侧闸门 + 事后审计**，本文件实现了三道：

  闸门①（调用前，本地，零成本）assess_evidence()
      基因名可读性、表达强度、簇规模不达标 -> 直接返回"证据不足，建议人工复核"，
      **根本不发起 HTTP 请求**。不给瞎编的机会，同时省 token、省时间。
  闸门②（调用前）prompt 里的 GENES 白名单 + 输入清洗
      .h5ad 的 var_names 是不可信输入，理论上可被构造成 "ignore previous instructions"；
      非法字符的基因名一律不进 prompt（见 gene_kb.is_valid_symbol）。
  审计③（调用后）_audit_cited_genes()
      模型回答里引用了白名单外的基因 -> 判为不可采信，写入 fabricated_genes
      并把置信度强制降到"低"。

无 API Key / 断网 / 网关报错时：自动退回 gene_kb 的本地规则推断，并在结果里
**明确标注 source=local_kb**，绝不冒充大模型输出。演示现场没网也能完整讲通。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from ai import gene_kb
from ai.prompts import (
    IMMUNOLOGIST_SYSTEM_PROMPT,
    RESOLUTION_SYSTEM_PROMPT,
    SECTION_CONF,
    SECTION_REASON,
    SECTION_TYPE,
    build_cluster_prompt,
    build_resolution_prompt,
)

try:  # requests 是本层唯一必需的网络依赖
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

__all__ = [
    "REFUSAL",
    "LLMConfig",
    "Evidence",
    "Interpretation",
    "ResolutionExplanation",
    "assess_evidence",
    "interpret_cluster",
    "interpret_result_cluster",
    "explain_resolution_change",
    "diff_clusters",
    "load_dotenv",
]

REFUSAL = "证据不足，建议人工复核"
_CONF_ORDER = {"高": 3, "中": 2, "低": 1}

# 模型自拒判定的用词表（见 _self_refusal）
_PAREN = re.compile(r"（[^（）]*）")
_HEDGE_WORDS = ("无法确定", "无法判断", "不能确定", "难以确定", "不确定")
# 限定到"分不到更细一层"的说法：保留它，但不因此丢掉已经给出的类型
_SUBTYPE_HEDGE = re.compile(
    r"(?:无法确定|无法判断|不能确定|难以确定|不确定)[^，。；]{0,10}?"
    r"(?:亚型|亚群|细分|更细|下层|具体亚群)"
)


# --------------------------------------------------------------------------- #
# 配置与密钥
# --------------------------------------------------------------------------- #
def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> int:
    """极简 .env 解析（不引入 python-dotenv）：只设置尚未存在的环境变量，返回写入条数。

    只处理 KEY=VALUE 与 # 注释；值两侧引号会被剥掉。密钥永远不进代码库。
    """
    candidates = [Path(path)] if path else [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    env_file = next((p for p in candidates if p.exists()), None)
    if env_file is None:
        return 0
    n = 0
    for line in env_file.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, _, value = raw.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and (override or key not in os.environ):
            os.environ[key] = value
            n += 1
    return n


@dataclass
class LLMConfig:
    """OpenAI 兼容网关配置。密钥只从环境变量读，绝不硬编码。"""

    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "qwen-plus"
    temperature: float = 0.2
    timeout: float = 60.0
    max_retries: int = 2
    cache: bool = True                 # 同一 prompt 命中缓存就不再计费，也不改变结论

    @classmethod
    def from_env(cls, *, autoload: bool = True) -> "LLMConfig":
        if autoload:
            load_dotenv()

        def _f(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, "") or default)
            except ValueError:
                return default

        return cls(
            api_key=os.getenv("LLM_API_KEY", "").strip(),
            base_url=(os.getenv("LLM_BASE_URL", "").strip()
                      or "https://api.openai.com/v1").rstrip("/"),
            model=os.getenv("LLM_MODEL", "").strip() or "qwen-plus",
            temperature=_f("LLM_TEMPERATURE", 0.2),
            timeout=_f("LLM_TIMEOUT", 60.0),
            max_retries=int(_f("LLM_MAX_RETRIES", 2)),
            cache=os.getenv("LLM_CACHE", "1").strip().lower() not in {"0", "false", "no", "off"},
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key) and requests is not None

    @property
    def disabled_by_env(self) -> bool:
        return os.getenv("SCCOPILOT_DISABLE_AI", "0").strip().lower() in {"1", "true", "yes"}


# --------------------------------------------------------------------------- #
# 闸门①：证据充分性评估（纯本地，不联网）
# --------------------------------------------------------------------------- #
@dataclass
class Evidence:
    """一个簇"值不值得让模型解读"的量化结论。reasons / suggestions 会原样展示给用户。"""

    ok: bool
    n_input: int
    n_usable: int                       # 名字合法 + 表达达标 + 有分类价值的基因数
    informative: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    unannotated: list[str] = field(default_factory=list)
    uninformative: list[str] = field(default_factory=list)
    rejected_by_expression: list[str] = field(default_factory=list)
    unsafe_input: list[str] = field(default_factory=list)
    n_strong: int = 0                   # 检出率>0.2 且 mean_expr>0.3
    novel_ratio: float = 0.0
    mito_ratio: float = 0.0
    n_cells: int | None = None
    has_expression: bool = False
    reasons: list[str] = field(default_factory=list)
    # 非法基因名只是"被清洗掉了"，剩余证据够用就照常解读 → 不能作为拒绝理由
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _as_records(
    marker_genes_list: Sequence[Any], evidence: Any = None
) -> list[dict[str, Any]]:
    """统一输入形态：既接受纯基因名列表，也接受 core 产出的证据表（DataFrame/记录列表）。"""
    gene_to_extra: dict[str, dict[str, Any]] = {}
    if evidence is not None:
        data = evidence.to_dict("records") if hasattr(evidence, "to_dict") else list(evidence)
        for r in data:
            g = str(r.get("gene", "")).strip()
            if g:
                gene_to_extra.setdefault(g, dict(r))

    rows: list[dict[str, Any]] = []
    source: Iterable[Any] = list(gene_to_extra.keys()) if not marker_genes_list else marker_genes_list
    for g in source:
        if isinstance(g, dict):
            rows.append(dict(g))
        else:
            gene = str(g).strip()
            if gene:
                rows.append({**gene_to_extra.get(gene, {}), "gene": gene})
    return rows


def _num(r: dict[str, Any], key: str) -> float:
    """安全取数值；None/NaN/非数字统一返回 nan，由调用方判缺。"""
    try:
        v = float(r.get(key))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return v


def assess_evidence(
    marker_genes_list: Sequence[Any],
    evidence: Any = None,
    cluster_stats: dict[str, Any] | None = None,
    *,
    min_genes: int = 3,
    min_pct_expressed: float = 0.05,
    min_mean_expr: float = 0.1,
    min_cluster_cells: int = 10,
    max_unannotated_ratio: float = 0.5,
    max_mito_ratio: float = 0.5,
) -> Evidence:
    """判定证据是否充分。任一条件不满足即 ok=False，并给出面向用户的理由与补救建议。

    诚实原则：若调用方只给了基因名（没有 mean_expr / pct_expressed），我们**不假装知道**
    表达量——只做名字层面的判断，并在 suggestions 里提示"表达证据缺失"。
    """
    rows = _as_records(marker_genes_list, evidence)
    stats = cluster_stats or {}
    has_expr = any(
        _num(r, "mean_expr") == _num(r, "mean_expr") or _num(r, "pct_expressed") == _num(r, "pct_expressed")
        for r in rows
    )

    rec = gene_kb.recognize([r["gene"] for r in rows])
    safe_rows = [r for r in rows if gene_kb.is_valid_symbol(r["gene"])]

    # ---- 表达强度过滤（仅当拿到了真实表达量时生效）----
    kept: list[dict[str, Any]] = list(safe_rows)
    weak: list[str] = []
    strong: list[str] = []
    if has_expr:
        kept = []
        for r in safe_rows:
            pct, mean = _num(r, "pct_expressed"), _num(r, "mean_expr")
            too_low = (pct == pct and pct < min_pct_expressed) or (mean == mean and mean < min_mean_expr)
            if too_low:
                weak.append(gene_kb.normalize_symbol(r["gene"]))
            else:
                kept.append(r)
            if pct == pct and mean == mean and pct > 0.2 and mean > 0.3:
                strong.append(gene_kb.normalize_symbol(r["gene"]))

    strong_set = set(strong)
    informative = [g for g in rec["informative"] if g not in set(weak)]      # type: ignore[index]
    unknown = [g for g in rec["unknown"] if g not in set(weak)]              # type: ignore[index]
    unannot = list(rec["unannotated"]) + list(rec["invalid"])                # type: ignore[index]
    # 分母用全部输入行：unsafe_input（非法名）也被算进分子，用 safe_rows 当分母会算出 >100% 的占比
    n_total = max(len(rows), 1)
    novel_ratio = round(len(unannot) / n_total, 3)
    mito_ratio = round(sum(1 for r in rows if r.get("is_mito")) / max(len(rows), 1), 3)
    n_cells = stats.get("n_cells")

    ev = Evidence(
        ok=True, n_input=len(rows), n_usable=len(informative) + len(unknown),
        informative=informative, unknown=unknown,
        unannotated=[str(g) for g in rec["unannotated"]],                     # type: ignore[index]
        uninformative=[str(g) for g in rec["uninformative"]],                 # type: ignore[index]
        rejected_by_expression=weak, unsafe_input=[str(g) for g in rec["invalid"]],  # type: ignore[index]
        n_strong=len(strong_set), novel_ratio=novel_ratio, mito_ratio=mito_ratio,
        n_cells=int(n_cells) if isinstance(n_cells, (int, float)) and n_cells == n_cells else None,
        has_expression=has_expr,
    )

    # ---- 逐条判定 ----
    if ev.unsafe_input:
        # 记为 warning 而非 reason：非法名已被剥离，是否调用模型由剩下的证据决定
        ev.warnings.append(
            f"{len(ev.unsafe_input)} 个基因名含非法字符，已拒绝进入模型输入（prompt 注入防护）。"
        )
    if not rows:
        ev.reasons.append("marker 列表为空。")
    elif not safe_rows:
        ev.reasons.append("marker 列表中的基因名全部非法（数据异常或注入文本），已拒绝调用大模型。")
    elif not informative and unannot:
        ev.reasons.append(
            f"Top 基因中 {len(unannot)}/{len(rows)} 个是 LOC/LINC/ENSG 一类未注释名"
            f"（unannotated_ratio={novel_ratio}），无法据此判断细胞类型。"
        )
    elif not informative and ev.uninformative:
        ev.reasons.append(
            "Top 基因全部是管家/核糖体/线粒体/应激基因（如 "
            + ", ".join(ev.uninformative[:4]) + "），这类基因对细胞类型没有区分价值。"
        )
    elif not informative:
        ev.reasons.append("没有命中本地知识库中任何可用于定型的标志基因。")
    elif len(informative) < min_genes:
        ev.reasons.append(
            f"可作为证据的已知标志基因只有 {len(informative)} 个（需要 ≥{min_genes} 个）。"
        )

    if has_expr and ev.n_strong == 0:
        ev.reasons.append("所有 marker 的簇内检出率或平均表达量都过低，判定为技术噪音。")
    elif has_expr and informative and ev.n_strong < 2:
        ev.reasons.append(
            f"仅 {ev.n_strong} 个 marker 达到强证据标准（检出率>0.2 且 mean_expr>0.3）。"
        )
    if novel_ratio > max_unannotated_ratio:
        ev.reasons.append(
            f"未注释基因占比 {novel_ratio:.0%} 超过阈值 {max_unannotated_ratio:.0%}。"
        )
    if mito_ratio >= max_mito_ratio:
        ev.reasons.append(
            f"Top 基因中线粒体基因占 {mito_ratio:.0%}，提示该簇更可能是濒死/低质量细胞而非特定类型。"
        )
    if ev.n_cells is not None and ev.n_cells < min_cluster_cells:
        ev.reasons.append(f"该簇仅 {ev.n_cells} 个细胞（<{min_cluster_cells}），统计功效不足。")

    if ev.reasons:
        ev.ok = False
        if ev.unsafe_input and not safe_rows:
            ev.suggestions.append("检查上传文件的 var_names：合法 gene symbol 只允许字母、数字与 -_. 字符。")
        if not informative and unannot:
            ev.suggestions.append("补齐基因注释（Ensembl ID → symbol）后重跑，或上传带 symbol 的 .h5ad。")
        if has_expr and ev.n_strong == 0:
            ev.suggestions.append("调严 QC（降低 max_pct_mito、提高 min_genes）剔除低质量细胞后重跑。")
        if len(informative) < min_genes:
            ev.suggestions.append("把 n_markers 从 8 提到 15~20，或降低分辨率让小簇合并。")
        if mito_ratio >= max_mito_ratio:
            ev.suggestions.append("建议把该簇标注为「低质量/应激细胞」，不要赋予生物学细胞类型。")
        if ev.n_cells is not None and ev.n_cells < min_cluster_cells:
            ev.suggestions.append("用 resolution≈0.4 重新聚类，让小簇并入邻近群体。")
        if not has_expr:
            ev.suggestions.append("本次仅收到基因名，缺少 mean_expr/pct_expressed，结论强度受限。")
        ev.suggestions = list(dict.fromkeys(ev.suggestions))
    return ev


# --------------------------------------------------------------------------- #
# 结果对象
# --------------------------------------------------------------------------- #
@dataclass
class Interpretation:
    """一次细胞类型解读的完整回执（含审计信息，前端直接渲染）。"""

    cluster: str
    predicted_type: str
    confidence_label: str              # 高 / 中 / 低 / —
    confidence_score: float | None
    reasoning: str
    refused: bool = False
    source: str = "llm"                # llm | local_kb | gate
    gate: dict[str, Any] = field(default_factory=dict)
    cited_genes: list[str] = field(default_factory=list)
    fabricated_genes: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    model: str = ""
    latency_s: float = 0.0
    prompt_chars: int = 0
    raw: str = ""
    cache_hit: bool = False                # True = 本次结论取自本地缓存，没有真的打网关
    cached_at: str = ""                    # 首次请求落盘时间
    notes: list[str] = field(default_factory=list)

    def headline(self) -> str:
        if self.refused:
            return REFUSAL
        if self.confidence_label == "—":
            return self.predicted_type
        return f"{self.predicted_type} · 置信度{self.confidence_label}"

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "gate"}
        d["headline"] = self.headline()
        return d


# --------------------------------------------------------------------------- #
# LLM 传输：OpenAI 兼容 /chat/completions
# --------------------------------------------------------------------------- #
def _request_chat(system: str, user: str, cfg: LLMConfig) -> tuple[str | None, str | None]:
    """真正发一次 HTTP 请求。返回 (content, error)。可重试错误按指数退避；不可重试错误直接给中文说明。"""
    if requests is None:
        return None, "未安装 requests，无法调用大模型。"
    url = f"{cfg.base_url}/chat/completions"
    payload = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        # 故意不设 max_tokens：各网关字段名不一致（有的只认 max_completion_tokens），
        # 设错反而 400。三段式输出本身已被 prompt 限制在 ~300 token。
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}

    last_err = "未知错误"
    for attempt in range(cfg.max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout)
        except requests.exceptions.Timeout:
            last_err = f"请求超时（>{cfg.timeout:.0f}s）"
        except requests.exceptions.RequestException as exc:
            last_err = f"网络异常：{type(exc).__name__}"
        else:
            if resp.status_code == 200:
                try:
                    msg = resp.json()["choices"][0]["message"]
                    # 推理型模型可能把正文放在 reasoning_content
                    content = msg.get("content") or msg.get("reasoning_content") or ""
                    return str(content).strip(), None
                except Exception as exc:  # noqa: BLE001
                    return None, f"响应结构无法解析：{type(exc).__name__}"
            if resp.status_code in (401, 403):
                return None, f"API Key 无效或无权限（HTTP {resp.status_code}）。"
            if resp.status_code == 404:
                return None, (f"模型或地址不存在（HTTP 404）：请检查 LLM_MODEL / LLM_BASE_URL"
                              f"（当前 model={cfg.model}）。")
            if resp.status_code == 429:
                last_err = "触发限流（HTTP 429）"
            elif 500 <= resp.status_code < 600:
                last_err = f"网关错误（HTTP {resp.status_code}）"
            else:
                return None, f"HTTP {resp.status_code}：{resp.text[:160]}"
        if attempt < cfg.max_retries:
            time.sleep(0.8 * (2 ** attempt))          # 0.8s, 1.6s, ...
    return None, last_err


# --------------------------------------------------------------------------- #
# 响应缓存：同一个 prompt + 模型只敲一次网关
#   · 演示时反复点同一簇不再重复计费；
#   · "每个数字都能复现"要求第二次跑出来的结论和第一次一样，缓存把它从愿望变成实现；
#   · 只把请求内容做哈希写盘，Key 既不进文件名也不进文件体。
# --------------------------------------------------------------------------- #
def _cache_dir() -> Path:
    raw = os.getenv("LLM_CACHE_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parents[1] / "data" / "cache" / "llm"


def _cache_file(system: str, user: str, cfg: LLMConfig) -> Path | None:
    if not cfg.cache:
        return None
    blob = json.dumps(
        {"model": cfg.model, "base_url": cfg.base_url, "temperature": cfg.temperature,
         "system": system, "user": user},
        ensure_ascii=False, sort_keys=True,
    )
    return _cache_dir() / f"{hashlib.sha256(blob.encode('utf-8')).hexdigest()[:32]}.json"


def _post_chat(system: str, user: str,
               cfg: LLMConfig) -> tuple[str | None, str | None, dict[str, Any]]:
    """带缓存的调用。返回 (content, error, meta)，meta 里明确交代这次有没有真的打网关。"""
    path = _cache_file(system, user, cfg)
    if path is not None and path.is_file():
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rec = None
        if isinstance(rec, dict) and isinstance(rec.get("content"), str) and rec["content"]:
            return str(rec["content"]), None, {
                "cache_hit": True,
                "cached_at": str(rec.get("created_at", "")),
                "model": str(rec.get("model", cfg.model)),
            }
    content, err = _request_chat(system, user, cfg)
    if content and path is not None:
        try:  # 缓存只是加速手段，写失败不能影响结论
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.part")
            tmp.write_text(json.dumps(
                {"content": content, "model": cfg.model, "temperature": cfg.temperature,
                 "created_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            pass
    return content, err, {"cache_hit": False, "cached_at": "", "model": cfg.model}


# --------------------------------------------------------------------------- #
# 解析与审计
# --------------------------------------------------------------------------- #
# 模型常照抄提示里的标题文字却漏掉方括号（DeepSeek V3 实测如此）。方括号只是排版约定，
# 不是证据，所以允许在【】完全缺席时退回"行首裸标题"。只锚定行首、只认这几个规范名，
# 免得把理由正文里出现的"置信度"之类词也当成标题。
_BARE_SECTION_RE = re.compile(
    r"^[ \t]*(?P<title>预测细胞类型|细胞类型预测|置信度|推断理由|理由|依据)[ \t]*[:：]?[ \t]*",
    re.MULTILINE,
)
_BARE_TITLE_ALIASES = {"细胞类型预测": SECTION_TYPE, "理由": SECTION_REASON, "依据": SECTION_REASON}

_SECTION_RE = re.compile(r"【\s*(?P<title>[^】]{1,20})\s*】\s*[:：]?\s*")
# 看起来像基因符号的 token：全大写、含数字或带 -1A 后缀；用于事后审计
_GENE_LIKE_RE = re.compile(
    r"^(?:[A-Z][A-Z0-9]{1,9}(?:-\d+[A-Za-z]{0,2})?|ENSG\d{6,}|LOC\d+|LINC\d+|MIR\d+[A-Z]?)$"
)
# 下划线挡在两侧：提示词里的 LOCAL_KB_PRIOR 这类标识符曾被切成 LOCAL / PRIOR 两个"基因"，
# 把一句诚实的"本次未提供这些亚群基因"误判成编造证据。原子组 (?>...) 防止回溯退化成 LOCA。
_GENE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_\-])(?>[A-Z][A-Za-z0-9\-]{1,15})(?![a-z_])")
# CD4⁺ / CD8+ 这类写法是表型命名，不等于"本次测到了这个基因"，与类型行按同一套处理
_PHENOTYPE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z][A-Z0-9\-]{1,15})[⁺+−]")
# 白名单外的基因只在这种"声明本次没有它"的限定句里出现时，是模型在交代局限或建议补测，
# 不是拿它当证据
_HEDGING_RE = re.compile(
    r"未提供|未检测|未测|未检出|未见|未包含|未验证|无法(验证|区分|判定|确认)"
    # "CD4/CD8 未在输入中，故仅定到泛 T 层面"：模型在交代数据边界。漏了这类说法，
    # 同半句里带 ⁺ 的 CD4 会被当成限定、CD8 却按编造证据降级——一句诚实话两种判法。
    r"|未在|不在输入|未列入|缺少|缺乏|不采纳|不足以|没有|排除|不计入|不作证据|建议追加|建议补充|建议增加"
    r"|追加|待补|待验证|not provided|not detected|absent"
)
_CLAUSE_SPLIT_RE = re.compile(r"[\n。；;！!？?]")
# 回答里常见但不是基因的全大写词：放进黑名单避免误伤（审计宁漏不误）
_NON_GENE_TOKENS = {
    "QC", "UMAP", "PCA", "HVG", "TOP", "LFC", "MEAN", "LOG", "CPM", "RNA", "DNA", "ATP",
    "ND", "CO", "CYB", "CLUSTER", "GENE", "GENES", "MARKER", "MARKERS", "RANK", "PCT",
    "SCORE", "WILCOXON", "LEIDEN", "SCANPY", "SEURAT", "REST", "NA", "AND", "OR", "NOT",
    "KB", "LLM", "AI", "T", "B", "NK", "DC", "PDC", "MONO", "MHC", "MITO", "GZM",
    "TCR", "BCR", "CAR", "CD",
    # 免疫学缩写，形状像基因符号但不是基因（CTL 曾把正确的 NK 结论打到「低」）
    "CTL", "TREG", "TFR", "TFH", "TH1", "TH2", "TH17", "ILC", "ILC2",
    "PBMC", "BMNC", "MNC", "FACS", "CFSE", "H5AD", "CNV", "GEO", "BED",
}
_KEY_MAP = {
    SECTION_TYPE: ("类型", "细胞", "cell", "type"),
    SECTION_CONF: ("置信", "confidence", "conf"),
    SECTION_REASON: ("理由", "依据", "reason", "推断"),
}


def _parse_sections(text: str) -> dict[str, str]:
    """按【标题】切片解析，容忍模型少写/多写冒号、把内容写在下一段之前。"""
    hits = list(_SECTION_RE.finditer(text or ""))
    if not hits:
        hits = list(_BARE_SECTION_RE.finditer(text or ""))
    out: dict[str, str] = {}
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        title = _BARE_TITLE_ALIASES.get(m.group("title").strip(), m.group("title").strip())
        # 同名标题重复出现时取第一次：正文里再起一行"依据："不该把主结论覆盖掉
        if title and title not in out:
            out[title] = text[m.end():end].strip(" \n:：-—")
    return out


def _pick(sec: dict[str, str], wanted: str) -> str:
    """标题模糊匹配：模型可能写成"预测细胞类型"或"细胞类型预测"。"""
    if wanted in sec:
        return sec[wanted]
    for token in _KEY_MAP.get(wanted, (wanted,)):
        for k, v in sec.items():
            kl = k.lower()
            if token.lower() in kl or token in k:
                return v
    return ""


def _self_refusal(text: str) -> bool:
    """模型有没有在【预测细胞类型】里自己否认这个结论。

    只看去掉括号后的主句，并且把"亚型/亚群待定"这类限定排除在外：那是承认没法分到更细
    的层次，不是否认已经给出的谱系结论。整串扫描会把
    ``T 细胞（…，但本次输入未提供 CD4、CD8A，无法确定亚型）`` 这种 0.90 置信度的诚实回答
    误判成拒答——实测踩过，卡片上显示"AI 主动拒绝回答"，而原文明明给了类型。
    """
    head = _PAREN.sub("", text or "").strip()
    if REFUSAL[:4] in head:                      # "证据不足"——模型自拒，没有别的读法
        return True
    for hedge in _HEDGE_WORDS:
        i = head.find(hedge)
        if i >= 0 and not _SUBTYPE_HEDGE.match(head, i):
            return True
    return False


def _parse_confidence(text: str) -> tuple[str, float | None]:
    """从"高（0.82）" / "0.85" / "82%" 里归一化出 (档位, 数值)。"""
    t = (text or "").strip()
    label = next((k for k in ("高", "中", "低") if k in t), "—")
    score: float | None = None
    m = re.search(r"0?\.\d+|(?<![\d.%])(?:[0-9]{1,2}|100)(?![\d.])", t)
    if m:
        token = m.group(0)
        try:
            score = float(token)
        except ValueError:
            score = None
        if score is not None and score > 1.5:            # 有人写 "82" 表示 0.82
            score /= 100.0
        if score is not None:
            score = max(0.0, min(1.0, score))
    if label == "—" and score is not None:               # 只给数字不给档位
        label = "高" if score >= 0.75 else ("中" if score >= 0.45 else "低")
    if score is None:
        score = {"高": 0.85, "中": 0.6, "低": 0.3}.get(label)
    return label, score


def _audit_cited_genes(
    answer: str, allowed: Iterable[str]
) -> tuple[list[str], list[str], list[str], dict[str, tuple[str, ...]]]:
    """审计③：区分"回答引用的基因确实在输入白名单里"与"疑似凭空引用"。

    启发式，宁漏报不误伤：先要求 token 长得像基因符号，再排除常见非基因全大写词。

    Returns
    -------
    (cited, fabricated, mentioned, aliased)
        mentioned 是白名单外、但只在限定句或家族命名里出现的名字——模型在交代局限或简称，
        拿它当编造证据降级等于惩罚诚实，所以单列出来进 notes，不参与降级。
        aliased 是惯用别名（CD20、NKG2D）-> 本次 marker 符号，按真实引用计入 cited。
    """
    text = answer or ""
    allowed_up = {gene_kb.normalize_symbol(g) for g in allowed}

    def symbols(s: str) -> set[str]:
        # 模型常写 CD3D/CD3E 这种斜杠连写，拆开再判一次
        return {p for t in _GENE_TOKEN_RE.findall(s) for p in re.split(r"[、,/]", t) if p}

    tokens = {t for t in symbols(text)
              if _GENE_LIKE_RE.match(t) and t not in _NON_GENE_TOKENS}
    # 命中判定再放宽一次大小写：小鼠数据（Cd3d）不会被全大写模式抓到，但它引用的确实是
    # 本次输入里的基因。只按白名单回收，不放宽 fabricated，避免把普通英文词误判成编造。
    loose = {
        gene_kb.normalize_symbol(t)
        for t in re.findall(r"(?<![A-Za-z0-9\-])([A-Za-z][A-Za-z0-9\-\.]{1,19})(?![A-Za-z0-9])", text)
        if gene_kb.normalize_symbol(t) in allowed_up
    }
    cited = sorted({t for t in tokens if t in allowed_up} | loose)
    outside = sorted(t for t in tokens if t not in allowed_up)

    # 逐子句判语境：一个词只在"宣称这就是本次证据"的句子里才算编造。
    clauses = [(symbols(c), c) for c in _CLAUSE_SPLIT_RE.split(text)]
    phenotype = set(_PHENOTYPE_RE.findall(text))
    fabricated, mentioned, aliased = [], [], {}
    for gene in outside:
        # CD20 之于 MS4A1：同一个分子的两种叫法，本次确实测到了它，按真实引用计
        hits_alias = tuple(s for s in gene_kb.alias_targets(gene) if s in allowed_up)
        if hits_alias:
            aliased[gene] = hits_alias
            continue
        in_clauses = [c for syms, c in clauses if gene in syms]
        caveat_only = bool(in_clauses) and all(_HEDGING_RE.search(c) for c in in_clauses)
        # CD3 之于 CD3D/CD3E/CD3G：复合体/家族的简称，本次确实测到了成员基因，不算空口引用
        family = any(a != gene and a.startswith(gene) for a in allowed_up)
        (mentioned if caveat_only or family or gene in phenotype else fabricated).append(gene)
    cited = sorted(set(cited) | {s for v in aliased.values() for s in v})
    return cited, fabricated, mentioned, aliased


# --------------------------------------------------------------------------- #
# 主入口①：interpret_cluster
# --------------------------------------------------------------------------- #
def _local_interpretation(informative: list[str], cluster: str, note: str) -> Interpretation:
    """无 LLM 时的兜底：本地知识库加权投票，并明确标注来源与局限。"""
    votes = gene_kb.vote_cell_types(informative, top=3)
    if not votes or int(votes[0]["score"]) < 2:                       # type: ignore[arg-type]
        # 拒答也必须给出可核验的数字：是"一个都没命中"还是"命中了但分数不够"
        if votes:
            top = votes[0]
            why = (f"本地知识库最高候选为 {top['cell_type']}，但得分仅 {top['score']}"
                   f"（阈值 2），命中的标志基因组合不足以定型。")
        else:
            why = (f"本次 {len(informative)} 个输入 marker 在本地知识库中零命中，"
                   "没有任何细胞类型的标志基因组合与之匹配。")
        return Interpretation(
            cluster=cluster, predicted_type=REFUSAL, confidence_label="—",
            confidence_score=None, refused=True, source="local_kb",
            reasoning=why,
            notes=[note, "建议：提供带 symbol 注释的数据，或配置 LLM_API_KEY 以获得更宽泛的推断。"],
        )
    top = votes[0]
    score = int(top["score"])                                         # type: ignore[arg-type]
    label = "中" if score >= 4 else "低"
    canonical = list(top["canonical_hits"])                           # type: ignore[arg-type]
    supportive = list(top["supportive_hits"])                         # type: ignore[arg-type]
    if canonical:
        # 基因名逐个列一遍会读起来很机械；这里按"命中几个 + 具体是哪些"组织，
        # 名字仍然原样出现在【推断理由】里，事后的引用审计照旧生效。
        reason = (f"命中 {len(canonical)} 个该谱系的经典标志基因："
                  f"{'、'.join(canonical[:8])}。")
        if supportive:
            reason += f"另有 {len(supportive)} 个辅助标志基因：{'、'.join(supportive[:6])}。"
    else:
        reason = (f"只命中 {len(supportive)} 个辅助标志基因：{'、'.join(supportive[:8])}，"
                  "未命中经典标志基因，证据强度有限。")
    alt = [v for v in votes[1:] if int(v["score"]) >= max(2, score // 2)]  # type: ignore[arg-type]
    if alt:
        reason += "｜备选：" + "、".join(f"{a['cell_type']}(score={a['score']})" for a in alt)
    return Interpretation(
        cluster=cluster, predicted_type=str(top["cell_type"]),
        confidence_label=label, confidence_score={"中": 0.55, "低": 0.3}[label],
        reasoning=reason, source="local_kb",
        cited_genes=[g for g in informative if g in set(canonical) | set(supportive)],
        notes=[note, f"谱系归属：{top['lineage']}。规则推断是字面 marker 匹配，未按表达量加权，请人工复核。"],
    )


def interpret_cluster(
    marker_genes_list: Sequence[Any],
    evidence: Any = None,
    *,
    cluster: str = "",
    cluster_stats: dict[str, Any] | None = None,
    llm: LLMConfig | None = None,
    use_llm: bool = True,
    gate_kwargs: dict[str, Any] | None = None,
) -> Interpretation:
    """根据某簇的 Top 差异基因推断细胞类型（含三道防幻觉控制）。

    Parameters
    ----------
    marker_genes_list : 基因名列表，或 core.PipelineResult.markers 中该簇的记录行
    evidence : 可选，core 侧的证据表（含 mean_expr / pct_expressed / is_mito）
    cluster_stats : 可选，PipelineResult.clusters 对应那一行（n_cells 等）
    use_llm : False 时强制走本地规则推断（现场无 Key 也不会白屏）

    Returns
    -------
    Interpretation
        refused=True 表示被闸门拦下或模型自己承认不确定；此时 predicted_type 恒为 REFUSAL。
    """
    cfg = llm or LLMConfig.from_env()
    rows = _as_records(marker_genes_list, evidence)
    # 闸门②的数据准备：var_names 属于不可信输入，非法名一律不进 prompt
    safe_rows = [r for r in rows if gene_kb.is_valid_symbol(str(r.get("gene", "")))]
    gate = assess_evidence(rows, None, cluster_stats, **(gate_kwargs or {}))

    # ---- 闸门①：证据不足直接返回，绝不请求模型 ----
    if not gate.ok:
        body = ["现有证据不足以判定细胞类型，原因："]
        body += [f"{i}. {r}" for i, r in enumerate(gate.reasons, 1)]
        if gate.suggestions:
            body.append("改进建议：")
            body += [f"- {s}" for s in gate.suggestions]
        if gate.warnings:
            body.append("其他提示：" + "；".join(gate.warnings))
        return Interpretation(
            cluster=cluster, predicted_type=REFUSAL, confidence_label="—",
            confidence_score=None, reasoning="\n".join(body),
            refused=True, source="gate", gate=gate.as_dict(),
            notes=["未调用大模型：拦截发生在本地证据校验阶段（零 token 成本）。"],
        )

    genes_for_prompt = [r["gene"] for r in safe_rows]      # 闸门②：非法名不进 prompt
    prior = gene_kb.vote_cell_types(gate.informative + gate.unknown, top=3)
    species = "mouse" if gene_kb.looks_mouse(genes_for_prompt) else "human"
    prompt = build_cluster_prompt(
        cluster=cluster or "?", rows=safe_rows, cluster_stats=cluster_stats,
        species_hint=species, local_prior=prior,
    )

    # ---- 无 Key / 显式关闭：本地兜底 ----
    if not use_llm or cfg.disabled_by_env or not cfg.configured:
        why = ("调用方指定 use_llm=False" if not use_llm else
               "SCCOPILOT_DISABLE_AI=1" if cfg.disabled_by_env else
               "未检测到 LLM_API_KEY（见 .env.example）")
        itp = _local_interpretation(gate.informative, cluster,
                                    note=f"未调用大模型（{why}），以下结果由本地规则推断。")
        itp.gate, itp.prompt_chars = gate.as_dict(), len(prompt)
        return itp

    t0 = time.perf_counter()
    content, err, meta = _post_chat(IMMUNOLOGIST_SYSTEM_PROMPT, prompt, cfg)
    latency = round(time.perf_counter() - t0, 2)
    if content is None:
        itp = _local_interpretation(gate.informative, cluster,
                                    note=f"大模型调用失败（{err}），已自动退回本地规则推断。")
        itp.gate, itp.latency_s, itp.model = gate.as_dict(), latency, cfg.model
        return itp

    # ---- 解析三段 ----
    sec = _parse_sections(content)
    raw_type = _pick(sec, SECTION_TYPE)
    conf_label, conf_score = _parse_confidence(_pick(sec, SECTION_CONF))
    reasoning = _pick(sec, SECTION_REASON)

    if not raw_type and not reasoning:
        # 模型完全没按格式来：宁可拒绝也不硬解，把原文留给用户核对
        return Interpretation(
            cluster=cluster, predicted_type=REFUSAL, confidence_label="—", confidence_score=None,
            reasoning="模型未按约定三段格式输出，为避免误读不予采信。原始回复如下，请人工判读：\n"
                      + content[:400],
            refused=True, source="llm", gate=gate.as_dict(), model=cfg.model,
            latency_s=latency, raw=content, violations=["输出格式不符合三段式约定"],
            cache_hit=bool(meta.get("cache_hit")), cached_at=str(meta.get("cached_at", "")),
        )

    # 模型自己也说证据不足
    if _self_refusal(raw_type):
        return Interpretation(
            cluster=cluster, predicted_type=REFUSAL, confidence_label="—", confidence_score=None,
            reasoning=reasoning or "模型判定现有 marker 不足以确定细胞类型。",
            refused=True, source="llm", gate=gate.as_dict(), model=cfg.model,
            latency_s=latency, raw=content,
            cache_hit=bool(meta.get("cache_hit")), cached_at=str(meta.get("cached_at", "")),
        )

    # ---- 审计③：证据部分只认输入白名单 ----
    # 分两处审计的理由：模型常把类型写成"初始 CD4+ T 细胞（CCR7+SELL+）"，这类括号内容
    # 属于命名惯例而非本次证据，只在 notes 里提示；而【推断理由】里引用列表外的基因，
    # 就是实打实的编造证据，必须降级。
    cited_r, fabricated_r, mentioned_r, aliased_r = _audit_cited_genes(reasoning, genes_for_prompt)
    cited_t, fabricated_t, _, _ = _audit_cited_genes(raw_type, genes_for_prompt)
    cited = sorted(set(cited_r) | set(cited_t))
    fabricated = list(fabricated_r)
    violations: list[str] = []
    if fabricated:
        violations.append(
            f"【推断理由】引用了 {len(fabricated)} 个不在本次 marker 列表中的基因名（"
            + ", ".join(fabricated[:8]) + "），相关论述不可采信。"
        )
        conf_label, conf_score = "低", min(conf_score or 0.3, 0.3)
    if not cited_r:
        violations.append("模型没有引用任何本次输入的真实 marker，结论缺乏可核验依据。")
        conf_label, conf_score = "低", min(conf_score or 0.3, 0.3)
    if len(gate.unknown) >= max(3, len(genes_for_prompt) // 2):
        if _CONF_ORDER.get(conf_label, 0) > _CONF_ORDER["中"]:
            violations.append(
                f"{len(gate.unknown)} 个 marker 不在本地知识库内，置信度按规则由「高」降为「中」。"
            )
            conf_label, conf_score = "中", min(conf_score or 0.55, 0.6)

    extra_notes: list[str] = [f"数据质量提示：{w}" for w in gate.warnings]
    if _SUBTYPE_HEDGE.search(raw_type):
        # 不吞掉模型的限定：接受谱系层面的结论，但要说清它没细到亚群
        extra_notes.append(
            "模型在类型名里保留了「亚型待定」：本次 marker 只够定到当前命名层次，"
            "不要把它当成更细的亚群名使用。")
    if aliased_r:
        extra_notes.append(
            "【推断理由】的 " + "、".join(k + "=" + "/".join(v) for k, v in sorted(aliased_r.items()))
            + " 是本次 marker 的惯用别名（CD 编号 / NKG 俗名），按同一个分子计入引用。"
        )
    if mentioned_r:
        extra_notes.append(
            "【推断理由】提到的 " + "、".join(mentioned_r[:8])
            + " 不在本次 marker 列表内，但只出现在「本次没有/无法验证」的限定句、"
            "家族简称或表型命名（X⁺）里，未计入证据，也未判为编造。"
        )
    if fabricated_t:
        extra_notes.append(
            "类型名称中出现本次未提供的基因（" + ", ".join(fabricated_t[:6])
            + "），按命名惯例处理、未计入证据审计。"
        )

    if meta.get("cache_hit"):
        extra_notes.insert(
            0,
            f"本次结论取自本地响应缓存（首次请求时间 {meta.get('cached_at') or '未知'}），"
            "未产生新的 API 调用；同一份输入重跑必然得到同一句话。"
            "关掉缓存：在 .env 里设 LLM_CACHE=0。",
        )

    return Interpretation(
        cluster=cluster, predicted_type=raw_type.strip(),
        confidence_label=conf_label, confidence_score=conf_score,
        reasoning=reasoning, refused=False, source="llm", gate=gate.as_dict(),
        cited_genes=cited, fabricated_genes=fabricated, violations=violations,
        model=cfg.model, latency_s=latency, prompt_chars=len(prompt), raw=content,
        cache_hit=bool(meta.get("cache_hit")), cached_at=str(meta.get("cached_at", "")),
        notes=extra_notes,
    )


def interpret_result_cluster(
    result: Any, cluster: str, *, llm: LLMConfig | None = None, use_llm: bool = True
) -> Interpretation:
    """便利入口：从 core.PipelineResult 直接取该簇的 marker + 证据 + 簇统计。

    用鸭子类型而非 import core，避免 ai <-> core 循环依赖。
    """
    markers = result.evidence_of(cluster)
    stats: dict[str, Any] = {}
    table = getattr(result, "clusters", None)
    if isinstance(table, pd.DataFrame) and not table.empty:
        row = table[table["cluster"].astype(str) == str(cluster)]
        if not row.empty:
            stats = {
                k: (None if isinstance(v, float) and v != v else v)
                for k, v in row.iloc[0].to_dict().items()
            }
    return interpret_cluster(
        markers.to_dict("records"), None, cluster=str(cluster),
        cluster_stats=stats, llm=llm, use_llm=use_llm,
    )


# --------------------------------------------------------------------------- #
# 主入口②：explain_resolution_change
# --------------------------------------------------------------------------- #
@dataclass
class ResolutionExplanation:
    resolution: float
    prev_resolution: float | None
    headline: str                      # 一句人话结论（本地算出，与模型无关）
    summary: str
    why: str
    next_steps: str
    diff: dict[str, Any] = field(default_factory=dict)
    source: str = "rule"
    model: str = ""
    latency_s: float = 0.0
    raw: str = ""
    notes: list[str] = field(default_factory=list)


def diff_clusters(prev: Any, new: Any, *, max_items: int = 6,
                  min_share: float = 0.05) -> dict[str, Any]:
    """对比同一批细胞在两次聚类下的标签，算出真实的拆分 / 合并关系。

    prev / new 是可对齐的两个标签序列（pandas.Series 或 list）。返回的全是可核验数字，
    模型只能引用它们，不能自己发挥。

    min_share：一个去向至少要占该簇这么多个细胞才算数。Leiden 换分辨率后本来就会有个位
    数细胞改判到隔壁，若按"出现过多标签即拆分"判定，450 个细胞的簇会因为跑掉 1 个被判成
    "拆成 3 份、纯度 99.6%"——自相矛盾（实测踩过）。阈值之下的零头归为标签噪声。
    """
    p = pd.Series(list(prev)).astype(str).reset_index(drop=True)
    n = pd.Series(list(new)).astype(str).reset_index(drop=True)
    if len(p) != len(n):
        raise ValueError(f"前后标签长度不一致（{len(p)} vs {len(n)}），无法对比")

    def material(series: pd.Series) -> pd.Series:
        total = int(series.sum())
        floor = max(1, math.ceil(total * min_share))
        return series[series >= floor]

    ct = pd.crosstab(p, n)
    splits: list[dict[str, Any]] = []
    for old, row in ct.iterrows():
        hit = material(row[row > 0])
        if len(hit) > 1:
            splits.append({
                "from": str(old), "into": [str(k) for k in hit.index],
                "sizes": [int(v) for v in hit.to_numpy()],
                "purity": round(float(hit.max()) / float(hit.sum()), 3),
            })
    merges: list[dict[str, Any]] = []
    # crosstab 的行索引是旧标签、列索引是新标签：拆分沿行看，合并必须沿列看，否则会把自己算成对方
    for newc, col in ct.items():
        hit = material(col[col > 0])
        if len(hit) > 1:
            merges.append({
                "into": str(newc), "from": [str(k) for k in hit.index],
                "sizes": [int(v) for v in hit.to_numpy()],
            })
    splits.sort(key=lambda d: -len(d["into"]))
    merges.sort(key=lambda d: -len(d["from"]))

    sp_, sn_ = p.value_counts(), n.value_counts()
    return {
        "n_before": int(sp_.size), "n_after": int(sn_.size),
        "median_before": float(sp_.median()), "median_after": float(sn_.median()),
        "max_before": int(sp_.max()), "max_after": int(sn_.max()),
        "n_splits": len(splits), "n_merges": len(merges),
        "splits": splits[:max_items], "merges": merges[:max_items],
    }


def _rule_resolution_text(
    resolution: float, prev: float | None, diff: dict[str, Any]
) -> tuple[str, str, str]:
    """模型缺席时的确定性解释：只用真实数字，不编造。"""
    if diff:
        arrow = f"{prev} → {resolution}" if prev is not None else f"{resolution}"
        if diff["n_after"] > diff["n_before"]:
            head = diff["splits"][0] if diff["splits"] else None
            headline = (
                f"分辨率 {arrow}：簇数 {diff['n_before']} → {diff['n_after']}，"
                f"中位簇规模 {diff['median_before']:.0f} → {diff['median_after']:.0f} 个细胞"
                + (f"；其中簇 {head['from']} 被拆成 {len(head['into'])} 份" if head else "")
            )
            why = (
                "Leiden 分辨率越高，越倾向在同一区域里切出更多小块，因此原本混在一起的细胞被"
                "分给不同簇。注意：UMAP 坐标没有变化，变的只是同一张邻居图上的切分粒度。"
            )
        elif diff["n_after"] < diff["n_before"]:
            head = diff["merges"][0] if diff["merges"] else None
            headline = (
                f"分辨率 {arrow}：簇数 {diff['n_before']} → {diff['n_after']}，"
                f"中位簇规模 {diff['median_before']:.0f} → {diff['median_after']:.0f} 个细胞"
                + (f"；新簇 {head['into']} 由 {len(head['from'])} 个旧簇合并而来" if head else "")
            )
            why = (
                "分辨率越低，判定「属于同一群」的门槛越宽松，转录组相近的群体被并成一个簇，"
                "原来可辨的亚群结构会被抹平。"
            )
        else:
            headline = f"分辨率 {arrow}：簇数保持 {diff['n_after']}，但细胞归属可能已重排"
            why = "簇数相同不代表结果相同，需要用 marker 与前后标签交叉表确认边界是否移动。"
        steps = (
            "看新拆出的簇是否各自拥有稳健特异的 marker（AI 面板对每个簇都给出证据判定）；"
            "若新簇只由线粒体基因或未注释名支撑，应视为低质量群体而非真实亚群；"
            "建议在 0.4 / 1.0 / 2.0 三档对比簇数与 marker 命中数后再定稿。"
        )
        return headline, why, steps

    headline = (
        f"resolution = {resolution}：该参数只决定在同一张邻居图上把细胞「切多细」，"
        "不改变 UMAP 坐标，也不引入新的生物学信息。"
    )
    why = (
        "调大 resolution，Leiden 更容易把一个区域切成多块（亚群被拆分、簇数上升、平均簇规模下降）；"
        "调小则把相近群体合并（簇数下降）。"
    )
    steps = (
        "本次未获得前后聚类对比数据，以上为一般性说明。"
        "请保持 PCA/邻居图不变、只改 resolution，并逐个簇检查特异 marker 是否成立。"
    )
    return headline, why, steps


def explain_resolution_change(
    resolution: float,
    *,
    prev_resolution: float | None = None,
    prev_labels: Any = None,
    new_labels: Any = None,
    clusters_table: Any = None,
    llm: LLMConfig | None = None,
    use_llm: bool = True,
) -> ResolutionExplanation:
    """解释"为何细胞亚群被拆分/合并"。

    关键取舍：**所有结论性数字都由 diff_clusters() 本地算出**，模型只负责把它们组织成
    通顺的自然语言；没传前后标签时退化为一般性原理说明，并在 notes 里声明数据缺口，
    绝不编造"从 12 簇变成 20 簇"这种没算过的数字。
    """
    cfg = llm or LLMConfig.from_env()
    notes: list[str] = []

    diff: dict[str, Any] = {}
    if prev_labels is not None and new_labels is not None:
        try:
            diff = diff_clusters(prev_labels, new_labels)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"前后标签对比失败：{exc}")
    else:
        notes.append("未获得前后聚类标签，解释只覆盖一般性原理，未引用具体簇数。")

    headline, why_rule, steps_rule = _rule_resolution_text(resolution, prev_resolution, diff)
    fallback = ResolutionExplanation(
        resolution=resolution, prev_resolution=prev_resolution, headline=headline,
        summary=headline, why=why_rule, next_steps=steps_rule, diff=diff,
        source="rule", notes=notes,
    )

    rows: list[dict[str, Any]] = []
    if isinstance(clusters_table, pd.DataFrame) and not clusters_table.empty:
        cols = [c for c in ("cluster", "n_cells", "n_strong_markers",
                            "novel_marker_ratio", "evidence_ok") if c in clusters_table.columns]
        rows = clusters_table[cols].to_dict("records")

    prompt = build_resolution_prompt(
        resolution=resolution, prev_resolution=prev_resolution,
        before=({"n_clusters": diff["n_before"], "median_size": diff["median_before"],
                 "max_size": diff["max_before"]} if diff else None),
        after=({"n_clusters": diff["n_after"], "median_size": diff["median_after"],
                "max_size": diff["max_after"]} if diff else None),
        split_pairs=[
            f"旧簇{s['from']} → 新簇{'/'.join(s['into'])}（sizes={s['sizes']}，主簇纯度 {s['purity']}）"
            for s in diff.get("splits", [])
        ] or None,
        table_rows=rows,
    )

    if not use_llm or cfg.disabled_by_env or not cfg.configured:
        return fallback

    t0 = time.perf_counter()
    content, err, meta = _post_chat(RESOLUTION_SYSTEM_PROMPT, prompt, cfg)
    latency = round(time.perf_counter() - t0, 2)
    if meta.get("cache_hit"):
        notes = list(notes) + [f"本次解释取自本地缓存（首次请求 {meta.get('cached_at') or '未知'}）。"]
        fallback.notes = notes
    if content is None:
        fallback.notes = notes + [f"大模型调用失败（{err}），已返回本地规则解释。"]
        fallback.latency_s = latency
        return fallback

    sec = _parse_sections(content)
    summary, why, nxt = _pick(sec, "变化摘要"), _pick(sec, "为什么会这样"), _pick(sec, "下一步建议")
    if not (summary and why):
        fallback.notes = notes + ["模型未按三段格式输出，已返回本地规则解释。"]
        fallback.raw, fallback.latency_s = content, latency
        return fallback

    # 数字审计：模型写出的整数簇数必须来自本地计算，否则标记为未经计算
    if diff:
        allowed = {str(int(v)) for v in (diff["n_before"], diff["n_after"], diff["max_before"],
                                        diff["max_after"]) if v == v}
        allowed |= {str(int(round(diff["median_before"]))), str(int(round(diff["median_after"])))}
        allowed |= {str(int(resolution)), str(int(prev_resolution or 0))}
        invented = [
            tok for tok in re.findall(r"(?<![\d.])(\d{1,4})(?![\d.])", summary)
            if tok not in allowed and int(tok) > 3
        ]
        if invented:
            fallback.notes = notes + [
                f"模型引用了未经计算的数字：{', '.join(sorted(set(invented))[:5])}（已保留其文字，"
                f"但请以 headline 中本地计算的数字为准）。"
            ]
    fallback.summary, fallback.why, fallback.next_steps = summary, why, nxt or steps_rule
    fallback.source, fallback.model, fallback.raw, fallback.latency_s = "llm", cfg.model, content, latency
    return fallback
