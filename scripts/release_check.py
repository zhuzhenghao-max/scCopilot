# -*- coding: utf-8 -*-
"""上线前门禁：一条命令判断「这份材料现在能不能发」。

    python -X utf8 scripts/release_check.py [--skip-tests] [--reexport]

为什么不只看测试绿：静态站是预计算 JSON，下面三类问题都不会让任何测试失败，
却会在答辩现场被当场问出来——代码改了没重导出数据、前端展示的数字与证据闸门
自相矛盾、把 .env 里的 key 打进发布物。

--reexport 会真的重跑一遍导出（几分钟），用来验证 web/data/*.json 里的生物学数字能被当前
代码复现（比对时忽略耗时与时间戳）。它先把原文件备份，比对完无条件恢复，所以失败也不会污染
已上线数据（本项目不是 git 仓库，没有 git 兜底）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB, DIST = ROOT / "web", ROOT / "dist"
TEXT_EXT = {".html", ".css", ".js", ".json", ".svg", ".txt", ".md"}
VOLATILE_KEYS = {"timings", "timestamp", "duration_s", "generated_at"}
results: list[tuple[str, str]] = []


def report(name: str, ok: bool, detail: str = "") -> bool:
    results.append(("OK" if ok else "FAIL", name))
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f"\n       {detail}" if detail else ""))
    return ok


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tree(d: Path) -> dict[str, Path]:
    return {str(f.relative_to(d)): f for f in sorted(d.rglob("*")) if f.is_file()}


def check_tests() -> None:
    r = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "scripts" / "run_tests.py")],
                       cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    # 用例数不写死在标签里：写死就会漂成下一个「145 vs 152」。数字只从 run_tests 的输出里读。
    line = next((l for l in (r.stdout or "").splitlines() if l.startswith("合计")), "")
    m = re.search(r"合计 (\d+)/(\d+) passed", line)
    name = f"全量回归全绿（{m.group(2)} 用例）" if m else "全量回归全绿"
    report(name, r.returncode == 0 and "passed" in line, line or (r.stdout or "")[-400:])


def check_dist_sync() -> None:
    if not DIST.exists():
        return report("发布副本 dist/ 与 web/ 同步", False, "dist/ 不存在，发布前需要复制一份")
    src, dst = tree(WEB), tree(DIST)
    missing = sorted(set(src) - set(dst))
    stale = [k for k in set(src) & set(dst) if digest(src[k]) != digest(dst[k])]
    extra = [k for k in sorted(set(dst) - set(src)) if not k.endswith(".map")]
    bad = missing + stale + extra
    report("发布副本 dist/ 与 web/ 同步", not bad,
           "缺：" + ", ".join(missing) + " | 旧：" + ", ".join(stale) + " | 多：" + ", ".join(extra))


def check_scenarios() -> None:
    """静态站展示的数字必须自洽，且与证据闸门判定口径一致。"""
    problems: list[str] = []
    manifest = json.loads((WEB / "data" / "manifest.json").read_text(encoding="utf-8"))
    for s in manifest["scenarios"]:
        f = WEB / "data" / s["file"]
        if not f.exists():
            problems.append(f"{s['id']}: 缺文件 {s['file']}")
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        qc, clusters, ai = d["qc"], d["clusters"], d["ai"]
        sid = s["id"]
        if s["n_cells"] != qc["n_cells_final"]:
            problems.append(f"{sid}: manifest n_cells={s['n_cells']} ≠ qc.n_cells_final={qc['n_cells_final']}")
        if s["n_cells"] != sum(c["n_cells"] for c in clusters):
            problems.append(f"{sid}: 簇表细胞数合计 {sum(c['n_cells'] for c in clusters)} ≠ {s['n_cells']}")
        if s["n_clusters"] != len(clusters):
            problems.append(f"{sid}: manifest n_clusters={s['n_clusters']} ≠ 簇表 {len(clusters)}")
        if len(clusters) != len(ai):
            problems.append(f"{sid}: 簇表 {len(clusters)} 行，AI 结论 {len(ai)} 条")
        if "umap" in d and len(d["umap"]["x"]) != s["n_cells"]:
            problems.append(f"{sid}: UMAP 点数 {len(d['umap']['x'])} ≠ n_cells {s['n_cells']}")
        diff = d.get("resolution_diff")
        if diff and diff["n_after"] != len(clusters):
            problems.append(f"{sid}: resolution_diff.n_after={diff['n_after']} ≠ 簇表 {len(clusters)}")
        for c in clusters:
            entry = ai.get(str(c["cluster"]))
            if not entry:
                continue
            gate_ok = bool((entry.get("gate") or {}).get("ok"))
            if gate_ok != bool(c["evidence_ok"]):
                problems.append(f"{sid}/簇{c['cluster']}: 前端 evidence_ok={c['evidence_ok']} "
                                f"与闸门 gate.ok={gate_ok} 不一致")
            if entry.get("refused") and "证据不足" not in entry.get("predicted_type", ""):
                problems.append(f"{sid}/簇{c['cluster']}: refused=True 但类型名没写「证据不足」")
            if not entry.get("refused") and entry.get("violations") \
                    and entry.get("confidence_label") != "低":
                problems.append(f"{sid}/簇{c['cluster']}: 有 violations 但置信度没降到「低」")
    report("静态站三个场景数字自洽 + 与闸门口径一致", not problems, "\n       ".join(problems[:12]))


def check_secrets() -> None:
    env = ROOT / ".env"
    values = []
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if v and len(v) >= 16 and k.strip().upper().endswith(("KEY", "TOKEN", "SECRET")):
                    values.append((k.strip(), v))
    hits = []
    for d in (WEB, DIST):
        for rel, f in tree(d).items() if d.exists() else []:
            if f.suffix in TEXT_EXT and any(v in f.read_text(encoding="utf-8", errors="replace")
                                            for _, v in values):
                hits.append(f"{d.name}/{rel}")
    ok = not hits and not (WEB / ".env").exists() and not (DIST / ".env").exists()
    report(f"发布物不含密钥（已核 {len(values)} 个 .env 值）", ok, "命中：" + ", ".join(hits))


def _canonical(p: Path) -> bytes:
    """去掉墙钟字段后的规范化内容。

    导出 JSON 里有 timings.*、trace.*.timestamp、duration_s——这些每次跑必然不同，
    拿逐字节比对去要求它们相等，等于给门禁装了一个永远亮红灯的开关。
    """
    if p.suffix != ".json":
        return p.read_bytes()
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in VOLATILE_KEYS}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    d = strip(json.loads(p.read_text(encoding="utf-8")))
    return json.dumps(d, ensure_ascii=False, sort_keys=True).encode()


def check_reexport() -> None:
    backup = Path(tempfile.mkdtemp(prefix="release_data_"))
    original = WEB / "data"
    shutil.copytree(original, backup / "data")
    try:
        r = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "scripts" / "export_site_data.py")],
                           cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        if r.returncode != 0:
            report("web/data 生物学数字可由当前代码复现", False,
                   "导出失败：" + (r.stderr or r.stdout)[-400:])
            return
        before, after = tree(backup / "data"), tree(original)
        changed = [k for k in set(before) & set(after)
                   if _canonical(before[k]) != _canonical(after[k])]
        added, gone = sorted(set(after) - set(before)), sorted(set(before) - set(after))
        ok = not (changed or added or gone)
        report("web/data 生物学数字可由当前代码复现（忽略耗时/时间戳）", ok,
               f"变化：{', '.join(changed) or '无'}；新增：{', '.join(added) or '无'}；"
               f"消失：{', '.join(gone) or '无'}" + ("" if ok else "（已恢复原文件）"))
    finally:
        for f in tree(original).values():
            f.unlink()
        shutil.copytree(backup / "data", original, dirs_exist_ok=True)
        shutil.rmtree(backup, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tests", action="store_true", help="只查发布物，不跑 152 用例")
    ap.add_argument("--reexport", action="store_true", help="重跑导出，验证 web/data 可复现")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"# release check @ {ROOT}")
    if not args.skip_tests:
        check_tests()
    check_dist_sync()
    check_scenarios()
    check_secrets()
    if args.reexport:
        check_reexport()

    failed = [n for s, n in results if s == "FAIL"]
    print(f"\n{len(results) - len(failed)}/{len(results)} 项通过"
          + (f"，未通过：{', '.join(failed)}" if failed else "，可以发布"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
