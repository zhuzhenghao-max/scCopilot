# -*- coding: utf-8 -*-
"""一条命令跑完全部测试并汇总用例数：docs/使用手册.md 里的「N 个用例」以此为准。

每档起一个子进程，而不是 import 进来跑——各测试文件会改环境变量、写缓存目录、
加载真实 h5ad，同进程跑会互相污染，那样"全绿"就不可信了。
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOTAL_RE = re.compile(r"^(\d+)/(\d+) passed", re.MULTILINE)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEMO_H5AD = ROOT / "data" / "raw" / "demo_pbmc_sim.h5ad"
if not DEMO_H5AD.exists():
    # 演示数据不入库（.gitignore），但 app 的演示分支要读它：缺了就不是失败，是整档跳过。
    # 所以现场生成，让干净克隆和 CI 跑出来的用例数跟开发机一致。
    g = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_demo_h5ad.py")],
                       cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    print(g.stdout.strip() or "[setup] 演示数据生成失败：" + (g.stderr.strip()[-200:]))
    if g.returncode != 0:
        sys.exit(1)

passed = total = 0
bad: list[str] = []
for f in sorted((ROOT / "tests").glob("test_*.py")):
    r = subprocess.run([sys.executable, str(f)], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = TOTAL_RE.search(r.stdout or "")
    n, alln = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    passed += n
    total += alln
    print(f"{f.name:<26s} {n}/{alln} passed" + ("" if r.returncode == 0 else "   <-- FAILED"))
    if r.returncode != 0:
        bad.append(f.name)
        print("\n".join(l for l in (r.stdout or "").splitlines() if l.startswith("[FAIL]")))

print(f"\n合计 {passed}/{total} passed" + (f"，失败档：{', '.join(bad)}" if bad else "，全部通过"))
sys.exit(1 if bad or passed != total else 0)
