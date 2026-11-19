# -*- coding: utf-8 -*-
"""让 pytest 与被直接执行的脚本都能 import 项目内的绝对包名（core / ai / app）。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
