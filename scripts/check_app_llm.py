# -*- coding: utf-8 -*-
"""无头浏览器验收：本地 Streamlit(8501) 的 AI 卡片是否真的走了 DeepSeek。

跑法：python -X utf8 scripts/check_app_llm.py [端口]
只做只读检查 + 截图，不改任何应用状态。
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from playwright.sync_api import sync_playwright

PORT = sys.argv[1] if len(sys.argv) > 1 else "8501"
URL = f"http://localhost:{PORT}"
OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    errors: list[str] = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(URL, wait_until="domcontentloaded")

    # 首屏要跑完 scanpy 才出图，慢机器上 numba 首次编译就要几十秒
    page.wait_for_selector(".sc-ai", timeout=300_000)
    page.wait_for_timeout(4_000)          # 让打字机与工作流跑完

    sidebar = page.locator('[data-testid="stSidebar"]')
    toggles = sidebar.locator("label").all_inner_texts()
    print("侧边栏开关：", [t for t in toggles if "大模型" in t or "自动解读" in t])

    card = page.locator(".sc-ai").first
    meta = card.locator(".sc-ai-meta").inner_text()
    predicted = card.locator(".sc-type").inner_text()
    body = card.inner_text()
    print("\nAI 卡片 meta：", meta)
    print("预测类型：", predicted)
    print("告警条数：", len(card.locator(".sc-warn, .sc-violation").all()))

    # 展开「知识图谱候选细胞类型」，确认表达层核实（簇内实测检出率）真的渲染出来了
    for exp in page.locator('[data-testid="stExpander"] summary').all():
        if "知识图谱候选" in exp.inner_text():
            exp.click()
            page.wait_for_timeout(1_500)
            break
    kg_text = page.locator('[data-testid="stExpander"]').filter(has_text="知识图谱候选").first.inner_text()
    print("\n知识图谱候选面板：\n" + kg_text[:400])
    print("含表达核实：", "support_ratio" in kg_text)
    print("命中本地缓存：", "本地缓存命中" in body)

    print("\n卡片正文（前 600 字）：\n" + body[:600])

    page.screenshot(path=str(OUT / "ai_deepseek_full.png"), full_page=True)
    card.scroll_into_view_if_needed()
    page.screenshot(path=str(OUT / "ai_deepseek_card.png"))
    print("\n截图：", OUT / "ai_deepseek_card.png", OUT / "ai_deepseek_full.png")
    print("浏览器 console 报错：", len(errors), errors[:3])
    browser.close()
