"""用本地 Playwright Chromium 给 scCopilot 页面拍真实渲染截图。

内置浏览器跑在云端，访问不到本机的 localhost，所以这里用本地 Chromium 起一个无头页面。
用法：先在另一个终端跑 streamlit run app/main.py --server.port 8532，再执行本脚本。
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8532/"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(__file__).resolve().parents[1] / "data" / "cache" / "screenshots")
OUT.mkdir(parents=True, exist_ok=True)

msgs, errors = [], []
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1.5)
    page.on("console", lambda m: msgs.append(f"{m.type}: {m.text[:200]}"))
    page.on("pageerror", lambda e: errors.append(str(e)[:300]))

    page.goto(URL, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_selector('div[data-testid="stAppViewContainer"]', timeout=60_000)
    print("shell ready", flush=True)

    # 首次加载要跑完整流水线（含 numba 预热），给足时间
    page.wait_for_timeout(3_000)
    t0 = time.time()
    try:
        page.wait_for_function(
            """() => {
                const t = document.body.innerText || '';
                return t.includes('【预测细胞类型】') || t.includes('三步跑通');
            }""",
            timeout=420_000,
        )
        print(f"app settled in {time.time()-t0:.1f}s", flush=True)
    except Exception as e:
        print("TIMEOUT waiting for app:", type(e).__name__, str(e)[:200], flush=True)

    page.wait_for_timeout(2_500)  # 等打字机/图表动画
    page.screenshot(path=str(OUT / "01_top.png"), full_page=False)
    page.screenshot(path=str(OUT / "02_full.png"), full_page=True)

    # 切到另一个簇：验证 AI 卡片重新出结论
    selects = page.locator('div[data-testid="stSelectbox"]')
    print("selectbox count:", selects.count(), flush=True)
    if selects.count() >= 1:
        try:
            picks = page.locator('div[data-testid="stSelectbox"] label')
            print("labels:", [x.inner_text() for x in picks.all][:5], flush=True)
        except Exception:
            pass

    txt = page.evaluate("() => document.body.innerText")
    (OUT / "page_text.txt").write_text(txt, encoding="utf-8")
    print("chars on page:", len(txt), flush=True)
    print("has AI title:", "【预测细胞类型】" in txt, "| has UMAP note:", "UMAP" in txt, flush=True)
    print("css injected:", bool(page.evaluate(
        "() => [...document.styleSheets].some(s => (s.ownerNode && s.ownerNode.textContent || '').includes('--sc-primary'))"
    )), flush=True)
    print("iframe count:", page.locator("iframe").count(), flush=True)
    print("console errors:", [m for m in msgs if m.startswith("error")][:5], flush=True)
    print("page errors:", errors[:5], flush=True)
    browser.close()
print("saved to", OUT, flush=True)
