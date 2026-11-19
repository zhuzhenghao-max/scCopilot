"""滚动截取页面下半部分：Streamlit 在内部容器里滚动，full_page 截图拿不到下面的内容。

用法：先在另一个终端跑 streamlit run app/main.py --server.port 8532，再执行本脚本。
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "cache" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1.5)
    pg.goto("http://localhost:8532/", wait_until="domcontentloaded")
    pg.wait_for_selector('text=【预测细胞类型】', timeout=420_000)
    pg.wait_for_timeout(6_000)          # 等打字机打完
    # 滚轮在这套页面上不生效，且"最大的可滚动元素"会选中侧边栏（它比正文长）。
    # 正文的滚动容器是 main.stMain，直接按 scrollTop 滚才拿得到页面下半部分。
    SCROLL = """(dy) => {
        const m = document.querySelector('main.stMain') || document.scrollingElement;
        m.scrollBy(0, dy);
        return m.scrollTop + '/' + (m.scrollHeight - m.clientHeight);
    }"""
    for i, step in enumerate((900, 900, 900), start=3):
        print("scrolled", pg.evaluate(SCROLL, step), flush=True)
        pg.wait_for_timeout(1_200)
        # animations="disabled"：打字机和 Plotly hover 动画会让截图一直等"页面稳定"直到超时
        pg.screenshot(path=str(OUT / f"0{i}_scroll.png"), animations="disabled", timeout=120_000)
        print("saved", f"0{i}_scroll.png", flush=True)
    print("buttons:", pg.evaluate(
        """() => [...document.querySelectorAll('[data-testid="stButton"] button, [data-testid="stDownloadButton"] button')]
                 .map(b => b.getAttribute('kind') + ' ' + getComputedStyle(b).backgroundColor
                       + ' / ' + b.innerText.slice(0, 8))"""
    ), flush=True)
    print("reasoning text len:", pg.evaluate(
        """() => { const f = document.querySelector('[data-testid="stIFrame"]'); return f ? f.contentDocument.body.innerText.length : -1 }"""
    ), flush=True)
    b.close()
