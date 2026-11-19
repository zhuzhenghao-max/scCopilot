# -*- coding: utf-8 -*-
"""scripts/release_check.py 的回归测试：证明这道门禁真的会拦，而不是只会亮绿灯。

全部在临时沙箱里跑，不碰仓库里的 web/ 与 dist/，也不读真实 .env——
密钥用合成串，否则测试本身就成了把凭据写进发布物的动作。
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import release_check as rc  # noqa: E402

FAKE_KEY = "sk-FAKE-0123456789abcdef0123456789abcdef"


def _sandbox(tmp: Path) -> None:
    """造一棵最小发布树：web/（真数据副本）+ dist/（同步副本）+ 假 .env。"""
    shutil.copytree(_ROOT / "web" / "data", tmp / "web" / "data")
    shutil.copytree(tmp / "web", tmp / "dist")
    (tmp / ".env").write_text(f"LLM_API_KEY={FAKE_KEY}\nLLM_MODEL=deepseek-chat\n", encoding="utf-8")
    rc.ROOT, rc.WEB, rc.DIST = tmp, tmp / "web", tmp / "dist"
    rc.results.clear()


def _run(*checks) -> tuple[list[str], str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        for c in checks:
            c()
    text = out.getvalue()
    return [n for s, n in rc.results if s == "FAIL"], text


def _patch_json(tmp: Path, name: str, mutate) -> None:
    f = tmp / "web" / "data" / name
    d = json.loads(f.read_text(encoding="utf-8"))
    mutate(d)
    f.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


def test_release_check_passes_on_clean_tree():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _sandbox(tmp)
        failed, _ = _run(rc.check_dist_sync, rc.check_scenarios, rc.check_secrets)
        assert failed == [], failed


def test_release_check_flags_stale_dist():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _sandbox(tmp)
        (tmp / "dist" / "data" / "normal.json").write_text("{}", encoding="utf-8")
        (tmp / "web" / "data" / "extra.json").write_text("{}", encoding="utf-8")
        failed, _ = _run(rc.check_dist_sync)
        assert failed == ["发布副本 dist/ 与 web/ 同步"], failed


def test_release_check_flags_number_drift():
    """代码改了没重导出：manifest 说 2240 个细胞，qc 里却写了别的数。"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _sandbox(tmp)
        _patch_json(tmp, "normal.json", lambda d: d["qc"].__setitem__("n_cells_final", 2239))
        failed, text = _run(rc.check_scenarios)
        assert failed == ["静态站三个场景数字自洽 + 与闸门口径一致"], failed
        assert "n_cells" in text, text


def test_release_check_flags_gate_contradiction():
    """前端绿点说「证据充足」，闸门记录说 ok=False——这种自相矛盾必须拦。"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _sandbox(tmp)

        def flip(d):
            d["clusters"][0]["evidence_ok"] = False

        _patch_json(tmp, "normal.json", flip)
        failed, text = _run(rc.check_scenarios)
        assert failed, "翻转 evidence_ok 后门禁没有报错，说明这条检查是空的"
        assert "闸门口径" in text or "gate.ok" in text, text


def test_release_check_flags_secret_in_assets():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _sandbox(tmp)
        (tmp / "web" / "index.html").write_text(
            f"<html><script>const k='{FAKE_KEY}'</script></html>", encoding="utf-8")
        failed, _ = _run(rc.check_secrets)
        assert failed == ["发布物不含密钥（已核 1 个 .env 值）"], failed


def test_release_check_ignores_non_secret_env_values():
    """base url / 模型名出现在前端是正常的，不该被当成密钥命中。"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _sandbox(tmp)
        (tmp / ".env").write_text(
            "LLM_API_KEY=\nLLM_BASE_URL=https://api.deepseek.com/v1\nLLM_MODEL=deepseek-chat\n",
            encoding="utf-8")
        (tmp / "web" / "index.html").write_text(
            "<html>deepseek-chat https://api.deepseek.com/v1</html>", encoding="utf-8")
        failed, _ = _run(rc.check_secrets)
        assert failed == [], failed


def test_release_check_flags_missing_scenario_file():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _sandbox(tmp)
        (tmp / "web" / "data" / "resolution.json").unlink()
        failed, text = _run(rc.check_scenarios)
        assert failed, "场景文件被删后门禁仍全绿"
        assert "缺文件" in text, text


if __name__ == "__main__":
    def _main() -> int:
        import time
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
        failed = 0
        for name, fn in fns:
            t0 = time.perf_counter()
            try:
                fn()
                print(f"[PASS] {name:<50s} {time.perf_counter() - t0:6.2f}s")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                import traceback
                print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
        print(f"\n{len(fns) - failed}/{len(fns)} passed")
        return 1 if failed else 0

    raise SystemExit(_main())
