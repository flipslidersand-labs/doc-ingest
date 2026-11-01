"""#223: `-v` が自パッケージの DEBUG を出す / 第三者 (httpx など) のログは静かに保つ。

CLI の実際の出力 (stderr) を子プロセスで検証する。pytest の caplog は root にハンドラを足すので、
`propagate=False` や root レベルの不具合を隠してしまうため。
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# main の callback (= -v の処理本体) を直接呼んで、その後に各ロガーへ書く。
_SCRIPT = """
import logging, sys
sys.path.insert(0, {root!r})
import cli
cli.main.callback(verbose={verbose})
logging.getLogger("ingest.external").debug("OWN-DEBUG")
logging.getLogger("ingest.external").info("OWN-INFO")
logging.getLogger("core.retry").debug("CORE-DEBUG")
logging.getLogger("cli").debug("CLI-DEBUG")
logging.getLogger("httpx").info("HTTPX-INFO https://hooks.example.com/TOKEN-SECRET")
logging.getLogger("httpx").debug("HTTPX-DEBUG")
logging.getLogger("urllib3.connectionpool").debug("URLLIB3-DEBUG")
logging.getLogger("qdrant_client").info("QDRANT-INFO")
logging.getLogger("httpx").warning("HTTPX-WARNING")
"""


def _stderr(verbose: bool) -> str:
    r = subprocess.run(
        [sys.executable, "-I", "-c", _SCRIPT.format(root=str(REPO_ROOT), verbose=verbose)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env={"PATH": "/usr/bin:/bin"},
        cwd=REPO_ROOT,
    )
    assert r.returncode == 0, r.stderr
    return r.stderr


class TestVerboseFlag:
    def test_verbose_shows_debug_of_this_repos_packages(self):
        err = _stderr(verbose=True)
        for marker in ("OWN-DEBUG", "OWN-INFO", "CORE-DEBUG", "CLI-DEBUG"):
            assert marker in err, f"{marker} が -v で出ていない"

    def test_default_shows_info_but_not_debug(self):
        err = _stderr(verbose=False)
        assert "OWN-INFO" in err
        for marker in ("OWN-DEBUG", "CORE-DEBUG", "CLI-DEBUG"):
            assert marker not in err, f"{marker} が -v 無しで出ている"

    def test_third_party_info_and_debug_stay_quiet_even_with_verbose(self):
        """httpx は INFO でリクエスト URL (通知 URL のトークン等) を出す。-v でも出さない。"""
        for verbose in (False, True):
            err = _stderr(verbose=verbose)
            for marker in (
                "HTTPX-INFO",
                "TOKEN-SECRET",
                "HTTPX-DEBUG",
                "URLLIB3-DEBUG",
                "QDRANT-INFO",
            ):
                assert marker not in err, f"{marker} が出ている (verbose={verbose})"

    def test_third_party_warnings_are_still_shown(self):
        assert "HTTPX-WARNING" in _stderr(verbose=False)

    def test_output_format_is_unchanged(self):
        line = next(ln for ln in _stderr(verbose=True).splitlines() if "OWN-DEBUG" in ln)
        # "2026-11-02 00:00:00,000 DEBUG ingest.external: OWN-DEBUG"
        assert " DEBUG ingest.external: OWN-DEBUG" in line
        assert line[:4].isdigit() and line[4] == "-"


def test_every_module_uses_the_standard_logger():
    """独自の get_logger (propagate=False・INFO 固定) は廃止した。再導入しない。"""
    assert not (REPO_ROOT / "core" / "logging.py").exists()
    offenders = [
        str(p.relative_to(REPO_ROOT))
        for d in ("core", "ingest")
        for p in (REPO_ROOT / d).glob("*.py")
        if "get_logger" in p.read_text(encoding="utf-8")
    ]
    assert offenders == []
