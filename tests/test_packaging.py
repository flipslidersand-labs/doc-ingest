"""配布物の構成。

#296: console script のモジュールが wheel に入ること。
#346: wheel の最上位は doc_ingest パッケージだけにする(cli / core / ingest のような一般的な名前を
site-packages の最上位に置くと、他の配布物と衝突する)。
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "doc_ingest"


def _cfg():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_entry_point_module_is_inside_the_package():
    module = _cfg()["project"]["scripts"]["doc-ingest"].split(":")[0]
    assert module.split(".")[0] == PACKAGE
    assert (ROOT / (module.replace(".", "/") + ".py")).exists()


def test_only_the_package_is_packaged_at_top_level():
    setuptools = _cfg()["tool"]["setuptools"]
    assert setuptools.get("packages", {}).get("find", {}).get("include") == [f"{PACKAGE}*"]
    assert "py-modules" not in setuptools


def test_no_code_left_at_repo_top_level():
    for name in ("cli.py", "core", "ingest"):
        assert not (ROOT / name).exists(), f"{name} がリポ直下に残っている"
