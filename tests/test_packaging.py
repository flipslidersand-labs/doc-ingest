"""#296: console script の cli.py(ルート直下)が配布物に入ること。"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_entry_point_module_is_packaged():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    module = cfg["project"]["scripts"]["doc-ingest"].split(":")[0]
    setuptools = cfg["tool"]["setuptools"]
    top_packages = setuptools.get("packages", {}).get("find", {}).get("include", [])
    packaged = module in setuptools.get("py-modules", []) or any(
        p.rstrip("*") == module for p in top_packages
    )
    assert (ROOT / f"{module}.py").exists()
    assert packaged, f"{module}.py は packages.find にも py-modules にも無く、wheel に入らない"
