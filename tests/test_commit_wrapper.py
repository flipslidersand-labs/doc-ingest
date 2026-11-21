"""commit-wrapper.sh must handle non-ASCII / whitespace file names (NUL-delimited git output, #321)."""

import os
import shutil
import stat
import subprocess
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts" / "precommit-metrics"

# Fake `pre-commit`: first run rewrites every tracked *.txt (auto-fix) and fails; second run passes.
FAKE_PRE_COMMIT = """#!/usr/bin/env bash
state="$FAKE_STATE"
if [ ! -e "$state" ]; then
  touch "$state"
  for f in *.txt; do echo fixed >> "$f"; done
  echo "- files were modified by this hook"
  exit 1
fi
exit 0
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def test_restage_handles_japanese_and_space_names(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    names = ["日本語.txt", "with space.txt", "plain.txt"]
    for n in names:
        (repo / n).write_text("x\n", encoding="utf-8")
    (repo / ".pre-commit-config.yaml").write_text("repos: []\n")
    _git(repo, "add", "--", *names)

    hooks = tmp_path / "hooks"
    hooks.mkdir()
    for f in ("commit-wrapper.sh", "pclog.sh"):
        shutil.copy(SCRIPTS / f, hooks / f)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "pre-commit"
    fake.write_text(FAKE_PRE_COMMIT)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)

    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "FAKE_STATE": str(tmp_path / "state"),
        "PRECOMMIT_METRICS_LOG": str(tmp_path / "events.jsonl"),
        "HOME": str(tmp_path),
    }
    r = subprocess.run(
        ["bash", str(hooks / "commit-wrapper.sh")],
        check=False,
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    # every auto-fixed file was re-staged (no unstaged diff left), including non-ASCII/space names
    assert _git(repo, "-c", "core.quotepath=off", "diff", "--name-only") == ""
    staged = _git(repo, "-c", "core.quotepath=off", "diff", "--cached", "--name-only").splitlines()
    assert sorted(staged) == sorted(names)
