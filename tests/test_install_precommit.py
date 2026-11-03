"""scripts/install-precommit.sh: core.hooksPath 方式 (#189)。

`pre-commit install` の再実行・別 worktree・.git/hooks の既存フック(secret-sweep の pre-push など)の
どれがあっても、実行結果の記録が止まらず、既存のフックも止まらないこと。実リポには触れず一時リポで試す。
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL = REPO_ROOT / "scripts" / "install-precommit.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("pre-commit") is None or shutil.which("git") is None,
    reason="pre-commit / git が必要",
)

CONFIG = """\
repos:
  - repo: local
    hooks:
      - id: tcheck
        name: tcheck
        entry: "true"
        language: system
        pass_filenames: false
        always_run: true
"""


@pytest.fixture
def env(tmp_path):
    e = {
        k: v for k, v in os.environ.items() if not k.startswith("GIT_")
    }  # GIT_DIR 漏れで外側を壊さない
    e.update(
        HOME=str(tmp_path / "home"),
        PRECOMMIT_METRICS_LOG=str(tmp_path / "events.jsonl"),
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.com",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.com",
    )
    (tmp_path / "home").mkdir()
    return e


def run(cmd, cwd, env, check=True):
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, check=check)


@pytest.fixture
def repo(tmp_path, env):
    r = tmp_path / "repo"
    r.mkdir()
    run(["git", "init", "-q", "-b", "main"], r, env)
    (r / ".pre-commit-config.yaml").write_text(CONFIG)
    (r / "a.txt").write_text("a")
    run(["git", "add", "."], r, env)
    return r


def install(repo, env):
    return run(["bash", str(INSTALL)], repo, env)


def events(env):
    log = Path(env["PRECOMMIT_METRICS_LOG"])
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def commit(repo, env, msg="c"):
    return run(["git", "commit", "-q", "-m", msg, "--allow-empty"], repo, env)


class TestInstall:
    def test_sets_hookspath_to_an_absolute_dir_inside_git(self, repo, env):
        install(repo, env)

        path = run(["git", "config", "core.hooksPath"], repo, env).stdout.strip()
        assert path == str(repo / ".git" / "managed-hooks")
        assert os.access(Path(path) / "pre-commit", os.X_OK)

    def test_is_idempotent(self, repo, env):
        install(repo, env)
        first = (repo / ".git" / "managed-hooks" / "pre-commit").read_text()

        install(repo, env)

        assert (repo / ".git" / "managed-hooks" / "pre-commit").read_text() == first
        assert (
            run(["git", "config", "--get-all", "core.hooksPath"], repo, env).stdout.count("\n") == 1
        )

    def test_commit_is_recorded(self, repo, env):
        install(repo, env)

        commit(repo, env)

        assert [(e["hook"], e["result"]) for e in events(env)] == [("tcheck", "pass")]

    def test_pre_commit_install_is_refused_and_logging_keeps_working(self, repo, env):
        """#189 の本題: pre-commit install を再実行してもラッパーが外れず、記録が止まらない。"""
        install(repo, env)
        before = (repo / ".git" / "managed-hooks" / "pre-commit").read_text()

        again = run(["pre-commit", "install"], repo, env, check=False)

        assert again.returncode != 0
        assert (repo / ".git" / "managed-hooks" / "pre-commit").read_text() == before
        assert not (repo / ".git" / "hooks" / "pre-commit").exists()
        commit(repo, env)
        assert len(events(env)) == 1

    def test_other_worktrees_behave_the_same(self, repo, env, tmp_path):
        commit(repo, env, "base")
        install(repo, env)
        wt = tmp_path / "wt"
        run(["git", "worktree", "add", "-q", "-b", "feature", str(wt)], repo, env)

        commit(wt, env, "in worktree")

        assert [e["result"] for e in events(env)] == ["pass"]

    def test_a_branch_without_the_scripts_still_gets_the_hook(self, repo, env, tmp_path):
        """hooksPath は作業ツリーでなく .git を指すので、スクリプトの無い旧ブランチでもフックが消えない。"""
        commit(repo, env, "base")
        install(repo, env)
        run(["git", "checkout", "-q", "--orphan", "old"], repo, env)

        commit(repo, env, "on an orphan branch")

        assert len(events(env)) == 1

    def test_without_pre_commit_on_path_the_commit_still_goes_through(self, repo, env):
        install(repo, env)
        no_pc = dict(
            env,
            PATH=os.pathsep.join(
                p for p in env["PATH"].split(os.pathsep) if not (Path(p) / "pre-commit").exists()
            ),
        )

        commit(repo, no_pc)  # check=True: 失敗すれば例外

        assert events(env) == []


def _git_version() -> tuple[int, ...]:
    out = subprocess.run(["git", "--version"], capture_output=True, text=True, check=False).stdout
    return tuple(int(n) for n in re.findall(r"\d+", out)[:2])


@pytest.mark.skipif(_git_version() < (2, 36), reason="git hook run は git 2.36 以降")
class TestForwarding:
    """core.hooksPath で .git/hooks のフックが止まらない。"""

    def _hook(self, repo, name, body):
        h = repo / ".git" / "hooks" / name
        h.write_text("#!/bin/sh\n" + body)
        h.chmod(0o755)

    def test_hook_in_dot_git_hooks_still_runs_with_its_arguments(self, repo, env, tmp_path):
        self._hook(repo, "pre-push", f'echo "$@" > {tmp_path}/pushed\n')
        install(repo, env)

        run(["git", "hook", "run", "pre-push", "--", "origin", "url"], repo, env)

        assert (tmp_path / "pushed").read_text().strip() == "origin url"

    def test_a_failing_hook_in_dot_git_hooks_still_blocks(self, repo, env):
        """secret-sweep のような pre-push の拒否 (exit 1) が転送でも効く。"""
        self._hook(repo, "pre-push", "exit 1\n")
        install(repo, env)

        r = run(["git", "hook", "run", "pre-push", "--", "origin", "url"], repo, env, check=False)

        assert r.returncode == 1

    def test_missing_hook_is_a_no_op(self, repo, env):
        install(repo, env)

        r = run(["git", "hook", "run", "--ignore-missing", "pre-push"], repo, env, check=False)

        assert r.returncode == 0


class TestMigration:
    def test_old_wrapper_is_removed_but_a_plain_pre_commit_hook_is_not(self, repo, env):
        hooks = repo / ".git" / "hooks"
        hooks.mkdir(exist_ok=True)
        (hooks / "pre-commit").write_text("#!/bin/sh\n# precommit-metrics-wrapper\n")
        (hooks / "pre-commit.real").write_text("x")
        (hooks / "pclog.sh").write_text("x")
        (hooks / "pre-push").write_text("#!/bin/sh\n# secret-sweep\n")

        install(repo, env)

        assert not (hooks / "pre-commit").exists()
        assert not (hooks / "pre-commit.real").exists()
        assert not (hooks / "pclog.sh").exists()
        assert (hooks / "pre-push").exists()  # 他のフックは残す

    def test_a_pre_commit_hook_not_made_by_us_is_left_alone(self, repo, env):
        hooks = repo / ".git" / "hooks"
        hooks.mkdir(exist_ok=True)
        (hooks / "pre-commit").write_text("#!/bin/sh\n# someone else's hook\n")

        install(repo, env)

        assert "someone else" in (hooks / "pre-commit").read_text()
