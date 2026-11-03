"""install-hooks.sh: the distributed SNIPPET must parse .env in a safe, non-evaluating way
(#80/#129) — never `source`/`set -a` it — for both the design hook and the --notes hook (#221)."""

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_HOOKS = REPO_ROOT / "scripts" / "install-hooks.sh"


def _extract_snippet() -> str:
    text = INSTALL_HOOKS.read_text(encoding="utf-8")
    match = re.search(r"<< 'SNIPPET'.*?\n(.*?)\nSNIPPET\n", text, re.DOTALL)
    assert match, "install-hooks.sh must define a heredoc SNIPPET block"
    return match.group(1)


class TestInstallHooksSnippet:
    def test_snippet_does_not_source_the_env_file(self):
        snippet = _extract_snippet()
        assert "source " not in snippet, (
            "the distributed snippet must not `source`/`. ` the .env file - "
            "that re-introduces #80's arbitrary command execution"
        )

    def test_snippet_does_not_execute_a_command_substitution_in_env_value(self, tmp_path):
        snippet = _extract_snippet()
        env_file = tmp_path / ".env"
        marker = tmp_path / "pwned"
        env_file.write_text(f"SAFE_VAR=hello\nEVIL=$(touch {marker})\n", encoding="utf-8")

        script = snippet.replace(
            '"$_DOC_INGEST_DIR/.venv/bin/doc-ingest" design 2>/dev/null || true',
            "true",  # skip the actual doc-ingest invocation, not under test here
        )
        subprocess.run(
            ["bash", "-c", script],
            check=True,
            env={"DOC_INGEST_DIR": str(tmp_path), "PATH": "/usr/bin:/bin"},
        )

        assert not marker.exists(), "a $(...) command substitution in .env must never execute"

    def test_snippet_exports_a_plain_key_value_line(self, tmp_path):
        snippet = _extract_snippet()
        env_file = tmp_path / ".env"
        env_file.write_text("SAFE_VAR=hello\n", encoding="utf-8")

        script = snippet.replace(
            '"$_DOC_INGEST_DIR/.venv/bin/doc-ingest" design 2>/dev/null || true',
            'echo "SAFE_VAR=$SAFE_VAR"',
        )
        result = subprocess.run(
            ["bash", "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env={"DOC_INGEST_DIR": str(tmp_path), "PATH": "/usr/bin:/bin"},
        )
        assert "SAFE_VAR=hello" in result.stdout


def _run_install(*args, home=None):
    import os

    env = {"PATH": "/usr/bin:/bin", "HOME": str(home or Path("/nonexistent"))}
    return subprocess.run(
        ["bash", str(INSTALL_HOOKS), *args],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **env},
    )


def _git_init(parent, name):
    d = parent / name
    d.mkdir()
    subprocess.run(["git", "-C", str(d), "init", "-q"], check=True, capture_output=True)
    return d


def _hook(repo):
    return repo / ".git" / "hooks" / "post-commit"


class TestNotesHook:
    """#221: README の手書きフック (.env を source) を install-hooks.sh --notes に一本化した。"""

    def test_notes_installs_a_hook_that_runs_notes_not_design(self, tmp_path):
        vault = _git_init(tmp_path, "vault")
        r = _run_install("--notes", str(vault))
        assert r.returncode == 0, r.stderr
        text = _hook(vault).read_text()
        assert '/doc-ingest" notes 2>/dev/null || true' in text
        assert '/doc-ingest" design' not in text
        assert "# doc-ingest-notes-hook" in text
        assert _hook(vault).stat().st_mode & 0o111  # 実行可能

    def test_notes_hook_never_executes_a_command_substitution_in_env(self, tmp_path):
        vault = _git_init(tmp_path, "vault")
        _run_install("--notes", str(vault))
        env_file, marker = tmp_path / ".env", tmp_path / "pwned"
        env_file.write_text(f"SAFE_VAR=hello\nEVIL=$(touch {marker})\n", encoding="utf-8")
        script = (
            _hook(vault)
            .read_text()
            .replace(
                '"$_DOC_INGEST_DIR/.venv/bin/doc-ingest" notes 2>/dev/null || true',
                'echo "SAFE=$SAFE_VAR"',
            )
        )
        r = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            check=False,
            env={"DOC_INGEST_DIR": str(tmp_path), "PATH": "/usr/bin:/bin"},
        )
        assert not marker.exists(), "a $(...) in .env must never execute"
        assert "SAFE=hello" in r.stdout

    def test_reinstall_is_a_no_op(self, tmp_path):
        vault = _git_init(tmp_path, "vault")
        _run_install("--notes", str(vault))
        before = _hook(vault).read_text()
        r = _run_install("--notes", str(vault))
        assert "already installed" in r.stdout
        assert _hook(vault).read_text() == before

    def test_design_and_notes_hooks_coexist_and_uninstall_independently(self, tmp_path):
        repo = _git_init(tmp_path, "vault")
        _run_install("--notes", str(repo))
        _run_install(str(repo))  # design
        text = _hook(repo).read_text()
        assert text.count('/doc-ingest" design') == 1 and text.count('/doc-ingest" notes') == 1

        _run_install("--notes", "--uninstall", str(repo))
        text = _hook(repo).read_text()
        assert text.count('/doc-ingest" design') == 1 and text.count('/doc-ingest" notes') == 0

        _run_install("--uninstall", str(repo))
        assert '/doc-ingest" design' not in _hook(repo).read_text()

    def test_a_legacy_hand_written_hook_that_sources_env_is_not_touched(self, tmp_path):
        vault = _git_init(tmp_path, "vault")
        legacy = (
            '#!/usr/bin/env bash\n_D="$HOME/projects/doc-ingest"\n'
            '[ -f "$_D/.env" ] && { set -a; . "$_D/.env"; set +a; }\n'
            'PYTHONPATH="$_D" "$_D/.venv/bin/doc-ingest" notes 2>/dev/null || true\n'
        )
        _hook(vault).write_text(legacy)
        r = _run_install("--notes", str(vault))
        assert "legacy" in r.stderr and "remove" in r.stderr
        assert _hook(vault).read_text() == legacy  # 追記も書き換えもしない (二重実行を避ける)

    def test_dry_run_writes_nothing(self, tmp_path):
        vault = _git_init(tmp_path, "vault")
        r = _run_install("--notes", "--dry-run", str(vault))
        assert "would create" in r.stdout and not _hook(vault).exists()

    def test_default_target_is_home_notes(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        _git_init(home, "notes")
        r = _run_install("--notes", "--dry-run", home=home)
        assert str(home / "notes") in r.stdout

    def test_a_non_git_path_is_skipped(self, tmp_path):
        r = _run_install("--notes", str(tmp_path))
        assert "not a git repo" in r.stdout


class TestReadmeDoesNotTeachTheUnsafePattern:
    def test_readme_never_sources_env_in_a_hook_recipe(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert '. "$_D/.env"' not in readme and "set -a" not in readme
