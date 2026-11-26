"""install-hooks.sh: the distributed SNIPPET must parse .env in a safe, non-evaluating way
(#80/#129) — never `source`/`set -a` it — for both the design hook and the --notes hook (#221).
It must also point at the checkout that ran the installer and log launch failures (#356)."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_HOOKS = REPO_ROOT / "scripts" / "install-hooks.sh"


def _extract_snippet() -> str:
    text = INSTALL_HOOKS.read_text(encoding="utf-8")
    match = re.search(r"<< 'SNIPPET'.*?\n(.*?)\nSNIPPET\n", text, re.DOTALL)
    assert match, "install-hooks.sh must define a heredoc SNIPPET block"
    return match.group(1)


def _stub(root: Path, body: str) -> Path:
    """Stand-in for `.venv/bin/doc-ingest` under `root` (the real CLI is never started)."""
    exe = root / ".venv" / "bin" / "doc-ingest"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    exe.chmod(0o755)
    return exe


def _clean_env(home: Path, **extra: str) -> dict[str, str]:
    """Minimal env: no os.environ (so no real DOC_INGEST_DIR / XDG_STATE_HOME / HOME leaks)."""
    return {"PATH": "/usr/bin:/bin", "HOME": str(home), **extra}


def _run_snippet(tmp_path: Path, stub_body: str, **extra: str) -> subprocess.CompletedProcess:
    """Run the raw SNIPPET with DOC_INGEST_DIR=tmp_path, whose .venv holds a stub CLI."""
    _stub(tmp_path, stub_body)
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return subprocess.run(
        ["bash", "-c", _extract_snippet()],
        check=True,
        capture_output=True,
        text=True,
        env=_clean_env(home, DOC_INGEST_DIR=str(tmp_path), **extra),
    )


class TestInstallHooksSnippet:
    def test_snippet_does_not_source_the_env_file(self):
        snippet = _extract_snippet()
        assert "source " not in snippet, (
            "the distributed snippet must not `source`/`. ` the .env file - "
            "that re-introduces #80's arbitrary command execution"
        )

    def test_snippet_does_not_execute_a_command_substitution_in_env_value(self, tmp_path):
        env_file = tmp_path / ".env"
        marker = tmp_path / "pwned"
        env_file.write_text(f"SAFE_VAR=hello\nEVIL=$(touch {marker})\n", encoding="utf-8")

        _run_snippet(tmp_path, "true")  # stub CLI: the real doc-ingest is not under test here

        assert not marker.exists(), "a $(...) command substitution in .env must never execute"

    def test_snippet_exports_a_plain_key_value_line(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("SAFE_VAR=hello\n", encoding="utf-8")

        result = _run_snippet(tmp_path, 'echo "SAFE_VAR=$SAFE_VAR"')
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
        assert '/doc-ingest" notes ' in text
        assert '/doc-ingest" design' not in text
        assert "# doc-ingest-notes-hook" in text
        assert _hook(vault).stat().st_mode & 0o111  # 実行可能

    def test_notes_hook_never_executes_a_command_substitution_in_env(self, tmp_path):
        vault = _git_init(tmp_path, "vault")
        _run_install("--notes", str(vault))
        env_file, marker = tmp_path / ".env", tmp_path / "pwned"
        env_file.write_text(f"SAFE_VAR=hello\nEVIL=$(touch {marker})\n", encoding="utf-8")
        _stub(tmp_path, 'echo "SAFE=$SAFE_VAR"')
        r = subprocess.run(
            ["bash", str(_hook(vault))],
            capture_output=True,
            text=True,
            check=False,
            env=_clean_env(tmp_path, DOC_INGEST_DIR=str(tmp_path)),
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


def _run_snippet_printing(tmp_path, env_text: str, names: list[str]) -> dict[str, str]:
    (tmp_path / ".env").write_text(env_text, encoding="utf-8")
    echo = "; ".join(f'printf "%s=<%s>\\n" {n} "${n}"' for n in names)
    out = _run_snippet(tmp_path, echo).stdout
    return dict(re.findall(r"^(\w+)=<(.*)>$", out, re.MULTILINE))


class TestSnippetQuotedValues:
    def test_matching_quotes_are_stripped(self, tmp_path):
        got = _run_snippet_printing(
            tmp_path,
            'A="dq value"\nB=\'sq value\'\nC=plain\nD="mismatch\'\nE=""\nF="\n',
            ["A", "B", "C", "D", "E", "F"],
        )
        assert got == {
            "A": "dq value",
            "B": "sq value",
            "C": "plain",
            "D": "\"mismatch'",
            "E": "",
            "F": '"',
        }

    def test_quoted_value_is_never_evaluated(self, tmp_path):
        marker = tmp_path / "pwned"
        got = _run_snippet_printing(
            tmp_path,
            f'A="$(touch {marker})"\nB=\'`touch {marker}`\'\nC="$HOME"\n',
            ["A", "B", "C"],
        )
        assert not marker.exists()
        assert got["C"] == "$HOME"
        assert got["A"] == f"$(touch {marker})"


def _clone_and_install(tmp_path: Path, clone_name: str = "clone") -> tuple[Path, Path, Path]:
    """Copy the installer into a fresh 'clone' (elsewhere than $HOME/projects/doc-ingest),
    install it into a throw-away git repo, and return (clone root, target repo, hook file)."""
    root = tmp_path / clone_name
    (root / "scripts").mkdir(parents=True)
    shutil.copy(INSTALL_HOOKS, root / "scripts" / "install-hooks.sh")
    repo = _git_init(tmp_path, "target")
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    r = subprocess.run(
        ["bash", str(root / "scripts" / "install-hooks.sh"), str(repo)],
        capture_output=True,
        text=True,
        check=False,
        env=_clean_env(home),
    )
    assert r.returncode == 0, r.stderr
    return root, repo, _hook(repo)


def _run_hook(tmp_path: Path, repo: Path, **extra: str) -> tuple[subprocess.CompletedProcess, Path]:
    """Run the installed hook as git would (cwd = repo). Returns (result, state dir)."""
    state = tmp_path / "state"
    r = subprocess.run(
        ["bash", str(_hook(repo))],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        env=_clean_env(tmp_path / "home", **{"XDG_STATE_HOME": str(state), **extra}),
    )
    return r, state / "doc-ingest" / "post-commit.log"


class TestHookTargetsTheInstallingCheckout:
    """#356: the hook used to hardcode $HOME/projects/doc-ingest."""

    @pytest.mark.parametrize("clone_name", ["clone", "my clone & 'x' $y"])
    def test_hook_runs_the_cli_of_the_checkout_that_installed_it(self, tmp_path, clone_name):
        root, repo, _ = _clone_and_install(tmp_path, clone_name)
        out = tmp_path / "stub.out"
        _stub(root, 'printf "%s|%s\\n" "$*" "$PYTHONPATH" >> "$STUB_OUT"')

        r, log = _run_hook(tmp_path, repo, STUB_OUT=str(out))

        assert r.returncode == 0, r.stderr
        assert out.read_text() == f"design|{root}\n"
        assert not log.exists(), "a successful run must not write a failure log"

    def test_the_clone_env_file_is_read_from_the_clone(self, tmp_path):
        root, repo, _ = _clone_and_install(tmp_path)
        (root / ".env").write_text("SAFE_VAR=from-clone\n", encoding="utf-8")
        out = tmp_path / "stub.out"
        _stub(root, 'echo "$SAFE_VAR" > "$STUB_OUT"')

        _run_hook(tmp_path, repo, STUB_OUT=str(out))

        assert out.read_text().strip() == "from-clone"

    def test_doc_ingest_dir_env_still_overrides_the_embedded_default(self, tmp_path):
        _, repo, _ = _clone_and_install(tmp_path)
        other = tmp_path / "other"
        out = tmp_path / "stub.out"
        _stub(other, 'echo other > "$STUB_OUT"')

        _run_hook(tmp_path, repo, DOC_INGEST_DIR=str(other), STUB_OUT=str(out))

        assert out.read_text().strip() == "other"

    def test_notes_hook_also_targets_the_installing_checkout(self, tmp_path):
        root = tmp_path / "clone"
        (root / "scripts").mkdir(parents=True)
        shutil.copy(INSTALL_HOOKS, root / "scripts" / "install-hooks.sh")
        vault = _git_init(tmp_path, "vault")
        (tmp_path / "home").mkdir()
        subprocess.run(
            ["bash", str(root / "scripts" / "install-hooks.sh"), "--notes", str(vault)],
            check=True,
            capture_output=True,
            env=_clean_env(tmp_path / "home"),
        )
        out = tmp_path / "stub.out"
        _stub(root, 'echo "$*" > "$STUB_OUT"')

        r, _ = _run_hook(tmp_path, vault, STUB_OUT=str(out))

        assert r.returncode == 0, r.stderr
        assert out.read_text().strip() == "notes"


class TestHookLogsLaunchFailures:
    """#356: `2>/dev/null || true` hid launch failures. Keep exit 0, but leave a trace."""

    def test_a_failing_cli_is_logged_and_the_hook_still_exits_zero(self, tmp_path):
        root, repo, _ = _clone_and_install(tmp_path)
        (root / ".env").write_text("ANTHROPIC_API_KEY=sk-should-not-leak\n", encoding="utf-8")
        _stub(root, "echo boom >&2; exit 3")

        r, log = _run_hook(tmp_path, repo)

        assert r.returncode == 0
        text = log.read_text()
        assert "exit=3" in text and "boom" in text
        assert str(repo) in text
        assert "sk-should-not-leak" not in text
        assert log.stat().st_mode & 0o077 == 0, "log must be private (0600)"
        assert log.parent.stat().st_mode & 0o077 == 0, "log dir must be private (0700)"

    def test_a_missing_executable_is_logged_and_the_hook_still_exits_zero(self, tmp_path):
        _, repo, _ = _clone_and_install(tmp_path)  # no .venv/bin/doc-ingest in the clone

        r, log = _run_hook(tmp_path, repo)

        assert r.returncode == 0
        text = log.read_text()
        assert "exit=127" in text and "No such file" in text

    def test_only_the_tail_of_stderr_is_logged(self, tmp_path):
        root, repo, _ = _clone_and_install(tmp_path)
        _stub(root, "i=1; while [ $i -le 50 ]; do echo line$i >&2; i=$((i+1)); done; exit 1")

        _, log = _run_hook(tmp_path, repo)

        text = log.read_text()
        assert "line50" in text and "line1\n" not in text

    def test_the_log_defaults_to_home_local_state(self, tmp_path):
        root, repo, _ = _clone_and_install(tmp_path)
        _stub(root, "exit 1")
        r = subprocess.run(
            ["bash", str(_hook(repo))],
            cwd=repo,
            capture_output=True,
            check=False,
            env=_clean_env(tmp_path / "home"),
        )
        assert r.returncode == 0
        log = tmp_path / "home" / ".local" / "state" / "doc-ingest" / "post-commit.log"
        assert "exit=1" in log.read_text()

    def test_the_log_is_trimmed_when_it_grows_past_64k(self, tmp_path):
        root, repo, _ = _clone_and_install(tmp_path)
        _stub(root, "echo boom >&2; exit 1")
        log = tmp_path / "state" / "doc-ingest" / "post-commit.log"
        log.parent.mkdir(parents=True)
        log.write_text("old entry\n" * 8000)

        _run_hook(tmp_path, repo)

        assert log.stat().st_size < 65536 and "boom" in log.read_text()

    def test_an_unwritable_log_location_does_not_break_the_hook(self, tmp_path):
        root, repo, _ = _clone_and_install(tmp_path)
        _stub(root, "exit 1")
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x")

        r, _ = _run_hook(tmp_path, repo, XDG_STATE_HOME=str(blocker))

        assert r.returncode == 0

    def test_the_failure_block_has_no_blank_line_so_uninstall_removes_all_of_it(self, tmp_path):
        _, repo, hook = _clone_and_install(tmp_path)
        assert "\n\n" not in hook.read_text().split("# doc-ingest-hook", 1)[1]

        _run_install("--uninstall", str(repo))

        assert "doc-ingest" not in hook.read_text()
