"""Regression test for #129: install-hooks.sh's distributed SNIPPET must
parse .env the same safe, non-evaluating way hooks/post-commit does (#80),
not `source`/`set -a` it."""

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
