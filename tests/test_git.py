"""Tests for core/git.py – changed_files and head_sha error handling."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.git import changed_files, head_sha


def _make_result(returncode: int, stdout: str = "", stderr: str = "") -> MagicMock:
    r = MagicMock(spec=subprocess.CompletedProcess)
    r.returncode = returncode
    r.stdout = stdout
    r.stderr = stderr
    return r


class TestChangedFiles:
    def test_success_returns_paths(self):
        result = _make_result(0, "foo.py\nbar.md\n")
        with patch("subprocess.run", return_value=result) as mock_run:
            paths = changed_files("/repo")
        mock_run.assert_called_once()
        assert paths == [Path("foo.py"), Path("bar.md")]

    def test_error_returns_empty_list(self):
        """On git diff failure, must return [] – not fall back to ls-files."""
        result = _make_result(128, "", "fatal: bad revision 'HEAD~1'")
        with patch("subprocess.run", return_value=result) as mock_run:
            paths = changed_files("/repo")
        # ls-files must NOT be called
        assert mock_run.call_count == 1
        assert paths == []

    def test_error_emits_warning(self, caplog):
        import logging

        result = _make_result(128, "", "fatal: bad revision 'HEAD~1'")
        with (
            patch("subprocess.run", return_value=result),
            caplog.at_level(logging.WARNING, logger="core.git"),
        ):
            changed_files("/repo")
        assert any("returncode=128" in r.message for r in caplog.records)

    def test_empty_output_returns_empty_list(self):
        result = _make_result(0, "")
        with patch("subprocess.run", return_value=result):
            assert changed_files() == []


class TestHeadSha:
    def test_success_returns_sha(self):
        result = _make_result(0, "abc1234\n")
        with patch("subprocess.run", return_value=result):
            assert head_sha("/repo") == "abc1234"

    def test_error_returns_empty_string(self):
        result = _make_result(128, "", "fatal: not a git repository")
        with patch("subprocess.run", return_value=result):
            assert head_sha("/repo") == ""

    def test_error_emits_warning(self, caplog):
        import logging

        result = _make_result(128, "", "fatal: not a git repository")
        with (
            patch("subprocess.run", return_value=result),
            caplog.at_level(logging.WARNING, logger="core.git"),
        ):
            head_sha("/repo")
        assert any("returncode=128" in r.message for r in caplog.records)
