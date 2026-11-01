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
        result = _make_result(0, "foo.py\0bar.md\0")
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


class TestChangedFilesInARealRepo:
    """#217: 実際の git の出力で検証する (モックでは quotepath とリネーム検出は再現できない)。"""

    @staticmethod
    def _repo(tmp_path):
        import subprocess as sp

        repo = tmp_path / "r"
        repo.mkdir()
        sp.run(["git", "-C", str(repo), "init", "-q"], check=True)
        return repo

    @staticmethod
    def _commit(repo, msg, *add, rm=(), mv=None):
        import subprocess as sp

        g = ["git", "-C", str(repo), "-c", "user.email=t@e.com", "-c", "user.name=t"]
        for f in add:
            sp.run([*g, "add", f], check=True)
        for f in rm:
            sp.run([*g, "rm", "-q", f], check=True)
        if mv:
            sp.run([*g, "mv", *mv], check=True)
        sp.run([*g, "commit", "-q", "-m", msg], check=True)

    def test_a_rename_lists_both_the_old_and_the_new_path(self, tmp_path):
        repo = self._repo(tmp_path)
        (repo / "old.md").write_text("\n".join(str(i) for i in range(40)))
        self._commit(repo, "one", "old.md")
        self._commit(repo, "rename", mv=("old.md", "new.md"))
        assert sorted(p.name for p in changed_files(repo)) == ["new.md", "old.md"]

    def test_a_deleted_file_is_listed(self, tmp_path):
        repo = self._repo(tmp_path)
        (repo / "keep.md").write_text("k")
        (repo / "gone.md").write_text("g")
        self._commit(repo, "one", "keep.md", "gone.md")
        self._commit(repo, "del", rm=["gone.md"])
        assert changed_files(repo) == [Path("gone.md")]

    def test_non_ascii_paths_come_back_unquoted_and_existing(self, tmp_path):
        repo = self._repo(tmp_path)
        (repo / "10-projects").mkdir()
        note = repo / "10-projects" / "日本語のノート.md"
        note.write_text("# メモ")
        (repo / "ascii.md").write_text("x")
        self._commit(repo, "one", "10-projects/日本語のノート.md", "ascii.md")
        note.write_text("# メモ\n追記")
        self._commit(repo, "two", "10-projects/日本語のノート.md")
        paths = changed_files(repo)
        assert paths == [Path("10-projects/日本語のノート.md")]  # 以前は "\\346\\227…" の引用符つき
        assert (repo / paths[0]).exists()

    def test_paths_with_spaces_and_newlines_survive(self, tmp_path):
        repo = self._repo(tmp_path)
        weird = ["with space.md", "new\nline.md"]
        for name in weird:
            (repo / name).write_text("x")
        self._commit(repo, "one", *weird)
        for name in weird:
            (repo / name).write_text("y")
        self._commit(repo, "two", *weird)
        assert sorted(str(p) for p in changed_files(repo)) == sorted(weird)
