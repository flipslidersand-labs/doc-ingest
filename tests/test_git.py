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

    def test_uncommitted_edits_and_deletions_are_ignored(self, tmp_path):
        """#313: HEAD~1 と作業ツリーではなく HEAD~1 と HEAD を比較する。"""
        repo = self._repo(tmp_path)
        for name in ("a.md", "b.md", "c.md"):
            (repo / name).write_text("x")
        self._commit(repo, "one", "a.md", "b.md", "c.md")
        (repo / "a.md").write_text("y")
        self._commit(repo, "two", "a.md")
        (repo / "b.md").write_text("uncommitted edit")
        (repo / "c.md").unlink()
        assert changed_files(repo) == [Path("a.md")]


class TestRunFromASubdirectory:
    """#354: git はリポルート相対のパスを返す。cwd がサブディレクトリだと、それを cwd 相対として
    開いて「存在しない」= 削除扱いになり、実在する文書の point を消していた。"""

    @staticmethod
    def _vault(tmp_path):
        """10-projects/sub/ を cwd にできる vault。直近のコミットで note.md と outside.md が変更される。"""
        repo = TestChangedFilesInARealRepo._repo(tmp_path)
        (repo / "10-projects" / "sub").mkdir(parents=True)
        (repo / "10-projects" / "note.md").write_text("## 節\n" + "内容 " * 40)
        (repo / "outside.md").write_text("x")
        TestChangedFilesInARealRepo._commit(repo, "one", "10-projects/note.md", "outside.md")
        (repo / "10-projects" / "note.md").write_text("## 節\n" + "追記 " * 40)
        (repo / "outside.md").write_text("y")
        TestChangedFilesInARealRepo._commit(repo, "two", "10-projects/note.md", "outside.md")
        return repo

    def test_changed_matching_returns_paths_that_exist_from_a_subdirectory(
        self, tmp_path, monkeypatch
    ):
        from core.ingest_pipeline import changed_matching

        repo = self._vault(tmp_path)
        monkeypatch.chdir(repo / "10-projects" / "sub")
        paths = changed_matching(lambda p: p.parts[0] == "10-projects")  # 絞り込みはルート相対
        assert [p.resolve() for p in paths] == [(repo / "10-projects" / "note.md").resolve()]
        assert all(p.exists() for p in paths)

    def test_a_file_changed_outside_the_cwd_subtree_is_still_found(self, tmp_path, monkeypatch):
        """`--relative` で cwd 配下だけに絞る案は採らない: 配下の外の変更 (と削除) を取りこぼす。"""
        from core.ingest_pipeline import changed_matching

        repo = self._vault(tmp_path)
        monkeypatch.chdir(repo / "10-projects" / "sub")
        paths = changed_matching(lambda p: p.name == "outside.md")
        assert [p.resolve() for p in paths] == [(repo / "outside.md").resolve()]

    def test_a_deleted_file_is_still_reported_as_missing_from_a_subdirectory(
        self, tmp_path, monkeypatch
    ):
        """本当に消えたファイルは、サブディレクトリからでも「存在しない」path として返る。"""
        from core.ingest_pipeline import changed_matching

        repo = self._vault(tmp_path)
        TestChangedFilesInARealRepo._commit(repo, "del", rm=["10-projects/note.md"])
        monkeypatch.chdir(repo / "10-projects" / "sub")
        paths = changed_matching(lambda p: p.suffix == ".md")
        assert [p.resolve() for p in paths] == [(repo / "10-projects" / "note.md").resolve()]
        assert not paths[0].exists()

    def test_from_the_repo_root_the_paths_stay_relative(self, tmp_path, monkeypatch):
        """post-commit フック (ルートで実行) の挙動は変えない。"""
        from core.ingest_pipeline import changed_matching

        repo = self._vault(tmp_path)
        monkeypatch.chdir(repo)
        paths = changed_matching(lambda p: p.suffix == ".md")
        assert sorted(paths) == [Path("10-projects/note.md"), Path("outside.md")]

    def test_ingest_notes_from_a_subdirectory_updates_the_note_and_deletes_nothing(
        self, tmp_path, monkeypatch, mocker
    ):
        from core.doc_key import _repo_info
        from core.ids import make_id
        from ingest.notes import ingest_notes

        _repo_info.cache_clear()
        repo = self._vault(tmp_path)
        monkeypatch.chdir(repo / "10-projects" / "sub")
        key = "r/10-projects/note.md"
        mocker.patch(
            "core.qdrant.ids_by_payload",
            side_effect=lambda coll, k, val: [make_id(f"{key}:0")] if val == key else [],
        )
        up = mocker.patch("core.qdrant.upsert")
        delete = mocker.patch("core.qdrant.delete_by_ids")
        mocker.patch("ingest.notes.distill_text", return_value="要約")

        ingest_notes()

        assert [p["file_path"] for p in up.call_args[0][1]] == [key]
        delete.assert_not_called()  # 既存の point は同じ id で置き換わる。消えてはいけない
        _repo_info.cache_clear()

    def test_unresolvable_root_returns_nothing_instead_of_guessing(
        self, tmp_path, monkeypatch, mocker, caplog
    ):
        """ルートが分からないとき、cwd 相対のまま返すと「全部削除」になる。空にして警告する。"""
        import logging

        from core.ingest_pipeline import changed_matching

        repo = self._vault(tmp_path)
        monkeypatch.chdir(repo / "10-projects" / "sub")
        mocker.patch("core.git.repo_root", return_value=None)
        with caplog.at_level(logging.WARNING, logger="core.git"):
            assert changed_matching(lambda p: True) == []
        assert any("repo root" in r.message for r in caplog.records)
