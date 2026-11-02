"""core/doc_key: ファイルの識別子 <repo>/<リポ内の相対パス> (#216)。実際の git リポで検証する。"""

import subprocess

import pytest

from core.doc_key import DocKey, _repo_info, doc_key
from core.ids import make_id


def _git(cwd, *args):
    subprocess.run(
        ["git", "-C", str(cwd), "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        check=True,
        capture_output=True,
    )


def _repo(parent, name, files=("CLAUDE.md",)):
    root = parent / name
    root.mkdir()
    _git(root, "init", "-q")
    for f in files:
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text("# doc\n")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "init")
    return root


@pytest.fixture(autouse=True)
def _clear_cache():
    _repo_info.cache_clear()
    yield
    _repo_info.cache_clear()


def test_same_filename_in_two_repos_gets_different_keys_and_ids(tmp_path):
    a, b = _repo(tmp_path, "repoA"), _repo(tmp_path, "repoB")
    ka, kb = doc_key(a / "CLAUDE.md"), doc_key(b / "CLAUDE.md")
    assert ka == DocKey("repoA", "repoA/CLAUDE.md")
    assert kb == DocKey("repoB", "repoB/CLAUDE.md")
    assert make_id(f"{ka.key}:0") != make_id(f"{kb.key}:0")  # #216: 以前は同一 ID


def test_hook_relative_path_and_manual_absolute_path_are_the_same_key(tmp_path, monkeypatch):
    repo = _repo(tmp_path, "forge", files=("docs/design.md",))
    monkeypatch.chdir(repo)  # post-commit フックは各リポのルートで走る
    from pathlib import Path

    assert doc_key(Path("docs/design.md")) == doc_key(repo / "docs" / "design.md")
    assert doc_key(Path("docs/design.md")).key == "forge/docs/design.md"


def test_linked_worktree_resolves_to_the_main_repo(tmp_path):
    main = _repo(tmp_path, "forge", files=("docs/a.md",))
    wt = tmp_path / "wt-feature"
    _git(main, "worktree", "add", "-q", "-b", "feature", str(wt))
    assert doc_key(wt / "docs" / "a.md") == doc_key(main / "docs" / "a.md")
    assert doc_key(wt / "docs" / "a.md").repo == "forge"


def test_deleted_file_still_gets_a_key(tmp_path, monkeypatch):
    """#217: コミットで削除されたファイルも、同じ識別子で旧 point を引ける。"""
    repo = _repo(tmp_path, "forge", files=("docs/gone.md",))
    (repo / "docs" / "gone.md").unlink()
    monkeypatch.chdir(repo)
    from pathlib import Path

    assert doc_key(Path("docs/gone.md")).key == "forge/docs/gone.md"


def test_deleted_directory_still_gets_a_key(tmp_path, monkeypatch):
    repo = _repo(tmp_path, "forge", files=("docs/sub/gone.md",))
    import shutil

    shutil.rmtree(repo / "docs")
    monkeypatch.chdir(repo)
    from pathlib import Path

    assert doc_key(Path("docs/sub/gone.md")).key == "forge/docs/sub/gone.md"


def test_outside_any_git_repo_falls_back_to_the_absolute_path(tmp_path):
    f = tmp_path / "loose.md"
    f.write_text("x")
    assert doc_key(f) == DocKey("", str(f.resolve()))


def test_symlink_pointing_outside_the_repo_falls_back_to_absolute(tmp_path):
    repo = _repo(tmp_path, "forge")
    outside = tmp_path / "secret.md"
    outside.write_text("x")
    (repo / "link.md").symlink_to(outside)
    k = doc_key(repo / "link.md")
    assert k == DocKey("", str(outside.resolve()))  # リポ配下の識別子を名乗らない


class TestEscapesRepo:
    """#224: symlink でリポの外へ出るパスを見つける。リポの根は字句的に .git までさかのぼって決める。"""

    @staticmethod
    def _outside(tmp_path):
        d = tmp_path / "outside"
        d.mkdir()
        (d / "secret.md").write_text("DUMMY")
        return d

    def test_a_normal_file_in_the_repo_does_not_escape(self, tmp_path):
        from core.doc_key import escapes_repo

        repo = _repo(tmp_path, "forge", files=("docs/a.md",))
        assert escapes_repo(repo / "docs" / "a.md") is False

    def test_a_file_symlink_to_outside_escapes(self, tmp_path):
        from core.doc_key import escapes_repo

        repo = _repo(tmp_path, "forge")
        (repo / "link.md").symlink_to(self._outside(tmp_path) / "secret.md")
        assert escapes_repo(repo / "link.md") is True

    def test_a_symlinked_directory_to_outside_escapes(self, tmp_path):
        """親ディレクトリが symlink: 解決後の親はリポ内に無いので、git rev-parse では見抜けない。"""
        from core.doc_key import escapes_repo

        repo = _repo(tmp_path, "forge")
        (repo / "docs").symlink_to(self._outside(tmp_path))
        assert escapes_repo(repo / "docs" / "secret.md") is True

    def test_a_symlink_that_stays_inside_the_repo_does_not_escape(self, tmp_path):
        from core.doc_key import escapes_repo

        repo = _repo(tmp_path, "forge", files=("docs/real.md",))
        (repo / "docs" / "alias.md").symlink_to(repo / "docs" / "real.md")
        assert escapes_repo(repo / "docs" / "alias.md") is False  # 拒否は is_symlink の側が担う

    def test_a_path_outside_any_repo_is_not_flagged(self, tmp_path):
        from core.doc_key import escapes_repo

        loose = tmp_path / "loose.md"
        loose.write_text("x")
        assert escapes_repo(loose) is False

    def test_a_linked_worktree_is_a_repo_too(self, tmp_path):
        """worktree の .git は『ファイル』。lexists で見るので検出できる。"""
        from core.doc_key import escapes_repo

        main = _repo(tmp_path, "forge", files=("docs/a.md",))
        wt = tmp_path / "wt"
        _git(main, "worktree", "add", "-q", "-b", "feature", str(wt))
        (wt / "leak.md").symlink_to(self._outside(tmp_path) / "secret.md")
        assert escapes_repo(wt / "docs" / "a.md") is False
        assert escapes_repo(wt / "leak.md") is True

    def test_a_relative_path_from_the_repo_root_works(self, tmp_path, monkeypatch):
        from pathlib import Path

        from core.doc_key import escapes_repo

        repo = _repo(tmp_path, "forge")
        (repo / "link.md").symlink_to(self._outside(tmp_path) / "secret.md")
        monkeypatch.chdir(repo)  # フックは各リポのルートで走り、相対パスを受け取る
        assert escapes_repo(Path("link.md")) is True

    def test_a_missing_path_does_not_crash(self, tmp_path):
        from core.doc_key import escapes_repo

        repo = _repo(tmp_path, "forge")
        assert escapes_repo(repo / "docs" / "gone.md") is False

    def test_a_repo_reached_through_a_symlinked_path_is_not_flagged(self, tmp_path):
        """~/projects/forge -> /data/forge のようにリポ自体が symlink 経由で置かれていても、
        正常なファイルを『リポの外』と誤判定しない (全ファイルの取り込みが止まるため)。"""
        from core.doc_key import escapes_repo

        real = _repo(tmp_path, "data_forge", files=("docs/a.md",))
        home = tmp_path / "home"
        home.mkdir()
        (home / "forge").symlink_to(real)
        assert escapes_repo(home / "forge" / "docs" / "a.md") is False

    def test_a_leak_symlink_inside_a_repo_reached_through_a_symlink_is_still_caught(self, tmp_path):
        from core.doc_key import escapes_repo

        real = _repo(tmp_path, "data_forge")
        (real / "leak.md").symlink_to(self._outside(tmp_path) / "secret.md")
        home = tmp_path / "home"
        home.mkdir()
        (home / "forge").symlink_to(real)
        assert escapes_repo(home / "forge" / "leak.md") is True
