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
