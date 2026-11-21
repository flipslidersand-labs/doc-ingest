"""escapes_repo: symlink ディレクトリを ``..`` で戻るパスもリポ外として扱う (#327)。"""

import subprocess
from pathlib import Path

import pytest

from core.doc_key import _repo_info, escapes_repo
from core.ingest_pipeline import ingest_changed_or_file, ingest_markdown_files
from tests.conftest import override_settings

SECRET = "TOP-SECRET-KEY-MATERIAL"


@pytest.fixture
def env(tmp_path):
    """tmp/repo (git) 内の link -> tmp/elsewhere/inner。link/../secret.md は tmp/elsewhere/secret.md。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True, capture_output=True)
    inner = tmp_path / "elsewhere" / "inner"
    inner.mkdir(parents=True)
    (tmp_path / "elsewhere" / "secret.md").write_text("## S\n" + f"{SECRET} genuine content " * 40)
    (repo / "link").symlink_to(inner)
    yield repo
    _repo_info.cache_clear()


def test_dotdot_after_an_outward_symlink_dir_escapes(env):
    assert escapes_repo(env / "link" / ".." / "secret.md") is True


def test_relative_dotdot_after_an_outward_symlink_dir_escapes(env, monkeypatch):
    monkeypatch.chdir(env)
    assert escapes_repo(Path("link/../secret.md")) is True


def test_dotdot_inside_the_repo_does_not_escape(env):
    (env / "docs").mkdir()
    (env / "a.md").write_text("x")
    assert escapes_repo(env / "docs" / ".." / "a.md") is False


def test_dotdot_after_a_symlink_dir_that_stays_in_the_repo_does_not_escape(env):
    (env / "a" / "b").mkdir(parents=True)
    (env / "in").symlink_to(env / "a" / "b")
    (env / "a" / "x.md").write_text("x")
    assert escapes_repo(env / "in" / ".." / "x.md") is False


def _distill_seen(seen):
    return lambda t: seen.append(t) or "summary"


def test_ingesting_such_a_path_does_not_send_the_content_to_the_llm(env, mocker):
    mocker.patch("core.qdrant.upsert")
    mocker.patch("core.qdrant.delete_by_ids")
    mocker.patch("core.qdrant.ids_by_payload", return_value=[])
    seen: list[str] = []
    ingest_markdown_files(
        [env / "link" / ".." / "secret.md"],
        collection="design",
        source_label="d",
        distill_fn=_distill_seen(seen),
        git_sha="s",
        now="n",
    )
    assert not any(SECRET in t for t in seen)


def test_file_option_with_widened_base_dir_does_not_ingest_it(env, mocker, monkeypatch):
    mocker.patch("core.ingest_pipeline.head_sha", return_value="abc1234")
    mocker.patch("core.qdrant.upsert")
    mocker.patch("core.qdrant.delete_by_ids")
    mocker.patch("core.qdrant.ids_by_payload", return_value=[])
    override_settings(doc_ingest_base_dir=str(env.parent))
    monkeypatch.chdir(env)
    seen: list[str] = []
    ingest_changed_or_file(
        "link/../secret.md",
        changed=list,
        collection="design",
        source_label="d",
        distill_fn=_distill_seen(seen),
        nothing_msg="none",
    )
    assert not any(SECRET in t for t in seen)
