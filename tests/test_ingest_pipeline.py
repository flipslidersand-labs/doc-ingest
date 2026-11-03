"""core/ingest_pipeline.ingest_markdown_files の契約(payload 組み立てと順序)。"""

from pathlib import Path

import pytest

from core.distiller import DistillError
from core.ids import make_id
from core.ingest_pipeline import ingest_markdown_files


def _run(path, **kw):
    ingest_markdown_files(
        [path],
        collection="col",
        source_label="label",
        distill_fn=lambda t: f"D:{t[:3]}",
        git_sha="sha123",
        now="2026-11-01T00:00:00+00:00",
        **kw,
    )


def test_payload_fields_are_populated(mocker, tmp_path):
    md = tmp_path / "a.md"
    md.write_text("## Section\n" + "content " * 20)
    mocker.patch("core.qdrant.ids_by_payload", return_value=[])
    mock_upsert = mocker.patch("core.qdrant.upsert")
    mocker.patch("core.qdrant.delete_by_ids")

    _run(md)

    coll, points = mock_upsert.call_args.args
    assert coll == "col"
    assert len(points) >= 1
    p = points[0]
    assert p["id"] == make_id(f"{md}:0")
    assert p["source"] == "label"
    assert p["trust"] == "internal"
    assert p["file_path"] == str(md)
    assert p["git_sha"] == "sha123"
    assert p["ingested_at"] == "2026-11-01T00:00:00+00:00"
    assert p["text"].startswith("D:")
    assert "section" in p


def test_upsert_precedes_stale_delete(mocker, tmp_path):
    md = tmp_path / "a.md"
    md.write_text("## Section\n" + "content " * 20)
    mocker.patch("core.qdrant.ids_by_payload", return_value=["stale-id"])
    calls = mocker.Mock()
    mocker.patch("core.qdrant.upsert", calls.upsert)
    mocker.patch("core.qdrant.delete_by_ids", calls.delete)

    _run(md)

    assert [c[0] for c in calls.mock_calls] == ["upsert", "delete"]
    assert calls.delete.call_args.args == ("col", ["stale-id"])


def _two_sections(tmp_path):
    md = tmp_path / "a.md"
    md.write_text("## First\n" + "alpha " * 60 + "\n\n## Second\n" + "beta " * 60)
    return md


def _fail_on_second(text):
    if text.startswith("Second"):
        raise DistillError("both backends failed")
    return "ok"


def test_failed_chunk_is_not_embedded_and_counted(mocker, tmp_path):
    md = _two_sections(tmp_path)
    mocker.patch("core.qdrant.ids_by_payload", return_value=[])
    mock_upsert = mocker.patch("core.qdrant.upsert")
    mocker.patch("core.qdrant.delete_by_ids")

    failed = ingest_markdown_files(
        [md], collection="c", source_label="l", distill_fn=_fail_on_second, git_sha="s", now="n"
    )

    points = mock_upsert.call_args.args[1]
    assert failed == 1
    assert [p["section"] for p in points] == ["First"]
    assert all(p["text"] == "ok" for p in points)  # 原文断片が要約として混入しない


def test_failed_chunk_keeps_its_existing_point(mocker, tmp_path):
    """一時障害で要約に失敗したチャンクの既存 point を stale として消さない。"""
    md = _two_sections(tmp_path)
    second_id = make_id(f"{md}:1")
    gone_id = "no-longer-exists"
    mocker.patch("core.qdrant.ids_by_payload", return_value=[second_id, gone_id])
    mocker.patch("core.qdrant.upsert")
    mock_delete = mocker.patch("core.qdrant.delete_by_ids")

    ingest_markdown_files(
        [md], collection="c", source_label="l", distill_fn=_fail_on_second, git_sha="s", now="n"
    )

    assert mock_delete.call_args.args == ("c", [gone_id])


def test_all_success_returns_zero(mocker, tmp_path):
    md = _two_sections(tmp_path)
    mocker.patch("core.qdrant.ids_by_payload", return_value=[])
    mocker.patch("core.qdrant.upsert")
    mocker.patch("core.qdrant.delete_by_ids")
    assert (
        ingest_markdown_files(
            [md], collection="c", source_label="l", distill_fn=lambda t: "ok", git_sha="s", now="n"
        )
        == 0
    )


def test_distill_workers_default_is_serial(monkeypatch):
    """llama-server は長い入力の並列で全滅する(n_ctx 共有)ので既定は直列。"""
    import importlib

    import core.ingest_pipeline as mod

    monkeypatch.delenv("DISTILL_WORKERS", raising=False)
    assert importlib.reload(mod).DISTILL_WORKERS == 1
    monkeypatch.setenv("DISTILL_WORKERS", "3")
    assert importlib.reload(mod).DISTILL_WORKERS == 3
    monkeypatch.delenv("DISTILL_WORKERS")
    importlib.reload(mod)


def _git_repo(parent, name):
    import subprocess

    root = parent / name
    root.mkdir()
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True, capture_output=True)
    (root / "CLAUDE.md").write_text("## Section\n" + "content " * 30)
    return root


def test_same_filename_in_two_repos_does_not_collide_or_delete_each_other(mocker, tmp_path):
    """#216: 別リポの CLAUDE.md が同じ point を上書きし、stale 削除で互いに消し合っていた。"""
    from core.doc_key import _repo_info

    _repo_info.cache_clear()
    a, b = _git_repo(tmp_path, "repoA"), _git_repo(tmp_path, "repoB")
    lookups = []
    mocker.patch(
        "core.qdrant.ids_by_payload",
        side_effect=lambda coll, key, val: lookups.append((key, val)) or [],
    )
    up = mocker.patch("core.qdrant.upsert")
    mocker.patch("core.qdrant.delete_by_ids")

    for repo in (a, b):
        ingest_markdown_files(
            [repo / "CLAUDE.md"],
            collection="design",
            source_label="design-doc",
            distill_fn=lambda t: "ok",
            git_sha="s",
            now="n",
        )

    ids = [call.args[1][0]["id"] for call in up.call_args_list]
    assert ids[0] != ids[1]
    # ID は「どこに置いたか」ではなく識別子から作られる (フック=相対 / 手動=絶対でも同じになる前提)
    assert ids == [make_id("repoA/CLAUDE.md:0"), make_id("repoB/CLAUDE.md:0")]
    assert lookups == [("file_path", "repoA/CLAUDE.md"), ("file_path", "repoB/CLAUDE.md")]
    payloads = [call.args[1][0] for call in up.call_args_list]
    assert [(p["repo"], p["file_path"]) for p in payloads] == [
        ("repoA", "repoA/CLAUDE.md"),
        ("repoB", "repoB/CLAUDE.md"),
    ]
    _repo_info.cache_clear()


def test_file_outside_git_gets_an_explicit_no_repo_marker(mocker, tmp_path):
    """repo を空にすると、移行スクリプトが「未移行 (repo 無し)」と区別できない。"""
    md = tmp_path / "loose.md"
    md.write_text("## Section\n" + "content " * 30)
    mocker.patch("core.qdrant.ids_by_payload", return_value=[])
    up = mocker.patch("core.qdrant.upsert")
    mocker.patch("core.qdrant.delete_by_ids")
    ingest_markdown_files(
        [md], collection="c", source_label="l", distill_fn=lambda t: "ok", git_sha="s", now="n"
    )
    p = up.call_args.args[1][0]
    assert p["repo"] == "-" and p["file_path"] == str(md.resolve())


def test_failed_chunk_keeps_its_existing_point_inside_a_git_repo(mocker, tmp_path):
    """要約に失敗したチャンクの既存 point を守る ID も、識別子 (<repo>/<path>) から作る (#185 × #216)。"""
    from core.doc_key import _repo_info

    _repo_info.cache_clear()
    repo = _git_repo(tmp_path, "repoA")
    doc = repo / "docs.md"
    doc.write_text("## First\n" + "alpha " * 60 + "\n\n## Second\n" + "beta " * 60)
    second_id = make_id("repoA/docs.md:1")
    mocker.patch("core.qdrant.ids_by_payload", return_value=[second_id, "gone"])
    mocker.patch("core.qdrant.upsert")
    delete = mocker.patch("core.qdrant.delete_by_ids")

    ingest_markdown_files(
        [doc],
        collection="design",
        source_label="design-doc",
        distill_fn=_fail_on_second,
        git_sha="s",
        now="n",
    )

    assert delete.call_args.args == ("design", ["gone"])
    _repo_info.cache_clear()


def test_hook_flow_renaming_a_japanese_note_replaces_its_points(mocker, tmp_path, monkeypatch):
    """#217: post-commit フックの流れ全体。日本語名のノートをリネームすると、
    旧パスの point が消え、新パスの point が作られる (以前は日本語パスが引用符つきで読めず、
    リネームは新パスしか返らず、削除されたファイルは無視されていた)。"""
    import subprocess

    from core.doc_key import _repo_info

    _repo_info.cache_clear()
    vault = tmp_path / "vault"
    (vault / "10-projects").mkdir(parents=True)
    g = ["git", "-C", str(vault), "-c", "user.email=t@e.com", "-c", "user.name=t"]
    subprocess.run(["git", "-C", str(vault), "init", "-q"], check=True)
    (vault / "10-projects" / "日本語のノート.md").write_text("## 節\n" + "内容 " * 40)
    subprocess.run([*g, "add", "10-projects/日本語のノート.md"], check=True)
    subprocess.run([*g, "commit", "-q", "-m", "one"], check=True)
    subprocess.run(
        [*g, "mv", "10-projects/日本語のノート.md", "10-projects/改名したノート.md"], check=True
    )
    subprocess.run([*g, "commit", "-q", "-m", "rename"], check=True)
    monkeypatch.chdir(vault)  # フックは各リポのルートで走る

    old_key = "vault/10-projects/日本語のノート.md"
    mocker.patch(
        "core.qdrant.ids_by_payload",
        side_effect=lambda coll, key, val: ["old-point"] if val == old_key else [],
    )
    up = mocker.patch("core.qdrant.upsert")
    delete = mocker.patch("core.qdrant.delete_by_ids")
    mocker.patch("ingest.notes.distill_text", return_value="要約")

    from ingest.notes import ingest_notes

    ingest_notes()

    assert [p["file_path"] for p in up.call_args[0][1]] == ["vault/10-projects/改名したノート.md"]
    delete.assert_called_once_with("notes", ["old-point"])
    _repo_info.cache_clear()


class TestSymlinksAreNotIngested:
    """#224: フック経路のパスは assert_safe_path を通らず、exists()/read_text() が symlink を辿った。
    リポにマージされた docs/x.md -> ~/.ssh/... の中身が LLM と Qdrant に入りえた。"""

    SECRET = "DUMMY-SECRET-CONTENT"

    @pytest.fixture
    def env(self, mocker, tmp_path):
        import subprocess

        from core.doc_key import _repo_info

        _repo_info.cache_clear()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.md").write_text("## Key\n" + (self.SECRET + " ") * 60)
        repo = tmp_path / "repo"
        (repo / "docs").mkdir(parents=True)
        subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True, capture_output=True)
        (repo / "docs" / "real.md").write_text("## Real\n" + "genuine content " * 40)
        self.up = mocker.patch("core.qdrant.upsert")
        self.delete = mocker.patch("core.qdrant.delete_by_ids")
        self.ids = mocker.patch("core.qdrant.ids_by_payload", return_value=[])
        self.seen = []
        yield repo, outside
        _repo_info.cache_clear()

    def _run(self, *paths):
        return ingest_markdown_files(
            list(paths),
            collection="design",
            source_label="d",
            distill_fn=lambda t: self.seen.append(t) or "summary",
            git_sha="s",
            now="n",
        )

    def test_a_file_symlink_to_outside_is_skipped_and_its_content_never_reaches_the_llm(
        self, env, monkeypatch
    ):
        repo, outside = env
        (repo / "docs" / "leak.md").symlink_to(outside / "secret.md")
        monkeypatch.chdir(repo)
        self._run(Path("docs/leak.md"))
        assert not any(self.SECRET in t for t in self.seen)
        self.up.assert_not_called()

    def test_a_symlinked_directory_to_outside_is_skipped(self, env, monkeypatch):
        repo, outside = env
        (repo / "linked").symlink_to(outside)
        monkeypatch.chdir(repo)
        self._run(Path("linked/secret.md"))
        assert not any(self.SECRET in t for t in self.seen)
        self.up.assert_not_called()

    def test_a_symlink_inside_the_repo_is_skipped_too(self, env, monkeypatch):
        """リンク先 (real.md) は別途取り込まれる。symlink 自体は文書ではない。"""
        repo, _ = env
        (repo / "docs" / "alias.md").symlink_to(repo / "docs" / "real.md")
        monkeypatch.chdir(repo)
        self._run(Path("docs/alias.md"))
        self.up.assert_not_called()

    def test_a_normal_file_next_to_the_symlinks_is_still_ingested(self, env, monkeypatch):
        repo, outside = env
        (repo / "docs" / "leak.md").symlink_to(outside / "secret.md")
        monkeypatch.chdir(repo)
        self._run(Path("docs/leak.md"), Path("docs/real.md"))
        assert self.up.call_count == 1
        assert self.up.call_args[0][1][0]["file_path"] == "repo/docs/real.md"

    def test_points_of_a_path_that_became_a_symlink_are_removed(self, env, monkeypatch):
        """文書だったパスが symlink に置き換わったら、古い point は残さない。"""
        repo, outside = env
        (repo / "docs" / "now_link.md").symlink_to(outside / "secret.md")
        self.ids.return_value = ["old-1", "old-2"]
        monkeypatch.chdir(repo)
        self._run(Path("docs/now_link.md"))
        self.delete.assert_called_once_with("design", ["old-1", "old-2"])
        self.up.assert_not_called()

    def test_a_skip_is_logged(self, env, monkeypatch, caplog):
        import logging

        repo, outside = env
        (repo / "docs" / "leak.md").symlink_to(outside / "secret.md")
        monkeypatch.chdir(repo)
        with caplog.at_level(logging.WARNING, logger="core.ingest_pipeline"):
            self._run(Path("docs/leak.md"))
        assert "skipping" in caplog.text and "symlink" in caplog.text
        assert "docs/leak.md" in caplog.text  # 原因の symlink のパス (リポ内) が分かる
        assert str(outside) not in caplog.text  # リポ外の実パスは出さない

    def test_a_symlink_outside_any_git_repo_is_skipped_too(self, mocker, tmp_path):
        """git 管理外でも symlink そのものは取り込まない。"""
        mocker.patch("core.qdrant.ids_by_payload", return_value=[])
        up = mocker.patch("core.qdrant.upsert")
        mocker.patch("core.qdrant.delete_by_ids")
        real = tmp_path / "real.md"
        real.write_text("## S\n" + "content " * 40)
        link = tmp_path / "link.md"
        link.symlink_to(real)
        ingest_markdown_files(
            [link], collection="c", source_label="l", distill_fn=lambda t: "s", git_sha="s", now="n"
        )
        up.assert_not_called()


class TestSizeAndChunkLimits:
    """巨大なファイル・大量のチャンクで LLM 要約と埋め込みが際限なく走らない (#224)。"""

    def test_a_file_over_the_byte_limit_is_skipped_without_deleting_points(self, mocker, tmp_path):
        md = tmp_path / "huge.md"
        md.write_text("## S\n" + "content " * 50)
        mocker.patch("core.ingest_pipeline.MAX_LOCAL_BYTES", 10)
        mocker.patch("core.qdrant.ids_by_payload", return_value=["old"])
        mock_upsert = mocker.patch("core.qdrant.upsert")
        mock_delete = mocker.patch("core.qdrant.delete_by_ids")
        distill = mocker.Mock(return_value="d")

        ingest_markdown_files(
            [md], collection="col", source_label="l", distill_fn=distill, git_sha="s", now="n"
        )

        distill.assert_not_called()
        mock_upsert.assert_not_called()
        mock_delete.assert_not_called()

    def test_default_byte_limit_is_10_mib(self):
        from core.ingest_pipeline import MAX_LOCAL_BYTES

        assert MAX_LOCAL_BYTES == 10 * 1024 * 1024

    def test_a_file_exactly_at_the_byte_limit_is_ingested(self, mocker, tmp_path):
        md = tmp_path / "edge.md"
        md.write_text("## S\n" + "content " * 20)
        mocker.patch("core.ingest_pipeline.MAX_LOCAL_BYTES", md.stat().st_size)
        mocker.patch("core.qdrant.ids_by_payload", return_value=[])
        mock_upsert = mocker.patch("core.qdrant.upsert")
        mocker.patch("core.qdrant.delete_by_ids")

        _run(md)

        mock_upsert.assert_called_once()

    def test_chunks_over_the_cap_are_not_distilled(self, mocker, tmp_path):
        md = tmp_path / "many.md"
        md.write_text("".join(f"## S{i}\n" + "content " * 20 + "\n" for i in range(8)))
        mocker.patch("core.chunker.MAX_CHUNKS_PER_DOC", 3)
        mocker.patch("core.qdrant.ids_by_payload", return_value=[])
        mock_upsert = mocker.patch("core.qdrant.upsert")
        mocker.patch("core.qdrant.delete_by_ids")
        distill = mocker.Mock(return_value="d")

        ingest_markdown_files(
            [md], collection="col", source_label="l", distill_fn=distill, git_sha="s", now="n"
        )

        assert distill.call_count == 3
        assert len(mock_upsert.call_args.args[1]) == 3
