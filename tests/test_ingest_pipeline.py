"""core/ingest_pipeline.ingest_markdown_files の契約(payload 組み立てと順序)。"""

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
    mocker.patch("core.ingest_pipeline.ids_by_payload", return_value=[])
    mock_upsert = mocker.patch("core.ingest_pipeline.upsert")
    mocker.patch("core.ingest_pipeline.delete_by_ids")

    _run(md)

    coll, points = mock_upsert.call_args.args
    assert coll == "col"
    assert len(points) >= 1
    p = points[0]
    assert p["id"] == make_id(f"{md}:0")
    assert p["source"] == "label"
    assert p["file_path"] == str(md)
    assert p["git_sha"] == "sha123"
    assert p["ingested_at"] == "2026-11-01T00:00:00+00:00"
    assert p["text"].startswith("D:")
    assert "section" in p


def test_upsert_precedes_stale_delete(mocker, tmp_path):
    md = tmp_path / "a.md"
    md.write_text("## Section\n" + "content " * 20)
    mocker.patch("core.ingest_pipeline.ids_by_payload", return_value=["stale-id"])
    calls = mocker.Mock()
    mocker.patch("core.ingest_pipeline.upsert", calls.upsert)
    mocker.patch("core.ingest_pipeline.delete_by_ids", calls.delete)

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
    mocker.patch("core.ingest_pipeline.ids_by_payload", return_value=[])
    mock_upsert = mocker.patch("core.ingest_pipeline.upsert")
    mocker.patch("core.ingest_pipeline.delete_by_ids")

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
    mocker.patch("core.ingest_pipeline.ids_by_payload", return_value=[second_id, gone_id])
    mocker.patch("core.ingest_pipeline.upsert")
    mock_delete = mocker.patch("core.ingest_pipeline.delete_by_ids")

    ingest_markdown_files(
        [md], collection="c", source_label="l", distill_fn=_fail_on_second, git_sha="s", now="n"
    )

    assert mock_delete.call_args.args == ("c", [gone_id])


def test_all_success_returns_zero(mocker, tmp_path):
    md = _two_sections(tmp_path)
    mocker.patch("core.ingest_pipeline.ids_by_payload", return_value=[])
    mocker.patch("core.ingest_pipeline.upsert")
    mocker.patch("core.ingest_pipeline.delete_by_ids")
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
        "core.ingest_pipeline.ids_by_payload",
        side_effect=lambda coll, key, val: lookups.append((key, val)) or [],
    )
    up = mocker.patch("core.ingest_pipeline.upsert")
    mocker.patch("core.ingest_pipeline.delete_by_ids")

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
    mocker.patch("core.ingest_pipeline.ids_by_payload", return_value=[])
    up = mocker.patch("core.ingest_pipeline.upsert")
    mocker.patch("core.ingest_pipeline.delete_by_ids")
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
    mocker.patch("core.ingest_pipeline.ids_by_payload", return_value=[second_id, "gone"])
    mocker.patch("core.ingest_pipeline.upsert")
    delete = mocker.patch("core.ingest_pipeline.delete_by_ids")

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
        "core.ingest_pipeline.ids_by_payload",
        side_effect=lambda coll, key, val: ["old-point"] if val == old_key else [],
    )
    up = mocker.patch("core.ingest_pipeline.upsert")
    delete = mocker.patch("core.ingest_pipeline.delete_by_ids")
    mocker.patch("ingest.notes.distill_text", return_value="要約")

    from ingest.notes import ingest_notes

    ingest_notes()

    assert [p["file_path"] for p in up.call_args[0][1]] == ["vault/10-projects/改名したノート.md"]
    delete.assert_called_once_with("notes", ["old-point"])
    _repo_info.cache_clear()
