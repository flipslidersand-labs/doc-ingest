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
