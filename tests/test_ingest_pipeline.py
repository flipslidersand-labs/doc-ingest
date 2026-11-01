"""core/ingest_pipeline.ingest_markdown_files の契約(payload 組み立てと順序)。"""

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
