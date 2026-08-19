"""Tests for ingest/external.py"""
import httpx
import pytest
import respx


SOURCES = [
    {
        "url": "https://example.com/api",
        "type": "external-api",
        "check_interval": "weekly",
        "tags": ["test"],
        "last_etag": "",
        "last_synced": "",
    }
]


@pytest.fixture
def mock_sources(mocker, tmp_path):
    """Patch _load_sources / _save_sources to avoid touching the real yaml."""
    saved = []

    def fake_save(sources):
        saved.extend(sources)

    mocker.patch("ingest.external._load_sources", return_value=list(SOURCES))
    mocker.patch("ingest.external._save_sources", side_effect=fake_save)
    mocker.patch("ingest.external.delete_by_payload")
    mocker.patch("ingest.external.upsert")
    return saved


class TestSyncExternal:
    @respx.mock
    def test_304_skips_ingest(self, mock_sources, mocker):
        respx.get("https://example.com/api").mock(
            return_value=httpx.Response(304)
        )
        from ingest.external import sync_external
        sync_external()
        import ingest.external as ext
        ext.upsert.assert_not_called()

    @respx.mock
    def test_new_content_is_upserted(self, mock_sources, mocker, capsys):
        content = "## Section\n" + "x" * 200
        respx.get("https://example.com/api").mock(
            return_value=httpx.Response(200, text=content, headers={"etag": '"abc"'})
        )
        from ingest.external import sync_external
        sync_external()
        import ingest.external as ext
        ext.upsert.assert_called_once()
        captured = capsys.readouterr()
        assert "synced" in captured.out

    @respx.mock
    def test_dry_run_skips_upsert(self, mock_sources):
        respx.get("https://example.com/api").mock(
            return_value=httpx.Response(200, text="## S\n" + "y" * 200)
        )
        from ingest.external import sync_external
        sync_external(dry_run=True)
        import ingest.external as ext
        ext.upsert.assert_not_called()

    @respx.mock
    def test_retry_on_500(self, mock_sources, capsys):
        respx.get("https://example.com/api").mock(
            return_value=httpx.Response(500)
        )
        from ingest.external import sync_external
        sync_external()
        captured = capsys.readouterr()
        assert "retry" in captured.out or "failed" in captured.out

    @respx.mock
    def test_render_true_fetches_via_jina(self, mocker):
        """render: true → fetch URL is prefixed with Jina Reader base."""
        render_source = [
            {
                "url": "https://example.com/js-only",
                "type": "external-api",
                "render": True,
                "check_interval": "weekly",
                "tags": ["test"],
                "last_etag": "",
                "last_synced": "",
            }
        ]
        mocker.patch("ingest.external._load_sources", return_value=render_source)
        mocker.patch("ingest.external._save_sources")
        mocker.patch("ingest.external.delete_by_payload")
        mocker.patch("ingest.external.upsert")

        jina_url = "https://r.jina.ai/https://example.com/js-only"
        respx.get(jina_url).mock(
            return_value=httpx.Response(200, text="## Rendered\n" + "content " * 50)
        )

        from ingest.external import sync_external
        sync_external()

        import ingest.external as ext
        ext.upsert.assert_called_once()

    @respx.mock
    def test_render_true_empty_body_skips(self, mocker, capsys):
        """render: true + empty Jina response → skip without upsert."""
        render_source = [
            {
                "url": "https://example.com/js-only",
                "type": "external-api",
                "render": True,
                "check_interval": "weekly",
                "tags": ["test"],
                "last_etag": "",
                "last_synced": "",
            }
        ]
        mocker.patch("ingest.external._load_sources", return_value=render_source)
        mocker.patch("ingest.external._save_sources")
        mocker.patch("ingest.external.delete_by_payload")
        mocker.patch("ingest.external.upsert")

        jina_url = "https://r.jina.ai/https://example.com/js-only"
        respx.get(jina_url).mock(return_value=httpx.Response(200, text="   "))

        from ingest.external import sync_external
        sync_external()

        import ingest.external as ext
        ext.upsert.assert_not_called()
        assert "skipping" in capsys.readouterr().out
