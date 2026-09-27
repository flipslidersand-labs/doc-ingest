"""Tests for ingest/external.py"""

from typing import ClassVar

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
    mocker.patch("ingest.external.ids_by_payload", return_value=[])
    mocker.patch("ingest.external.delete_by_ids")
    mocker.patch("ingest.external.upsert")
    return saved


class TestSyncExternal:
    @respx.mock
    def test_304_skips_ingest(self, mock_sources, mocker):
        respx.get("https://example.com/api").mock(return_value=httpx.Response(304))
        from ingest.external import sync_external

        sync_external()
        import ingest.external as ext

        ext.upsert.assert_not_called()

    @respx.mock
    def test_new_content_is_upserted(self, mock_sources, mocker, caplog):
        content = "## Section\n" + "x" * 200
        respx.get("https://example.com/api").mock(
            return_value=httpx.Response(200, text=content, headers={"etag": '"abc"'})
        )
        import logging

        from ingest.external import sync_external

        with caplog.at_level(logging.INFO, logger="ingest.external"):
            sync_external()
        import ingest.external as ext

        ext.upsert.assert_called_once()
        assert "synced" in caplog.text

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
    def test_retry_on_500(self, mock_sources, mocker, caplog):
        """500 response must trigger MAX_RETRIES attempts and log each retry."""
        respx.get("https://example.com/api").mock(return_value=httpx.Response(500))
        mocker.patch("ingest.external.time.sleep")  # avoid real waits
        import logging

        from ingest.external import MAX_RETRIES, sync_external

        with caplog.at_level(logging.DEBUG, logger="ingest.external"):
            sync_external()
        assert respx.calls.call_count == MAX_RETRIES, (
            f"expected {MAX_RETRIES} HTTP attempts, got {respx.calls.call_count}"
        )
        assert "retry" in caplog.text

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
        mocker.patch("ingest.external.ids_by_payload", return_value=[])
        mocker.patch("ingest.external.delete_by_ids")
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
    def test_4xx_recorded_as_failed_and_continues(self, mock_sources, mocker, caplog):
        """4xx response is recorded in failed_urls and loop continues (does not raise)."""
        respx.get("https://example.com/api").mock(return_value=httpx.Response(403))
        import logging

        from ingest.external import sync_external

        with caplog.at_level(logging.WARNING, logger="ingest.external"):
            sync_external()
        assert "failed" in caplog.text
        assert "403" in caplog.text
        import ingest.external as ext

        ext.upsert.assert_not_called()

    @respx.mock
    def test_discord_notify_called_on_sync(self, mock_sources, mocker):
        """Discord notify is called once after sync when webhook URL is set."""
        content = "## Section\n" + "x" * 200
        respx.get("https://example.com/api").mock(
            return_value=httpx.Response(200, text=content, headers={"etag": '"abc"'})
        )
        notify = mocker.patch("ingest.external._notify_discord")
        mocker.patch.dict("os.environ", {"DISCORD_WEBHOOK_URL": "https://discord.example/hook"})
        from ingest.external import sync_external

        sync_external()
        notify.assert_called_once()
        args = notify.call_args[0]
        assert args[0] == 1  # synced
        assert args[2] == []  # skipped
        assert args[3] == []  # failed

    @respx.mock
    def test_discord_not_called_on_dry_run(self, mock_sources, mocker):
        """Discord notify is NOT called during dry-run."""
        respx.get("https://example.com/api").mock(
            return_value=httpx.Response(200, text="## S\n" + "y" * 200)
        )
        notify = mocker.patch("ingest.external._notify_discord")
        from ingest.external import sync_external

        sync_external(dry_run=True)
        notify.assert_not_called()

    def test_save_sources_atomic_write(self, tmp_path):
        """_save_sources writes via temp file + atomic rename, leaving no partial file."""
        from ingest.external import _save_sources

        yaml_file = tmp_path / "external.yaml"

        import ingest.external as ext

        original = ext.SOURCES_FILE
        ext.SOURCES_FILE = yaml_file

        try:
            sources = [{"url": "https://example.com", "type": "external-api"}]
            _save_sources(sources)
            assert yaml_file.exists()
            # no leftover .tmp files
            tmp_files = list(tmp_path.glob("*.tmp"))
            assert tmp_files == [], f"leftover tmp files: {tmp_files}"
            # content is valid YAML
            from ruamel.yaml import YAML

            y = YAML()
            loaded = y.load(yaml_file.read_text())
            assert loaded[0]["url"] == "https://example.com"
        finally:
            ext.SOURCES_FILE = original

    @respx.mock
    def test_render_true_empty_body_skips(self, mocker, caplog):
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
        mocker.patch("ingest.external.ids_by_payload", return_value=[])
        mocker.patch("ingest.external.delete_by_ids")
        mocker.patch("ingest.external.upsert")

        jina_url = "https://r.jina.ai/https://example.com/js-only"
        respx.get(jina_url).mock(return_value=httpx.Response(200, text="   "))

        import logging

        from ingest.external import sync_external

        with caplog.at_level(logging.WARNING, logger="ingest.external"):
            sync_external()

        import ingest.external as ext

        ext.upsert.assert_not_called()
        assert "skipping" in caplog.text


class TestSyncExternalExceptionIsolation:
    """One source's exception must not stop the others (#144)."""

    TWO_SOURCES: ClassVar = [
        {
            "url": "https://example.com/broken",
            "type": "external-api",
            "check_interval": "weekly",
            "tags": ["test"],
            "last_etag": "",
            "last_synced": "",
        },
        {
            "url": "https://example.com/ok",
            "type": "external-api",
            "check_interval": "weekly",
            "tags": ["test"],
            "last_etag": "",
            "last_synced": "",
        },
    ]

    def test_phase1_fetch_exception_marks_source_failed_and_continues(self, mocker, caplog):
        """future.result() raising in the Phase 1 as_completed loop must be caught,
        the source marked failed, and the remaining sources still processed."""
        import logging

        mocker.patch("ingest.external._load_sources", return_value=list(self.TWO_SOURCES))
        mocker.patch("ingest.external._save_sources")
        mocker.patch("ingest.external.ids_by_payload", return_value=[])
        mocker.patch("ingest.external.delete_by_ids")
        mock_upsert = mocker.patch("ingest.external.upsert")

        def fake_fetch_source(source, force, now):
            if source["url"] == "https://example.com/broken":
                raise RuntimeError("boom")
            return {
                "status": "ready",
                "url": source["url"],
                "source": source,
                "new_etag": "",
                "points": [{"id": "x", "text": "ok"}],
            }

        mocker.patch("ingest.external._fetch_source", side_effect=fake_fetch_source)

        from ingest.external import sync_external

        with caplog.at_level(logging.WARNING, logger="ingest.external"):
            sync_external()

        # The broken source never reaches upsert; the ok source still does.
        mock_upsert.assert_called_once()
        assert "boom" in caplog.text or "RuntimeError" in caplog.text

    def test_phase2_upsert_exception_marks_source_failed_and_continues(self, mocker, caplog):
        """upsert() raising in the Phase 2 loop must be caught for that source,
        while the remaining sources are still written."""
        import logging

        mocker.patch("ingest.external._load_sources", return_value=list(self.TWO_SOURCES))
        saved = []
        mocker.patch("ingest.external._save_sources", side_effect=lambda s: saved.extend(s))
        mocker.patch("ingest.external.ids_by_payload", return_value=[])
        mocker.patch("ingest.external.delete_by_ids")

        def fake_upsert(collection, points):
            if points and points[0]["source_url"] == "https://example.com/broken":
                raise RuntimeError("qdrant unavailable")

        mocker.patch("ingest.external.upsert", side_effect=fake_upsert)

        def fake_fetch_source(source, force, now):
            return {
                "status": "ready",
                "url": source["url"],
                "source": source,
                "new_etag": "",
                "points": [
                    {
                        "id": f"{source['url']}:0",
                        "text": "content",
                        "source_url": source["url"],
                    }
                ],
            }

        mocker.patch("ingest.external._fetch_source", side_effect=fake_fetch_source)

        from ingest.external import sync_external

        with caplog.at_level(logging.WARNING, logger="ingest.external"):
            sync_external()

        assert "qdrant unavailable" in caplog.text or "RuntimeError" in caplog.text
        broken = next(s for s in saved if s["url"] == "https://example.com/broken")
        ok = next(s for s in saved if s["url"] == "https://example.com/ok")
        assert "failed_at" in broken
        assert "failed_at" not in ok
        assert ok["last_synced"]
