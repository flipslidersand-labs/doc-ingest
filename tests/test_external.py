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
        mocker.patch("time.sleep")  # avoid real waits
        import logging

        from ingest.external import MAX_RETRIES, sync_external

        with caplog.at_level(logging.DEBUG, logger="ingest.external"):
            sync_external()
        attempts = respx.calls.call_count
        assert attempts == MAX_RETRIES, f"expected {MAX_RETRIES} HTTP attempts, got {attempts}"
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


class TestFetchWithRetry:
    """#154: _fetch_with_retry の再試行の意味(with_retry への移行後も維持)。"""

    def test_5xx_then_success_returns_response(self, mocker):
        sleep = mocker.patch("time.sleep")
        ok = httpx.Response(200, text="ok")
        mocker.patch("ingest.external.safe_get", side_effect=[httpx.Response(503), ok])
        from ingest.external import _fetch_with_retry

        assert _fetch_with_retry("https://example.com", {}) is ok
        sleep.assert_called_once_with(1)

    def test_4xx_is_returned_not_retried(self, mocker):
        sleep = mocker.patch("time.sleep")
        m = mocker.patch("ingest.external.safe_get", return_value=httpx.Response(404))
        from ingest.external import _fetch_with_retry

        assert _fetch_with_retry("https://example.com", {}).status_code == 404
        assert m.call_count == 1
        sleep.assert_not_called()

    def test_request_error_is_retried_then_succeeds(self, mocker):
        sleep = mocker.patch("time.sleep")
        ok = httpx.Response(200, text="ok")
        m = mocker.patch(
            "ingest.external.safe_get", side_effect=[httpx.ConnectError("refused"), ok]
        )
        from ingest.external import _fetch_with_retry

        assert _fetch_with_retry("https://example.com", {}) is ok
        assert m.call_count == 2
        sleep.assert_called_once_with(1)

    def test_all_attempts_fail_returns_none(self, mocker):
        sleep = mocker.patch("time.sleep")
        m = mocker.patch("ingest.external.safe_get", return_value=httpx.Response(500))
        from ingest.external import MAX_RETRIES, _fetch_with_retry

        assert _fetch_with_retry("https://example.com", {}) is None
        assert m.call_count == MAX_RETRIES
        assert sleep.call_count == MAX_RETRIES - 1

    def test_unsafe_redirect_returns_none_without_retry(self, mocker):
        from core.url_guard import UnsafeURLError

        sleep = mocker.patch("time.sleep")
        m = mocker.patch("ingest.external.safe_get", side_effect=UnsafeURLError("private ip"))
        from ingest.external import _fetch_with_retry

        assert _fetch_with_retry("https://example.com", {}) is None
        assert m.call_count == 1
        sleep.assert_not_called()


class TestExternalChunkLimit:
    """#211: 生チャンクを埋め込む external-docs は e5 の 512 token に収まる上限で分割する。"""

    def _source_with_html(self, body):
        return [
            {
                "url": "https://example.com/big",
                "type": "external-api",
                "check_interval": "weekly",
                "tags": [],
                "last_etag": "",
                "last_synced": "",
            }
        ], body

    @respx.mock
    def test_chunks_are_split_with_the_external_limit(self, mocker):
        from core.chunker import _count_tokens
        from ingest.external import EXTERNAL_MAX_SECTION_TOKENS

        sources, _ = self._source_with_html("")
        big = "## Section\n" + "\n\n".join(f"paragraph {i} " + "word " * 80 for i in range(12))
        respx.get("https://example.com/big").mock(
            return_value=httpx.Response(200, text=big, headers={"content-type": "text/markdown"})
        )
        mocker.patch("ingest.external._load_sources", return_value=sources)
        mocker.patch("ingest.external._save_sources")
        mocker.patch("ingest.external.ids_by_payload", return_value=[])
        mocker.patch("ingest.external.delete_by_ids")
        up = mocker.patch("ingest.external.upsert")

        from ingest.external import sync_external

        sync_external(force=True)

        points = up.call_args[0][1]
        assert len(points) > 1
        assert max(_count_tokens(p["text"]) for p in points) <= EXTERNAL_MAX_SECTION_TOKENS

    def test_limit_is_below_the_default_chunker_limit(self):
        from core.chunker import MAX_SECTION_TOKENS
        from ingest.external import EXTERNAL_MAX_SECTION_TOKENS

        assert EXTERNAL_MAX_SECTION_TOKENS < MAX_SECTION_TOKENS


class TestWorkerErrorLogsAreRedacted:
    def test_an_exception_message_with_a_url_does_not_reach_the_log(self, mocker, caplog):
        """sync の Phase 1 で予期しない例外が出たとき、メッセージ中の URL (トークン付き) を伏せる。"""
        import logging

        import ingest.external as ext

        sources = [{"url": "https://docs.example.com/x"}]
        mocker.patch.object(ext, "_load_sources", return_value=sources)
        mocker.patch.object(ext, "_save_sources")
        mocker.patch.object(
            ext,
            "_fetch_source",
            side_effect=RuntimeError("boom https://other.example/p?token=SECRET"),
        )
        with caplog.at_level(logging.WARNING, logger="ingest.external"):
            ext.sync_external(force=True)
        assert "SECRET" not in caplog.text and "boom <url>" in caplog.text

    def test_a_phase2_write_error_with_a_url_is_redacted_and_the_rest_continue(
        self, mocker, caplog
    ):
        """Phase 2 (Qdrant への書き込み) の例外のログ。メッセージ中の URL を伏せ、失敗として記録する。"""
        import logging

        import ingest.external as ext

        source = {"url": "https://docs.example.com/x"}
        mocker.patch.object(ext, "_load_sources", return_value=[source])
        mocker.patch.object(ext, "_save_sources")
        ready = {
            "status": "ready",
            "url": source["url"],
            "source": source,
            "new_etag": "e",
            "points": [{"id": "p1", "text": "t"}],
        }
        mocker.patch.object(ext, "_fetch_source", return_value=ready)
        mocker.patch.object(ext, "ids_by_payload", return_value=[])
        mocker.patch.object(
            ext,
            "upsert",
            side_effect=RuntimeError("write failed at https://qdrant.internal/x?api-key=SECRET"),
        )
        with caplog.at_level(logging.WARNING, logger="ingest.external"):
            ext.sync_external(force=True)
        assert "SECRET" not in caplog.text and "write failed at <url>" in caplog.text
        assert "failed_at" in source  # 失敗として記録される

    def test_a_retry_debug_log_does_not_carry_a_url_from_the_error(self, mocker, caplog):
        """retry のデバッグログ (-v で出る)。httpx.RequestError のメッセージに URL が入っていても伏せる。"""
        import logging

        import httpx

        import ingest.external as ext

        mocker.patch("time.sleep")
        mocker.patch.object(
            ext,
            "safe_get",
            side_effect=httpx.ConnectError("refused: https://h.example/p?token=SECRET"),
        )
        with caplog.at_level(logging.DEBUG, logger="ingest.external"):
            assert ext._fetch_with_retry("https://h.example/p", {}) is None
        assert "SECRET" not in caplog.text and "refused: <url>" in caplog.text


class TestFailedResult:
    """#227: 同じ形の失敗結果を 4 か所に書いていたのを _failed に集約した。形を固定する。"""

    def test_shape(self):
        from ingest.external import _failed

        source = {"url": "https://example.com/x", "check_interval": "weekly"}
        assert _failed("https://example.com/x", source) == {
            "status": "failed",
            "url": "https://example.com/x",
            "source": source,
            "new_etag": "",
            "points": [],
        }

    def test_each_call_gets_its_own_points_list(self):
        from ingest.external import _failed

        a, b = _failed("u", {}), _failed("u", {})
        a["points"].append("x")
        assert b["points"] == []  # 共有された可変オブジェクトにしない

    def test_a_blocked_url_is_reported_failed_not_skipped(self, mocker):
        from core.url_guard import UnsafeURLError
        from ingest.external import _fetch_source

        mocker.patch("ingest.external.assert_safe_url", side_effect=UnsafeURLError("blocked"))
        result = _fetch_source({"url": "https://10.0.0.1/x"}, force=False, now="n")
        assert result["status"] == "failed" and result["points"] == []

    def test_an_exception_in_a_worker_is_reported_failed_and_the_rest_continue(self, mocker):
        """スレッド内の予期しない例外 (Phase 1 の隔離) も _failed で返る。"""
        import ingest.external as ext

        sources = [{"url": "https://a.example/x"}, {"url": "https://b.example/x"}]
        mocker.patch.object(ext, "_load_sources", return_value=sources)
        mocker.patch.object(ext, "_save_sources")

        def fake_fetch(source, force, now):
            if source["url"].startswith("https://a."):
                raise RuntimeError("boom")
            return {
                "status": "skipped",
                "url": source["url"],
                "source": source,
                "new_etag": "",
                "points": [],
            }

        mocker.patch.object(ext, "_fetch_source", side_effect=fake_fetch)
        mocker.patch.object(ext, "upsert")
        ext.sync_external(force=True)
        assert ext._save_sources.called  # 失敗しても最後まで進む


class TestOversizedPage:
    def test_a_page_over_the_limit_is_reported_failed_and_nothing_is_stored(self, mocker, caplog):
        """#224: 巨大な応答を全量メモリに読まない。上限超過は取得失敗として扱う。"""
        import logging

        import ingest.external as ext
        from core.url_guard import UnsafeURLError

        mocker.patch("time.sleep")
        mocker.patch.object(ext, "assert_safe_url")
        mocker.patch.object(
            ext,
            "safe_get",
            side_effect=UnsafeURLError(
                "response exceeds 52428800 bytes: 'https://docs.example.com/x'"
            ),
        )
        up = mocker.patch.object(ext, "upsert")
        with caplog.at_level(logging.WARNING, logger="ingest.external"):
            result = ext._fetch_source({"url": "https://docs.example.com/x"}, force=True, now="n")
        assert result["status"] == "failed" and result["points"] == []
        up.assert_not_called()
        assert "blocked" in caplog.text and "exceeds" in caplog.text


class TestExternalTrust:
    @respx.mock
    def test_points_are_marked_external(self, mocker):
        """取り込んだ外部文書は trust=external で保存する (#225)。"""
        sources = [
            {
                "url": "https://example.com/doc",
                "type": "external-api",
                "check_interval": "weekly",
                "tags": [],
                "last_etag": "",
                "last_synced": "",
            }
        ]
        respx.get("https://example.com/doc").mock(
            return_value=httpx.Response(
                200,
                text="## S\n" + "word " * 30,
                headers={"content-type": "text/markdown"},
            )
        )
        mocker.patch("ingest.external._load_sources", return_value=sources)
        mocker.patch("ingest.external._save_sources")
        mocker.patch("ingest.external.ids_by_payload", return_value=[])
        mocker.patch("ingest.external.delete_by_ids")
        up = mocker.patch("ingest.external.upsert")

        from ingest.external import sync_external

        sync_external(force=True)

        points = up.call_args[0][1]
        assert points and {p["trust"] for p in points} == {"external"}
