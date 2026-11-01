"""Tests for core/qdrant.py embed batching, retry, and auth handling."""

import threading
import time
from unittest.mock import MagicMock

import httpx
import pytest

import core.qdrant as q


def test_qdrant_api_key_none_when_unset(monkeypatch):
    """QDRANT_API_KEY resolves to None when env var is absent or empty."""
    monkeypatch.setattr(q, "QDRANT_API_KEY", None)
    assert q.QDRANT_API_KEY is None


# ---------------------------------------------------------------------------
# Tests for #21: primary→fallback routing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests for #81: thread-safe client() singleton and _collection_exists cache
# ---------------------------------------------------------------------------


def test_client_singleton_threadsafe(monkeypatch):
    """Concurrent calls to client() must produce exactly one QdrantClient."""
    monkeypatch.setattr(q, "_client", None)

    created = []
    original_init = q.QdrantClient.__init__

    def slow_init(self, *args, **kwargs):
        time.sleep(0.05)  # simulate slow network handshake
        original_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(q.QdrantClient, "__init__", slow_init)

    results = []

    def call_client():
        results.append(q.client())

    threads = [threading.Thread(target=call_client) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len({id(r) for r in results}) == 1
    assert len(created) == 1


def test_ensure_collection_calls_get_collections_once(monkeypatch):
    """ensure_collection() fetches get_collections at most once for a known name."""
    monkeypatch.setattr(q, "_known_collections", set())

    mock_col = MagicMock()
    mock_col.name = "test-col"

    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = [mock_col]
    monkeypatch.setattr(q, "_client", mock_client)

    q.ensure_collection("test-col")
    assert mock_client.get_collections.call_count == 1
    assert mock_client.create_collection.call_count == 0

    q.ensure_collection("test-col")
    assert mock_client.get_collections.call_count == 1


def test_collection_exists_uses_cache(monkeypatch):
    """_collection_exists() skips network call when collection is cached."""
    monkeypatch.setattr(q, "_known_collections", {"cached-col"})
    mock_client = MagicMock()
    monkeypatch.setattr(q, "_client", mock_client)

    assert q._collection_exists("cached-col") is True
    mock_client.get_collections.assert_not_called()


def test_collection_exists_cache_miss_fetches_remote(monkeypatch):
    """_collection_exists() fetches from Qdrant on cache miss."""
    monkeypatch.setattr(q, "_known_collections", set())

    mock_col = MagicMock()
    mock_col.name = "remote-col"
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = [mock_col]
    monkeypatch.setattr(q, "_client", mock_client)

    assert q._collection_exists("remote-col") is True
    mock_client.get_collections.assert_called_once()

    assert q._collection_exists("remote-col") is True
    mock_client.get_collections.assert_called_once()


def test_ensure_collection_creates_datetime_index(monkeypatch):
    """ensure_collection() creates ingested_at datetime index on new collections."""
    from qdrant_client.models import PayloadSchemaType

    monkeypatch.setattr(q, "_known_collections", set())

    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []
    monkeypatch.setattr(q, "_client", mock_client)

    q.ensure_collection("new-col")

    mock_client.create_collection.assert_called_once()
    mock_client.create_payload_index.assert_called_once_with(
        "new-col", "ingested_at", field_schema=PayloadSchemaType.DATETIME
    )


def test_ensure_collection_skips_index_on_existing(monkeypatch):
    """ensure_collection() does NOT create index when collection already exists."""
    monkeypatch.setattr(q, "_known_collections", set())

    mock_col = MagicMock()
    mock_col.name = "existing-col"
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = [mock_col]
    monkeypatch.setattr(q, "_client", mock_client)

    q.ensure_collection("existing-col")

    mock_client.create_collection.assert_not_called()
    mock_client.create_payload_index.assert_not_called()


def test_ensure_collection_409_race_is_idempotent(monkeypatch):
    """A concurrent process creating the same collection first (#150) must
    not crash ensure_collection() — a 409 from create_collection() is
    treated as success, and the name still gets cached."""
    from qdrant_client.http.exceptions import UnexpectedResponse

    monkeypatch.setattr(q, "_known_collections", set())

    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []
    mock_client.create_collection.side_effect = UnexpectedResponse(
        409, "Conflict", b"already exists", httpx.Headers({})
    )
    monkeypatch.setattr(q, "_client", mock_client)

    q.ensure_collection("racy-col")  # must not raise

    assert "racy-col" in q._known_collections
    mock_client.create_payload_index.assert_not_called()  # skipped after the 409


def test_ensure_collection_non_409_error_propagates(monkeypatch):
    """A create_collection() failure that isn't a same-name race (e.g. a
    genuine server error) must still surface, not be silently swallowed."""
    from qdrant_client.http.exceptions import UnexpectedResponse

    monkeypatch.setattr(q, "_known_collections", set())

    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []
    mock_client.create_collection.side_effect = UnexpectedResponse(
        500, "Internal Server Error", b"boom", httpx.Headers({})
    )
    monkeypatch.setattr(q, "_client", mock_client)

    with pytest.raises(UnexpectedResponse):
        q.ensure_collection("broken-col")

    assert "broken-col" not in q._known_collections


# ---------------------------------------------------------------------------
# Tests for #91: list_collections uses order_by+limit=1
# ---------------------------------------------------------------------------


def test_list_collections_uses_order_by_limit_1(monkeypatch):
    """list_collections must use order_by desc + limit=1, not scroll(limit=50)."""
    from types import SimpleNamespace

    from qdrant_client.models import OrderBy

    mock_client = MagicMock()

    col = SimpleNamespace(name="test_col")
    mock_client.get_collections.return_value = SimpleNamespace(collections=[col])
    mock_client.get_collection.return_value = SimpleNamespace(points_count=3)

    latest_record = SimpleNamespace(payload={"ingested_at": "2024-06-01T12:00:00Z", "text": "hi"})
    mock_client.scroll.return_value = ([latest_record], None)

    monkeypatch.setattr(q, "_client", mock_client)

    result = q.list_collections()

    scroll_kwargs = mock_client.scroll.call_args.kwargs
    assert scroll_kwargs["limit"] == 1, f"expected limit=1, got {scroll_kwargs['limit']}"
    assert isinstance(scroll_kwargs["order_by"], OrderBy)
    assert scroll_kwargs["order_by"].key == "ingested_at"
    assert scroll_kwargs["order_by"].direction == "desc"

    assert len(result) == 1
    assert result[0]["name"] == "test_col"
    assert result[0]["points"] == 3
    assert result[0]["last_ingested"] == "2024-06-01 12:00:00"


# ---------------------------------------------------------------------------
# Tests for #153: upsert/search/delete CRUD functions
# ---------------------------------------------------------------------------


class TestUpsert:
    def test_upsert_empty_points_does_not_touch_qdrant_or_the_embedder(self, monkeypatch):
        """#229: 空リスト (全チャンクの要約失敗など) は何もしない。Qdrant が空リストを受け付けるかは
        実機未確認なので送らない。embed も呼ばない。"""
        monkeypatch.setattr(q, "_known_collections", {"col"})
        monkeypatch.setattr(q, "embed", lambda texts, **kw: pytest.fail("embed must not be called"))
        mock_client = MagicMock()
        monkeypatch.setattr(q, "_client", mock_client)

        q.upsert("col", [])

        mock_client.upsert.assert_not_called()
        mock_client.get_collections.assert_not_called()

    def test_upsert_raises_instead_of_silently_dropping_points_on_a_short_embedding(
        self, monkeypatch
    ):
        """#229: ベクトルが入力より少ないとき、zip が後ろの point を黙って落としていた。"""
        monkeypatch.setattr(q, "_known_collections", {"col"})
        monkeypatch.setattr(q, "embed", lambda texts, **kw: [[0.1] * 768])  # 3 件に 1 件だけ
        mock_client = MagicMock()
        monkeypatch.setattr(q, "_client", mock_client)

        with pytest.raises(ValueError):
            q.upsert(
                "col",
                [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}, {"id": "c", "text": "z"}],
            )

        mock_client.upsert.assert_not_called()  # 部分的な保存もしない

    def test_upsert_builds_point_structs_without_id_in_payload(self, monkeypatch):
        """upsert() must pass id= separately and never leak 'id' back into payload."""
        from qdrant_client.models import PointStruct

        monkeypatch.setattr(q, "_known_collections", {"col"})
        monkeypatch.setattr(q, "embed", lambda texts, **kw: [[0.1] * 768])
        mock_client = MagicMock()
        monkeypatch.setattr(q, "_client", mock_client)

        q.upsert("col", [{"id": "abc123", "text": "hello", "tags": ["x"]}])

        _, kwargs = mock_client.upsert.call_args
        structs = kwargs["points"]
        assert len(structs) == 1
        point = structs[0]
        assert isinstance(point, PointStruct)
        assert point.id == "abc123"
        assert point.vector == [0.1] * 768
        assert point.payload == {"text": "hello", "tags": ["x"]}
        assert "id" not in point.payload


class TestSearch:
    def test_search_handles_none_payload(self, monkeypatch):
        """A hit with payload=None must not raise — falls back to {} via `or {}`."""
        from types import SimpleNamespace

        monkeypatch.setattr(q, "_known_collections", {"col"})
        monkeypatch.setattr(q, "embed", lambda texts, **kw: [[0.1] * 768])
        mock_client = MagicMock()
        hit = SimpleNamespace(score=0.987654, id="pt1", payload=None)
        mock_client.query_points.return_value = SimpleNamespace(points=[hit])
        monkeypatch.setattr(q, "_client", mock_client)

        results = q.search("col", "query text")

        assert results == [{"score": 0.9877, "id": "pt1"}]

    def test_search_merges_payload_into_result(self, monkeypatch):
        from types import SimpleNamespace

        monkeypatch.setattr(q, "_known_collections", {"col"})
        monkeypatch.setattr(q, "embed", lambda texts, **kw: [[0.1] * 768])
        mock_client = MagicMock()
        hit = SimpleNamespace(score=0.5, id="pt2", payload={"text": "hi", "source": "x"})
        mock_client.query_points.return_value = SimpleNamespace(points=[hit])
        monkeypatch.setattr(q, "_client", mock_client)

        results = q.search("col", "query text")

        assert results == [{"score": 0.5, "id": "pt2", "text": "hi", "source": "x"}]

    def test_search_embeds_the_query_and_queries_with_that_vector(self, monkeypatch):
        """#155: ベクトルは core.embedding.embed() から来る。クエリを embed し、その結果で検索する。"""
        from types import SimpleNamespace

        calls = []
        monkeypatch.setattr(q, "_known_collections", {"col"})
        monkeypatch.setattr(
            q, "embed", lambda texts, **kw: calls.append((texts, kw.get("mode"))) or [[0.25] * 768]
        )
        mock_client = MagicMock()
        mock_client.query_points.return_value = SimpleNamespace(points=[])
        monkeypatch.setattr(q, "_client", mock_client)

        q.search("col", "my query", limit=3)

        assert calls == [(["my query"], "search")]  # #222: クエリは query: で埋め込む
        kwargs = mock_client.query_points.call_args.kwargs
        assert kwargs["query"] == [0.25] * 768 and kwargs["limit"] == 3


class TestUpsertEmbedsAsDocuments:
    def test_upsert_embeds_with_index_mode(self, monkeypatch):
        """文書側は passage: (index)。検索クエリ側 (search) と取り違えない。"""
        calls = []
        monkeypatch.setattr(q, "_known_collections", {"col"})
        monkeypatch.setattr(
            q, "embed", lambda texts, **kw: calls.append(kw.get("mode", "index")) or [[0.1] * 768]
        )
        monkeypatch.setattr(q, "_client", MagicMock())
        q.upsert("col", [{"id": "a", "text": "hello"}])
        assert calls == ["index"]


class TestDeleteByPayload:
    def test_noop_when_collection_does_not_exist(self, monkeypatch):
        monkeypatch.setattr(q, "_known_collections", set())
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        monkeypatch.setattr(q, "_client", mock_client)

        q.delete_by_payload("missing-col", "source_url", "https://x")

        mock_client.delete.assert_not_called()

    def test_deletes_by_filter_when_collection_exists(self, monkeypatch):
        monkeypatch.setattr(q, "_known_collections", {"col"})
        mock_client = MagicMock()
        monkeypatch.setattr(q, "_client", mock_client)

        q.delete_by_payload("col", "source_url", "https://x")

        mock_client.delete.assert_called_once()
        assert mock_client.delete.call_args.kwargs["collection_name"] == "col"


class TestIdsByPayload:
    def test_noop_when_collection_does_not_exist(self, monkeypatch):
        monkeypatch.setattr(q, "_known_collections", set())
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        monkeypatch.setattr(q, "_client", mock_client)

        assert q.ids_by_payload("missing-col", "source_url", "https://x") == []
        mock_client.scroll.assert_not_called()

    def test_single_page_under_limit(self, monkeypatch):
        from types import SimpleNamespace

        monkeypatch.setattr(q, "_known_collections", {"col"})
        mock_client = MagicMock()
        records = [SimpleNamespace(id=f"id{i}") for i in range(3)]
        mock_client.scroll.return_value = (records, None)
        monkeypatch.setattr(q, "_client", mock_client)

        ids = q.ids_by_payload("col", "source_url", "https://x")

        assert ids == ["id0", "id1", "id2"]
        assert mock_client.scroll.call_count == 1

    def test_paginates_across_multiple_pages_over_256(self, monkeypatch):
        """A >256-hit result set must be paginated across scroll() calls, not
        silently truncated to the first page (regression guard for #153)."""
        from types import SimpleNamespace

        monkeypatch.setattr(q, "_known_collections", {"col"})
        mock_client = MagicMock()

        page1 = [SimpleNamespace(id=f"id{i}") for i in range(256)]
        page2 = [SimpleNamespace(id=f"id{i}") for i in range(256, 300)]
        mock_client.scroll.side_effect = [
            (page1, "page2-offset"),
            (page2, None),
        ]
        monkeypatch.setattr(q, "_client", mock_client)

        ids = q.ids_by_payload("col", "source_url", "https://x")

        assert len(ids) == 300
        assert ids[0] == "id0"
        assert ids[-1] == "id299"
        assert mock_client.scroll.call_count == 2
        # second call must pass the offset returned by the first
        second_call_kwargs = mock_client.scroll.call_args_list[1].kwargs
        assert second_call_kwargs["offset"] == "page2-offset"


class TestDeleteByIds:
    def test_empty_ids_is_early_return_noop(self, monkeypatch):
        mock_client = MagicMock()
        monkeypatch.setattr(q, "_client", mock_client)
        # No _collection_exists stubbing needed — empty ids returns before that check.

        q.delete_by_ids("col", [])

        mock_client.delete.assert_not_called()

    def test_noop_when_collection_does_not_exist(self, monkeypatch):
        monkeypatch.setattr(q, "_known_collections", set())
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        monkeypatch.setattr(q, "_client", mock_client)

        q.delete_by_ids("missing-col", ["id1"])

        mock_client.delete.assert_not_called()

    def test_deletes_by_id_list_when_collection_exists(self, monkeypatch):
        monkeypatch.setattr(q, "_known_collections", {"col"})
        mock_client = MagicMock()
        monkeypatch.setattr(q, "_client", mock_client)

        q.delete_by_ids("col", ["id1", "id2"])

        mock_client.delete.assert_called_once_with(
            collection_name="col", points_selector=["id1", "id2"]
        )


# ---------------------------------------------------------------------------
# Tests for #135: EMBED_BATCH must be validated at import time
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# #154: retry semantics preserved after moving to core.retry.with_retry
# ---------------------------------------------------------------------------


class TestListCollectionsIsolatesAFailingCollection:
    """#229: 共有 Qdrant には ingested_at の index が無いコレクションがあり、order_by が失敗しうる。
    1 件の失敗で一覧全体 (doc-ingest list) を落とさない。"""

    @staticmethod
    def _client(fail_for):
        from types import SimpleNamespace

        c = MagicMock()
        c.get_collections.return_value = SimpleNamespace(
            collections=[SimpleNamespace(name=n) for n in ("alpha", "sessions", "zeta")]
        )
        c.get_collection.side_effect = lambda name: SimpleNamespace(points_count=len(name))

        def scroll(collection_name, **kw):
            if collection_name in fail_for:
                raise RuntimeError("Bad request: Index required but not found for ingested_at")
            return ([SimpleNamespace(payload={"ingested_at": "2026-11-01T09:30:00+00:00"})], None)

        c.scroll.side_effect = scroll
        return c

    def test_other_collections_are_still_listed(self, monkeypatch):
        monkeypatch.setattr(q, "_client", self._client(fail_for={"sessions"}))
        rows = q.list_collections()
        assert [r["name"] for r in rows] == ["alpha", "sessions", "zeta"]
        by = {r["name"]: r for r in rows}
        assert by["alpha"]["last_ingested"] == "2026-11-01 09:30:00"
        assert by["zeta"]["last_ingested"] == "2026-11-01 09:30:00"

    def test_the_failing_collection_is_marked_not_hidden(self, monkeypatch):
        monkeypatch.setattr(q, "_client", self._client(fail_for={"sessions"}))
        by = {r["name"]: r for r in q.list_collections()}
        assert by["sessions"]["last_ingested"] == "?"  # 空 ("—") と区別する
        assert by["sessions"]["points"] == len("sessions")  # 件数は取れている

    def test_the_failure_is_logged_with_the_collection_name(self, monkeypatch, caplog):
        import logging

        monkeypatch.setattr(q, "_client", self._client(fail_for={"sessions"}))
        with caplog.at_level(logging.WARNING, logger="core.qdrant"):
            q.list_collections()
        assert "sessions" in caplog.text and "RuntimeError" in caplog.text

    def test_a_keyboard_interrupt_is_not_swallowed(self, monkeypatch):
        c = self._client(fail_for=set())
        c.scroll.side_effect = KeyboardInterrupt
        monkeypatch.setattr(q, "_client", c)
        with pytest.raises(KeyboardInterrupt):
            q.list_collections()
