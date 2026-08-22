"""Tests for core/ids.py — deterministic UUID chunk ID generation."""
import uuid

from core.ids import make_id


class TestMakeId:
    def test_returns_valid_uuid_string(self):
        result = make_id("https://example.com:0")
        assert len(result) == 36
        # raises ValueError if not a valid UUID
        uuid.UUID(result)

    def test_deterministic_same_key(self):
        assert make_id("https://example.com:0") == make_id("https://example.com:0")

    def test_different_keys_different_ids(self):
        assert make_id("https://example.com:0") != make_id("https://example.com:1")

    def test_chunk_index_in_key_matters(self):
        ids = {make_id(f"url:{i}") for i in range(10)}
        assert len(ids) == 10  # all distinct

    def test_no_collisions_at_scale(self):
        ids = [make_id(f"https://source-{i}.example.com/doc:{j}") for i in range(100) for j in range(100)]
        assert len(ids) == len(set(ids))  # 10 000 pairs, 0 collisions

    def test_url_in_key_matters(self):
        assert make_id("https://a.com:0") != make_id("https://b.com:0")

    def test_empty_string_is_stable(self):
        id1 = make_id("")
        id2 = make_id("")
        assert id1 == id2

    def test_uuid_version_field(self):
        # UUID bytes come from SHA-256 — version/variant nibbles are raw hash bits,
        # not forced to v4/v5, but the string must still parse as a UUID.
        result = make_id("test")
        parsed = uuid.UUID(result)
        assert str(parsed) == result
