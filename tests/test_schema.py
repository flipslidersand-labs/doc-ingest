from core.schema import Collection, collection_names, payload_origin


def test_collection_names():
    assert collection_names() == ["external-docs", "research", "design", "notes"]
    assert Collection.RESEARCH.value == "research"


def test_origin_prefers_source_url():
    assert payload_origin({"source_url": "u", "file_path": "f", "arxiv_id": "a"}) == "u"


def test_origin_falls_back_to_file_path_then_arxiv_id():
    assert payload_origin({"file_path": "f", "arxiv_id": "a"}) == "f"
    assert payload_origin({"arxiv_id": "a"}) == "a"


def test_origin_new_key_wins_and_empty():
    assert payload_origin({"origin": "o", "source_url": "u"}) == "o"
    assert payload_origin({}) == ""
