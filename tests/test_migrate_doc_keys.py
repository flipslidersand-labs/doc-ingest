"""scripts/migrate_doc_keys.py: repo の無い (#216 以前の) point だけを消す。dry-run が既定。"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client.models import FilterSelector, IsEmptyCondition

_spec = importlib.util.spec_from_file_location(
    "migrate_doc_keys", Path(__file__).parent.parent / "scripts" / "migrate_doc_keys.py"
)
mig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mig)


def _client(count=3):
    c = MagicMock()
    c.count.return_value = SimpleNamespace(count=count)
    c.scroll.return_value = (
        [SimpleNamespace(payload={"file_path": "CLAUDE.md"}) for _ in range(min(count, 2))],
        None,
    )
    return c


def test_dry_run_is_the_default_and_deletes_nothing():
    c = _client()
    assert mig.main([], client=c) == 0
    c.delete.assert_not_called()
    c.count.assert_called()


def test_apply_deletes_by_the_unmigrated_filter_in_both_collections():
    c = _client()
    mig.main(["--apply"], client=c)
    assert [k.kwargs["collection_name"] for k in c.delete.call_args_list] == ["design", "notes"]
    sel = c.delete.call_args.kwargs["points_selector"]
    assert isinstance(sel, FilterSelector)
    cond = sel.filter.must[0]
    assert isinstance(cond, IsEmptyCondition) and cond.is_empty.key == "repo"


def test_apply_skips_delete_when_nothing_is_unmigrated():
    c = _client(count=0)
    mig.main(["--apply"], client=c)
    c.delete.assert_not_called()


def test_only_design_and_notes_are_accepted():
    with pytest.raises(SystemExit):
        mig.main(["external-docs", "--apply"], client=_client())


def test_single_collection_can_be_selected():
    c = _client()
    mig.main(["notes", "--apply"], client=c)
    assert [k.kwargs["collection_name"] for k in c.delete.call_args_list] == ["notes"]
