"""Shared pytest fixtures for doc-ingest tests."""

import pytest


@pytest.fixture(autouse=True)
def _reset_settings():
    """Drop any cached/overridden Settings so tests never leak config into each other."""
    from core import config

    # Defaults only: never go through load_dotenv(), which would pull the developer's
    # real .env into os.environ for the whole process.
    # doc_ingest_base_dir="/" lets path_guard allow tmp_path and other arbitrary paths
    # without requiring real project files; tests of the guard itself override it.
    config._settings = config.Settings(doc_ingest_base_dir="/")
    yield
    config._settings = config.Settings()


def override_settings(**changes):
    """Replace the cached Settings with a copy that has `changes` applied (reset per test)."""
    import dataclasses

    from core import config

    config._settings = dataclasses.replace(config.get_settings(), **changes)
