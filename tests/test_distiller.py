"""Tests for core/distiller.py"""
import json
from unittest.mock import MagicMock, patch

from core import distiller


def _mock_urlopen(response_text: str):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"response": response_text}).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestGenerate:
    def test_ollama_success(self):
        with patch("urllib.request.urlopen", return_value=_mock_urlopen("result text")):
            out = distiller._generate("prompt", source_text="raw")
        assert out == "result text"

    def test_ollama_failure_no_api_key_returns_raw(self, mocker):
        mocker.patch("urllib.request.urlopen", side_effect=OSError("refused"))
        mocker.patch.object(distiller, "ANTHROPIC_API_KEY", "")
        out = distiller._generate("prompt", source_text="fallback content")
        assert out == "fallback content"

    def test_ollama_failure_with_api_key_tries_anthropic(self, mocker):
        mocker.patch("urllib.request.urlopen", side_effect=OSError("refused"))
        mocker.patch.object(distiller, "ANTHROPIC_API_KEY", "sk-fake")
        mock_anthropic = mocker.patch.object(distiller, "_anthropic", return_value="haiku result")
        out = distiller._generate("prompt", source_text="raw")
        mock_anthropic.assert_called_once_with("prompt")
        assert out == "haiku result"

    def test_both_fail_returns_raw(self, mocker):
        mocker.patch("urllib.request.urlopen", side_effect=OSError("refused"))
        mocker.patch.object(distiller, "ANTHROPIC_API_KEY", "sk-fake")
        mocker.patch.object(distiller, "_anthropic", side_effect=Exception("api error"))
        out = distiller._generate("prompt", source_text="raw fallback here")
        assert out == "raw fallback here"

    def test_raw_fallback_truncates_at_500(self, mocker):
        mocker.patch("urllib.request.urlopen", side_effect=OSError("x"))
        mocker.patch.object(distiller, "ANTHROPIC_API_KEY", "")
        long_text = "a" * 1000
        out = distiller._generate("prompt", source_text=long_text)
        assert len(out) == 500



    def test_memory_error_not_swallowed_ollama(self, mocker):
        mocker.patch("urllib.request.urlopen", side_effect=MemoryError("oom"))
        import pytest
        with pytest.raises(MemoryError):
            distiller._generate("prompt", source_text="raw")

    def test_keyboard_interrupt_not_swallowed_anthropic(self, mocker):
        mocker.patch("urllib.request.urlopen", side_effect=OSError("refused"))
        mocker.patch.object(distiller, "ANTHROPIC_API_KEY", "sk-fake")
        mocker.patch.object(distiller, "_anthropic", side_effect=KeyboardInterrupt)
        import pytest
        with pytest.raises(KeyboardInterrupt):
            distiller._generate("prompt", source_text="raw")

    def test_fallback_emits_warning(self, mocker, caplog):
        mocker.patch("urllib.request.urlopen", side_effect=OSError("refused"))
        mocker.patch.object(distiller, "ANTHROPIC_API_KEY", "")
        import logging
        with caplog.at_level(logging.WARNING):
            distiller._generate("prompt", source_text="fallback text")
        assert any("raw text fallback used" in r.message for r in caplog.records)


class TestDistillFunctions:
    def test_distill_paper_calls_generate(self, mocker):
        mock_gen = mocker.patch.object(distiller, "_generate", return_value="distilled")
        result = distiller.distill_paper("abstract text", body="body text")
        assert result == "distilled"
        assert mock_gen.call_count == 1
        prompt = mock_gen.call_args[0][0]
        assert "abstract text" in prompt

    def test_distill_design_doc_calls_generate(self, mocker):
        mock_gen = mocker.patch.object(distiller, "_generate", return_value="ok")
        distiller.distill_design_doc("section content")
        prompt = mock_gen.call_args[0][0]
        assert "section content" in prompt

    def test_distill_webpage_calls_generate(self, mocker):
        mock_gen = mocker.patch.object(distiller, "_generate", return_value="ok")
        distiller.distill_webpage("article content")
        prompt = mock_gen.call_args[0][0]
        assert "article content" in prompt

    def test_distill_text_calls_generate(self, mocker):
        mock_gen = mocker.patch.object(distiller, "_generate", return_value="ok")
        distiller.distill_text("some text", "extract key points")
        prompt = mock_gen.call_args[0][0]
        assert "extract key points" in prompt
        assert "some text" in prompt

