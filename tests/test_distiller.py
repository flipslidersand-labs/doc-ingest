"""Tests for core/distiller.py"""

import json
from unittest.mock import MagicMock, patch

import pytest

from core import distiller


def _mock_urlopen(response_text: str, finish: str = "stop"):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(
        {"choices": [{"message": {"content": response_text}, "finish_reason": finish}]}
    ).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestGenerate:
    def test_local_success(self):
        with patch("urllib.request.urlopen", return_value=_mock_urlopen("result text")):
            out = distiller._generate("prompt")
        assert out == "result text"

    def test_local_failure_no_api_key_raises(self, mocker):
        mocker.patch("urllib.request.urlopen", side_effect=OSError("refused"))
        mocker.patch.object(distiller, "ANTHROPIC_API_KEY", "")
        with pytest.raises(distiller.DistillError):
            distiller._generate("prompt")

    def test_local_failure_with_api_key_tries_anthropic(self, mocker):
        mocker.patch("urllib.request.urlopen", side_effect=OSError("refused"))
        mocker.patch.object(distiller, "ANTHROPIC_API_KEY", "sk-fake")
        mock_anthropic = mocker.patch.object(distiller, "_anthropic", return_value="haiku result")
        out = distiller._generate("prompt")
        mock_anthropic.assert_called_once_with("prompt")
        assert out == "haiku result"

    def test_both_fail_raises_and_never_returns_source_text(self, mocker):
        mocker.patch("urllib.request.urlopen", side_effect=OSError("refused"))
        mocker.patch.object(distiller, "ANTHROPIC_API_KEY", "sk-fake")
        mocker.patch.object(distiller, "_anthropic", side_effect=Exception("api error"))
        with pytest.raises(distiller.DistillError):
            distiller._generate("prompt")

    def test_local_only_success_uses_local(self):
        with patch("urllib.request.urlopen", return_value=_mock_urlopen("local result")):
            assert distiller._generate("prompt", local_only=True) == "local result"

    def test_local_only_never_calls_anthropic_even_with_api_key(self, mocker):
        mocker.patch("urllib.request.urlopen", side_effect=OSError("refused"))
        mocker.patch.object(distiller, "ANTHROPIC_API_KEY", "sk-fake")
        mock_anthropic = mocker.patch.object(distiller, "_anthropic", return_value="LEAKED")
        with pytest.raises(distiller.DistillError):
            distiller._generate("private prompt", local_only=True)
        mock_anthropic.assert_not_called()

    def test_memory_error_not_swallowed_local(self, mocker):
        mocker.patch("urllib.request.urlopen", side_effect=MemoryError("oom"))
        with pytest.raises(MemoryError):
            distiller._generate("prompt")

    def test_keyboard_interrupt_not_swallowed_anthropic(self, mocker):
        mocker.patch("urllib.request.urlopen", side_effect=OSError("refused"))
        mocker.patch.object(distiller, "ANTHROPIC_API_KEY", "sk-fake")
        mocker.patch.object(distiller, "_anthropic", side_effect=KeyboardInterrupt)
        with pytest.raises(KeyboardInterrupt):
            distiller._generate("prompt")


class TestLocalRequest:
    def _capture(self, mocker, text="ok", finish="stop"):
        return mocker.patch("urllib.request.urlopen", return_value=_mock_urlopen(text, finish))

    def test_posts_openai_chat_completions(self, mocker):
        mocker.patch.object(distiller, "LOCAL_LLM_URL", "http://h:18080/")
        mocker.patch.object(distiller, "LOCAL_LLM_MODEL", "")
        m = self._capture(mocker, "要約")
        assert distiller._local("hello") == "要約"
        req = m.call_args.args[0]
        assert req.full_url == "http://h:18080/v1/chat/completions"
        body = json.loads(req.data)
        assert body["messages"] == [{"role": "user", "content": "hello"}]
        assert body["max_tokens"] == distiller.LOCAL_LLM_MAX_TOKENS >= 1500
        assert body["temperature"] == 0 and body["stream"] is False
        assert "model" not in body

    def test_model_is_sent_when_configured(self, mocker):
        mocker.patch.object(distiller, "LOCAL_LLM_MODEL", "gemma-4-12b")
        m = self._capture(mocker)
        distiller._local("p")
        assert json.loads(m.call_args.args[0].data)["model"] == "gemma-4-12b"

    def test_empty_content_is_an_error_not_a_summary(self, mocker):
        self._capture(mocker, "  ", finish="length")
        with pytest.raises(ValueError, match="length"):
            distiller._local("p")

    def test_empty_content_falls_to_anthropic_then_distill_error(self, mocker):
        self._capture(mocker, "", finish="length")
        mocker.patch.object(distiller, "ANTHROPIC_API_KEY", "")
        with pytest.raises(distiller.DistillError):
            distiller._generate("p")

    def test_does_not_call_ollama_api(self, mocker):
        m = self._capture(mocker)
        distiller._local("p")
        assert "/api/generate" not in m.call_args.args[0].full_url


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
        assert mock_gen.call_args.kwargs["local_only"] is True  # 設計 doc は非公開

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

    def test_distill_text_forwards_local_only(self, mocker):
        mock_gen = mocker.patch.object(distiller, "_generate", return_value="ok")
        distiller.distill_text("t", "p", local_only=True)
        assert mock_gen.call_args.kwargs["local_only"] is True
        distiller.distill_text("t", "p")
        assert mock_gen.call_args.kwargs["local_only"] is False
