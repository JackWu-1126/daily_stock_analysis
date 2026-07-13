# -*- coding: utf-8 -*-
"""Tests for the narrow-scope Claude WebSearch news fallback (_load_claude_search_context)."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.core.pipeline import StockAnalysisPipeline


def _pipeline(*, news_claude_search_enabled: bool) -> StockAnalysisPipeline:
    pipeline = StockAnalysisPipeline.__new__(StockAnalysisPipeline)
    pipeline.config = SimpleNamespace(news_claude_search_enabled=news_claude_search_enabled)
    pipeline.db = MagicMock()
    pipeline.query_id = "test-query-id"
    pipeline.query_source = "test"
    pipeline.source_message = None
    return pipeline


_SAMPLE_ITEMS = [
    {
        "title": "和大(1536) 訂單成長",
        "source": "經濟日報",
        "date": "2026-07-10",
        "summary": "產能利用率提升",
        "url": "https://news.example.com/1536",
    }
]


class ClaudeSearchContextTestCase(unittest.TestCase):
    def test_disabled_returns_none_without_invoking_cli(self) -> None:
        pipeline = _pipeline(news_claude_search_enabled=False)
        with patch("src.llm.local_cli_backend.LocalCliGenerationBackend") as backend_cls:
            context = pipeline._load_claude_search_context(code="1536", stock_name="和大")
        self.assertIsNone(context)
        backend_cls.assert_not_called()
        pipeline.db.save_news_intel.assert_not_called()

    def test_enabled_success_returns_context_and_persists_news_intel(self) -> None:
        pipeline = _pipeline(news_claude_search_enabled=True)
        mock_backend = MagicMock()
        mock_backend.generate.return_value = SimpleNamespace(text=json.dumps(_SAMPLE_ITEMS))
        with patch(
            "src.llm.local_cli_backend.LocalCliGenerationBackend",
            return_value=mock_backend,
        ) as backend_cls:
            context = pipeline._load_claude_search_context(code="1536", stock_name="和大")

        self.assertIsNotNone(context)
        assert context is not None
        self.assertIn("Claude WebSearch 情报", context)
        self.assertIn("和大", context)
        self.assertIn("1536", context)
        self.assertIn("訂單成長", context)
        backend_cls.assert_called_once()
        mock_backend.generate.assert_called_once()

        # Structured items must also be persisted so they surface in the
        # "相关资讯" history UI, not just the AI-facing news_context text.
        pipeline.db.save_news_intel.assert_called_once()
        _, kwargs = pipeline.db.save_news_intel.call_args
        self.assertEqual(kwargs["code"], "1536")
        self.assertEqual(kwargs["dimension"], "claude_websearch")
        saved_response = kwargs["response"]
        self.assertEqual(len(saved_response.results), 1)
        saved_item = saved_response.results[0]
        self.assertEqual(saved_item.title, "和大(1536) 訂單成長")
        self.assertEqual(saved_item.url, "https://news.example.com/1536")
        self.assertEqual(saved_item.source, "經濟日報")
        self.assertEqual(saved_item.published_date, "2026-07-10")

    def test_enabled_empty_json_array_returns_none(self) -> None:
        pipeline = _pipeline(news_claude_search_enabled=True)
        mock_backend = MagicMock()
        mock_backend.generate.return_value = SimpleNamespace(text="[]")
        with patch(
            "src.llm.local_cli_backend.LocalCliGenerationBackend",
            return_value=mock_backend,
        ):
            context = pipeline._load_claude_search_context(code="1536", stock_name="和大")
        self.assertIsNone(context)
        pipeline.db.save_news_intel.assert_not_called()

    def test_enabled_malformed_json_returns_none(self) -> None:
        pipeline = _pipeline(news_claude_search_enabled=True)
        mock_backend = MagicMock()
        mock_backend.generate.return_value = SimpleNamespace(text="not json at all")
        with patch(
            "src.llm.local_cli_backend.LocalCliGenerationBackend",
            return_value=mock_backend,
        ):
            context = pipeline._load_claude_search_context(code="1536", stock_name="和大")
        self.assertIsNone(context)
        pipeline.db.save_news_intel.assert_not_called()

    def test_enabled_json_wrapped_in_code_fence_is_parsed(self) -> None:
        pipeline = _pipeline(news_claude_search_enabled=True)
        mock_backend = MagicMock()
        fenced = "```json\n" + json.dumps(_SAMPLE_ITEMS) + "\n```"
        mock_backend.generate.return_value = SimpleNamespace(text=fenced)
        with patch(
            "src.llm.local_cli_backend.LocalCliGenerationBackend",
            return_value=mock_backend,
        ):
            context = pipeline._load_claude_search_context(code="1536", stock_name="和大")
        self.assertIsNotNone(context)
        pipeline.db.save_news_intel.assert_called_once()

    def test_enabled_empty_text_returns_none(self) -> None:
        pipeline = _pipeline(news_claude_search_enabled=True)
        mock_backend = MagicMock()
        mock_backend.generate.return_value = SimpleNamespace(text="   ")
        with patch(
            "src.llm.local_cli_backend.LocalCliGenerationBackend",
            return_value=mock_backend,
        ):
            context = pipeline._load_claude_search_context(code="1536", stock_name="和大")
        self.assertIsNone(context)

    def test_enabled_cli_exception_fails_open(self) -> None:
        pipeline = _pipeline(news_claude_search_enabled=True)
        with patch(
            "src.llm.local_cli_backend.LocalCliGenerationBackend",
            side_effect=RuntimeError("claude not found"),
        ):
            context = pipeline._load_claude_search_context(code="1536", stock_name="和大")
        self.assertIsNone(context)

    def test_enabled_db_save_failure_still_returns_context(self) -> None:
        """A persistence failure must not take down the AI-facing news_context
        (fail-open for the save step specifically, not just the whole call)."""
        pipeline = _pipeline(news_claude_search_enabled=True)
        pipeline.db.save_news_intel.side_effect = RuntimeError("db boom")
        mock_backend = MagicMock()
        mock_backend.generate.return_value = SimpleNamespace(text=json.dumps(_SAMPLE_ITEMS))
        with patch(
            "src.llm.local_cli_backend.LocalCliGenerationBackend",
            return_value=mock_backend,
        ):
            context = pipeline._load_claude_search_context(code="1536", stock_name="和大")
        self.assertIsNotNone(context)


if __name__ == "__main__":
    unittest.main()
