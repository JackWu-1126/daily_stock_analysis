# -*- coding: utf-8 -*-
"""Unit tests for LiteLLMGenerationBackend's pre-dispatch cancellation check."""

from __future__ import annotations

import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm.generation_backend import GenerationError, GenerationErrorCode
from src.llm.litellm_backend import LiteLLMGenerationBackend


class LiteLLMBackendCancellationTestCase(unittest.TestCase):
    def test_pre_set_cancel_event_raises_without_calling_completion(self) -> None:
        calls = []

        def _completion_callable(*args, **kwargs):
            calls.append((args, kwargs))
            return "text", "model", {}

        backend = LiteLLMGenerationBackend(_completion_callable)
        cancel_event = threading.Event()
        cancel_event.set()

        with self.assertRaises(GenerationError) as ctx:
            backend.generate("prompt", {"model": "openai/gpt-4"}, cancel_event=cancel_event)

        self.assertEqual(ctx.exception.error_code, GenerationErrorCode.CANCELLED)
        self.assertFalse(ctx.exception.retryable)
        self.assertFalse(ctx.exception.fallbackable)
        self.assertEqual(calls, [])

    def test_unset_cancel_event_proceeds_normally(self) -> None:
        def _completion_callable(*args, **kwargs):
            return "text", "model", {}

        backend = LiteLLMGenerationBackend(_completion_callable)
        cancel_event = threading.Event()

        result = backend.generate("prompt", {"model": "openai/gpt-4"}, cancel_event=cancel_event)
        self.assertEqual(result.text, "text")

    def test_no_cancel_event_proceeds_normally(self) -> None:
        def _completion_callable(*args, **kwargs):
            return "text", "model", {}

        backend = LiteLLMGenerationBackend(_completion_callable)
        result = backend.generate("prompt", {"model": "openai/gpt-4"})
        self.assertEqual(result.text, "text")


if __name__ == "__main__":
    unittest.main()
