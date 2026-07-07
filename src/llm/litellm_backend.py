# -*- coding: utf-8 -*-
"""LiteLLM generation backend wrapper."""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional, Tuple

from src.llm.generation_backend import (
    GenerationBackend,
    GenerationCapabilities,
    GenerationError,
    GenerationErrorCode,
    GenerationResult,
)


LiteLLMCallable = Callable[..., Tuple[str, str, Dict[str, Any]]]


def _provider_from_model(model: str) -> str:
    if not model:
        return ""
    if "/" in model:
        return model.split("/", 1)[0]
    return "openai"


class LiteLLMGenerationBackend(GenerationBackend):
    """Thin adapter around the existing LiteLLM analyzer call path."""

    backend_id = "litellm"
    capabilities = GenerationCapabilities(
        supports_json=True,
        supports_tools=True,
        supports_stream=True,
        supports_vision=False,
        supports_health_check=False,
        supports_smoke_test=False,
    )

    def __init__(self, completion_callable: LiteLLMCallable):
        self._completion_callable = completion_callable

    def generate(
        self,
        prompt: str,
        generation_config: Dict[str, Any],
        *,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        stream_progress_callback: Optional[Callable[[int], None]] = None,
        response_validator: Optional[Callable[[str], None]] = None,
        audit_context: Optional[Dict[str, Any]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> GenerationResult:
        if cancel_event is not None and cancel_event.is_set():
            # litellm.completion() is a single blocking HTTP call with no external
            # abort handle -- cancellation can only be honored before dispatch.
            raise GenerationError(
                error_code=GenerationErrorCode.CANCELLED,
                stage="execution",
                retryable=False,
                fallbackable=False,
                backend=self.backend_id,
                provider=_provider_from_model(generation_config.get("model", "")),
                details={"reason": "cancelled"},
            )
        text, model, usage = self._completion_callable(
            prompt,
            generation_config,
            system_prompt=system_prompt,
            stream=stream,
            stream_progress_callback=stream_progress_callback,
            response_validator=response_validator,
            audit_context=audit_context,
        )
        provider = str((usage or {}).get("provider") or _provider_from_model(model))
        return GenerationResult(
            text=text,
            model=model,
            provider=provider,
            backend=self.backend_id,
            usage=usage or {},
            raw=None,
            diagnostics={},
        )
