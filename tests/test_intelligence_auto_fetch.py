# -*- coding: utf-8 -*-
"""Tests for the always-on free RSS/NewsNow auto-fetch background loop in api/app.py.

This loop is intentionally independent of SCHEDULE_ENABLED / RuntimeSchedulerService --
it must run whenever the FastAPI app is up, seed the built-in free intelligence
sources once, and keep fetching them on an interval without ever dying from a
single bad iteration.
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.app import _intelligence_auto_fetch_loop, _schedule_intelligence_auto_fetch  # noqa: E402


class IntelligenceAutoFetchLoopTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_seeds_defaults_then_fetches_until_cancelled(self) -> None:
        service = MagicMock()
        service.create_default_sources.return_value = {"created_count": 8}
        service.fetch_enabled_sources.return_value = {"source_count": 8}

        sleep_calls = []
        stop_after = 2

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)
            if len(sleep_calls) >= stop_after:
                raise asyncio.CancelledError()

        with patch("src.services.intelligence_service.IntelligenceService", return_value=service), \
                patch("asyncio.sleep", side_effect=fake_sleep):
            with self.assertRaises(asyncio.CancelledError):
                await _intelligence_auto_fetch_loop(20)

        service.create_default_sources.assert_called_once_with({"enabled": True})
        # First fetch happens immediately (before the first sleep), then one more
        # per completed sleep before cancellation.
        self.assertEqual(service.fetch_enabled_sources.call_count, stop_after)
        self.assertEqual(sleep_calls, [20 * 60] * stop_after)

    async def test_seed_failure_does_not_block_fetch_loop(self) -> None:
        service = MagicMock()
        service.create_default_sources.side_effect = RuntimeError("db unavailable")
        service.fetch_enabled_sources.return_value = {"source_count": 0}

        async def fake_sleep(_seconds):
            raise asyncio.CancelledError()

        with patch("src.services.intelligence_service.IntelligenceService", return_value=service), \
                patch("asyncio.sleep", side_effect=fake_sleep):
            with self.assertRaises(asyncio.CancelledError):
                await _intelligence_auto_fetch_loop(20)

        service.fetch_enabled_sources.assert_called_once()

    async def test_fetch_cycle_exception_does_not_kill_loop(self) -> None:
        service = MagicMock()
        service.create_default_sources.return_value = {"created_count": 0}
        service.fetch_enabled_sources.side_effect = [
            RuntimeError("network down"),
            {"source_count": 8},
        ]

        sleep_calls = []

        async def fake_sleep(seconds):
            sleep_calls.append(seconds)
            if len(sleep_calls) >= 2:
                raise asyncio.CancelledError()

        with patch("src.services.intelligence_service.IntelligenceService", return_value=service), \
                patch("asyncio.sleep", side_effect=fake_sleep):
            with self.assertRaises(asyncio.CancelledError):
                await _intelligence_auto_fetch_loop(20)

        self.assertEqual(service.fetch_enabled_sources.call_count, 2)

    async def test_cancelled_error_during_fetch_propagates(self) -> None:
        service = MagicMock()
        service.create_default_sources.return_value = {"created_count": 0}
        service.fetch_enabled_sources.side_effect = asyncio.CancelledError()

        with patch("src.services.intelligence_service.IntelligenceService", return_value=service):
            with self.assertRaises(asyncio.CancelledError):
                await _intelligence_auto_fetch_loop(20)


class ScheduleIntelligenceAutoFetchTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        # A never-completing coroutine so created tasks stay "pending" for
        # task.done() checks, mirroring how the real loop behaves.
        async def _hang(*_args, **_kwargs):
            await asyncio.Event().wait()

        self._hang = _hang

    async def test_does_not_schedule_when_disabled(self) -> None:
        app_state = SimpleNamespace()
        app = SimpleNamespace(state=app_state)
        config = SimpleNamespace(news_intel_auto_fetch_enabled=False, news_intel_auto_fetch_interval_minutes=20)

        with patch("api.app._intelligence_auto_fetch_loop", side_effect=self._hang):
            from api.app import _schedule_intelligence_auto_fetch as schedule_fn

            schedule_fn(app, config)

        self.assertIsNone(getattr(app_state, "intel_auto_fetch_task", None))

    async def test_schedules_task_when_enabled(self) -> None:
        app_state = SimpleNamespace()
        app = SimpleNamespace(state=app_state)
        config = SimpleNamespace(news_intel_auto_fetch_enabled=True, news_intel_auto_fetch_interval_minutes=20)

        with patch("api.app._intelligence_auto_fetch_loop", side_effect=self._hang):
            _schedule_intelligence_auto_fetch(app, config)
            task = app_state.intel_auto_fetch_task
            self.assertIsNotNone(task)
            self.assertFalse(task.done())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_does_not_double_schedule_when_task_already_running(self) -> None:
        app_state = SimpleNamespace()
        app = SimpleNamespace(state=app_state)
        config = SimpleNamespace(news_intel_auto_fetch_enabled=True, news_intel_auto_fetch_interval_minutes=20)

        with patch("api.app._intelligence_auto_fetch_loop", side_effect=self._hang):
            _schedule_intelligence_auto_fetch(app, config)
            first_task = app_state.intel_auto_fetch_task
            _schedule_intelligence_auto_fetch(app, config)
            second_task = app_state.intel_auto_fetch_task

            self.assertIs(first_task, second_task)
            first_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first_task


if __name__ == "__main__":
    unittest.main()
