# -*- coding: utf-8 -*-
"""Unit tests for cooperative task cancellation in AnalysisTaskQueue."""

from __future__ import annotations

import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.task_queue import AnalysisTaskQueue, TaskStatus
from src.services.task_cancellation import TaskCancelledError


class TaskQueueCancellationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._original_instance = AnalysisTaskQueue._instance
        AnalysisTaskQueue._instance = None
        self.queue = AnalysisTaskQueue(max_workers=2)

    def tearDown(self) -> None:
        queue = AnalysisTaskQueue._instance
        if queue is not None and queue is not self._original_instance:
            executor = getattr(queue, "_executor", None)
            if executor is not None and hasattr(executor, "shutdown"):
                executor.shutdown(wait=False, cancel_futures=True)
        AnalysisTaskQueue._instance = self._original_instance

    def test_cancel_unknown_task_returns_none(self) -> None:
        self.assertIsNone(self.queue.request_cancel("does-not-exist"))

    def test_cancel_still_queued_task_is_immediate(self) -> None:
        # Block the only worker with a long-running blocker so the second
        # submitted task stays queued (never picked up by the pool).
        block_started = threading.Event()
        release_block = threading.Event()

        def _blocker():
            block_started.set()
            release_block.wait(timeout=5)
            return {"ok": True}

        def _queued_task():
            return {"ok": True}

        try:
            single_worker_queue = AnalysisTaskQueue._instance
            single_worker_queue._executor = None
            single_worker_queue._max_workers = 1

            blocker_task = self.queue.submit_background_task(
                _blocker, stock_code="BLOCKER"
            )
            self.assertTrue(block_started.wait(timeout=2))

            queued_task = self.queue.submit_background_task(
                _queued_task, stock_code="QUEUED"
            )
            self.assertEqual(
                self.queue.get_task(queued_task.task_id).status, TaskStatus.PENDING
            )

            cancelled = self.queue.request_cancel(queued_task.task_id)
            self.assertIsNotNone(cancelled)
            self.assertEqual(cancelled.status, TaskStatus.CANCELLED)
            self.assertIsNone(self.queue._cancel_events.get(queued_task.task_id))
        finally:
            release_block.set()

    def test_cancel_running_task_sets_cancel_requested_and_event(self) -> None:
        started = threading.Event()

        def _run_task():
            started.set()
            time.sleep(2)
            return {"ok": True}

        task = self.queue.submit_background_task(_run_task, stock_code="RUNNING")
        self.assertTrue(started.wait(timeout=2))

        updated = self.queue.request_cancel(task.task_id)
        self.assertIsNotNone(updated)
        self.assertEqual(updated.status, TaskStatus.CANCEL_REQUESTED)

        cancel_event = self.queue._cancel_events.get(task.task_id)
        self.assertIsNotNone(cancel_event)
        self.assertTrue(cancel_event.is_set())

    def test_execute_background_task_marks_cancelled_on_task_cancelled_error(self) -> None:
        def _run_task():
            raise TaskCancelledError("boom")

        task = self.queue.submit_background_task(_run_task, stock_code="CANCELME")

        deadline = time.monotonic() + 3
        final = self.queue.get_task(task.task_id)
        while final is not None and final.status not in (
            TaskStatus.CANCELLED,
            TaskStatus.FAILED,
        ) and time.monotonic() < deadline:
            time.sleep(0.02)
            final = self.queue.get_task(task.task_id)

        self.assertIsNotNone(final)
        self.assertEqual(final.status, TaskStatus.CANCELLED)
        self.assertIsNone(self.queue._cancel_events.get(task.task_id))

    def test_cancel_already_completed_task_is_noop(self) -> None:
        def _run_task():
            return {"ok": True}

        task = self.queue.submit_background_task(_run_task, stock_code="DONE")

        deadline = time.monotonic() + 3
        final = self.queue.get_task(task.task_id)
        while final is not None and final.status != TaskStatus.COMPLETED and time.monotonic() < deadline:
            time.sleep(0.02)
            final = self.queue.get_task(task.task_id)
        self.assertEqual(final.status, TaskStatus.COMPLETED)

        result = self.queue.request_cancel(task.task_id)
        self.assertIsNotNone(result)
        self.assertEqual(result.status, TaskStatus.COMPLETED)

    def test_register_cancel_event_is_idempotent(self) -> None:
        event1 = self.queue.register_cancel_event("t1")
        event2 = self.queue.register_cancel_event("t1")
        self.assertIs(event1, event2)


if __name__ == "__main__":
    unittest.main()
