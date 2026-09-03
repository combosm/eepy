from __future__ import annotations

import threading
import time
import unittest

from vision.monitoring_worker import MonitoringWorker


class SequenceCapture:
    def __init__(self, frames) -> None:
        self.frames = list(frames)
        self.released = False

    def read(self):
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def release(self) -> None:
        self.released = True


class BlockingCapture:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.continue_read = threading.Event()
        self.released = False

    def read(self):
        self.entered.set()
        self.continue_read.wait(1.0)
        return False, None

    def release(self) -> None:
        self.released = True


class BytesProcessor:
    def process(self, frame, loop_started_at: float) -> bytes:
        return b"jpeg-" + frame


class FailOnceProcessor:
    def __init__(self) -> None:
        self.calls = 0

    def process(self, frame, loop_started_at: float) -> bytes:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("inference failed")
        return b"jpeg-" + frame


class SourceAwareProcessor(BytesProcessor):
    def __init__(self) -> None:
        self.unavailable_count = 0

    def source_unavailable(self, observed_at: float) -> None:
        self.unavailable_count += 1


class MonitoringWorkerTests(unittest.TestCase):
    def make_worker(self, capture_factory, processor_factory=BytesProcessor):
        statuses = []
        worker = MonitoringWorker(
            capture_factory=capture_factory,
            processor_factory=processor_factory,
            status_callback=lambda running, error: statuses.append(
                (running, error)
            ),
            retry_seconds=0.001,
        )
        self.addCleanup(worker.stop)
        return worker, statuses

    def wait_until_sequence(self, worker, target: int):
        deadline = time.monotonic() + 1.0
        published = None
        sequence = 0
        while time.monotonic() < deadline and sequence < target:
            candidate = worker.wait_for_frame(sequence, timeout=0.05)
            if candidate is not None:
                published = candidate
                sequence = candidate.sequence
        self.assertIsNotNone(published)
        self.assertGreaterEqual(sequence, target)
        return published

    def test_start_is_idempotent_and_has_one_capture_owner(self) -> None:
        capture = BlockingCapture()
        factory_calls = 0

        def factory():
            nonlocal factory_calls
            factory_calls += 1
            return capture

        worker, _ = self.make_worker(factory)
        worker.start()
        worker.start()
        self.assertTrue(capture.entered.wait(1.0))

        self.assertEqual(factory_calls, 1)
        capture.continue_read.set()
        worker.stop()
        self.assertTrue(capture.released)

    def test_only_latest_frame_is_retained_for_slow_consumers(self) -> None:
        first = SequenceCapture([b"one", b"two", b"three"])
        calls = 0

        def factory():
            nonlocal calls
            calls += 1
            if calls == 1:
                return first
            raise RuntimeError("camera unavailable")

        worker, _ = self.make_worker(factory)
        worker.start()
        self.wait_until_sequence(worker, 3)

        latest = worker.wait_for_frame(0, timeout=0.1)

        self.assertEqual(latest.sequence, 3)
        self.assertEqual(latest.jpeg, b"jpeg-three")

    def test_multiple_consumers_read_the_same_published_frame(self) -> None:
        capture = SequenceCapture([b"shared"])
        worker, _ = self.make_worker(lambda: capture)
        worker.start()
        self.wait_until_sequence(worker, 1)

        first = worker.wait_for_frame(0, timeout=0.1)
        second = worker.wait_for_frame(0, timeout=0.1)

        self.assertEqual(first, second)

    def test_camera_open_failure_retries_without_restarting_worker(self) -> None:
        capture = SequenceCapture([b"recovered"])
        calls = 0
        processor = SourceAwareProcessor()

        def factory():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("camera unavailable")
            return capture

        worker, statuses = self.make_worker(factory, lambda: processor)
        worker.start()

        published = self.wait_until_sequence(worker, 1)

        self.assertEqual(published.jpeg, b"jpeg-recovered")
        self.assertIn((False, "camera unavailable"), statuses)
        self.assertIn((True, None), statuses)
        self.assertGreaterEqual(processor.unavailable_count, 1)

    def test_processing_error_drops_frame_and_continues_with_latest_input(self) -> None:
        capture = SequenceCapture([b"bad", b"good"])
        worker, statuses = self.make_worker(
            lambda: capture,
            FailOnceProcessor,
        )
        worker.start()

        published = self.wait_until_sequence(worker, 1)

        self.assertEqual(published.jpeg, b"jpeg-good")
        self.assertTrue(
            any(error and "frame processing failed" in error for _, error in statuses)
        )

    def test_stream_wraps_shared_jpeg_as_multipart_data(self) -> None:
        capture = SequenceCapture([b"frame"])
        worker, _ = self.make_worker(lambda: capture)

        chunk = next(worker.stream())

        self.assertIn(b"Content-Type: image/jpeg", chunk)
        self.assertIn(b"jpeg-frame", chunk)

    def test_negative_retry_delay_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MonitoringWorker(
                capture_factory=lambda: SequenceCapture([]),
                processor_factory=BytesProcessor,
                status_callback=lambda running, error: None,
                retry_seconds=-1.0,
            )


if __name__ == "__main__":
    unittest.main()
