"""Single-owner camera worker with latest-frame publication semantics."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Optional, Protocol


class Capture(Protocol):
    def read(self) -> tuple[bool, Any]: ...

    def release(self) -> None: ...


class FrameProcessor(Protocol):
    def process(self, frame: Any, loop_started_at: float) -> bytes: ...


@dataclass(frozen=True)
class PublishedFrame:
    sequence: int
    jpeg: bytes


StatusCallback = Callable[[bool, Optional[str]], None]


class MonitoringWorker:
    """Own capture/inference once and publish only the newest encoded frame."""

    def __init__(
        self,
        *,
        capture_factory: Callable[[], Capture],
        processor_factory: Callable[[], FrameProcessor],
        status_callback: StatusCallback,
        retry_seconds: float = 1.0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if retry_seconds < 0.0:
            raise ValueError("camera retry delay cannot be negative")
        self._capture_factory = capture_factory
        self._processor_factory = processor_factory
        self._status_callback = status_callback
        self._retry_seconds = retry_seconds
        self._clock = clock
        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: PublishedFrame | None = None
        self._sequence = 0

    @property
    def running(self) -> bool:
        with self._condition:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start once; concurrent or repeated calls are harmless."""
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            # Never expose the last frame from a previous worker lifetime as current.
            self._latest = None
            self._thread = threading.Thread(
                target=self._run,
                name="eepy-monitoring",
                daemon=True,
            )
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Request shutdown and wait briefly for the owner thread to release capture."""
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def wait_for_frame(
        self,
        after_sequence: int,
        timeout: float = 1.0,
    ) -> PublishedFrame | None:
        """Return a newer snapshot, skipping any intermediate published frames."""
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._latest is not None
                    and self._latest.sequence > after_sequence
                )
                or self._stop_event.is_set(),
                timeout=timeout,
            )
            if (
                self._latest is None
                or self._latest.sequence <= after_sequence
            ):
                return None
            return self._latest

    def stream(self):
        """Yield multipart JPEG chunks from shared latest-frame snapshots."""
        self.start()
        sequence = 0
        while not self._stop_event.is_set():
            published = self.wait_for_frame(sequence)
            if published is None:
                continue
            sequence = published.sequence
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + published.jpeg
                + b"\r\n"
            )

    def _publish(self, jpeg: bytes) -> None:
        with self._condition:
            self._sequence += 1
            self._latest = PublishedFrame(self._sequence, jpeg)
            self._condition.notify_all()

    def _notify_source_unavailable(self, processor: FrameProcessor) -> None:
        callback = getattr(processor, "source_unavailable", None)
        if callback is None:
            return
        try:
            callback(self._clock())
        except Exception as error:
            self._status_callback(
                False,
                f"camera-loss handling failed: {str(error) or type(error).__name__}",
            )

    @staticmethod
    def _release(capture: Capture | None) -> None:
        if capture is None:
            return
        try:
            capture.release()
        except Exception:
            pass

    def _run(self) -> None:
        capture: Capture | None = None
        try:
            processor = self._processor_factory()
            while not self._stop_event.is_set():
                try:
                    capture = self._capture_factory()
                except Exception as error:
                    self._status_callback(False, str(error) or type(error).__name__)
                    self._notify_source_unavailable(processor)
                    if self._stop_event.wait(self._retry_seconds):
                        break
                    continue

                self._status_callback(True, None)
                while not self._stop_event.is_set():
                    loop_started_at = self._clock()
                    try:
                        success, frame = capture.read()
                    except Exception as error:
                        success, frame = False, None
                        read_error = str(error) or type(error).__name__
                    else:
                        read_error = "camera frame read failed"

                    if not success:
                        self._status_callback(False, read_error)
                        self._notify_source_unavailable(processor)
                        break

                    try:
                        jpeg = processor.process(frame, loop_started_at)
                    except Exception as error:
                        self._status_callback(
                            False,
                            f"frame processing failed: {str(error) or type(error).__name__}",
                        )
                        continue

                    self._status_callback(True, None)
                    self._publish(jpeg)

                self._release(capture)
                capture = None
                if self._stop_event.wait(self._retry_seconds):
                    break
        except Exception as error:
            self._status_callback(
                False,
                f"monitoring startup failed: {str(error) or type(error).__name__}",
            )
        finally:
            self._release(capture)
            self._status_callback(False, "monitoring stopped")
            with self._condition:
                self._condition.notify_all()
