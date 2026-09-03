"""Low-overhead benchmarking helpers for the real-time camera pipeline."""

from __future__ import annotations

import math
import os
import statistics
import time


BENCHMARK_ENABLED = os.getenv("EEPY_BENCHMARK_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BENCHMARK_FRAME_COUNT = int(os.getenv("EEPY_BENCHMARK_FRAME_COUNT", "500"))


def _latency_stats(samples: list[float]) -> dict[str, float] | None:
    """return milliseconds using a nearest-rank p95"""
    if not samples:
        return None

    ordered = sorted(samples)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean": statistics.fmean(ordered) * 1000,
        "median": statistics.median(ordered) * 1000,
        "p95": ordered[p95_index] * 1000,
        "min": ordered[0] * 1000,
        "max": ordered[-1] * 1000,
    }


class EepyBenchmark:
    """collect timings in memory and print once after the requested frame count"""

    def __init__(self, enabled: bool, frame_count: int) -> None:
        self.enabled = enabled
        self.frame_count = max(1, frame_count)
        self.processing_latencies: list[float] = []
        self.loop_latencies: list[float] = []
        self.raw_to_confirmed: list[float] = []
        self.confirmed_to_intervention: list[float] = []
        self.raw_to_intervention: list[float] = []
        self._raw_signal_started_at: float | None = None
        self._confirmed_at: float | None = None
        self._raw_at_confirmation: float | None = None
        self._printed = False

    def now(self) -> float:
        return time.perf_counter()

    def observe_raw_signal(self, active: bool, observed_at: float) -> None:
        if not self.enabled or self._printed:
            return
        if active:
            if self._raw_signal_started_at is None:
                self._raw_signal_started_at = observed_at
        elif self._confirmed_at is None:
            # an unconfirmed episode ended. Do not reuse its timestamp later.
            self._raw_signal_started_at = None

    def confirm_fatigue(self, confirmed_at: float) -> None:
        if not self.enabled or self._printed or self._confirmed_at is not None:
            return
        self._confirmed_at = confirmed_at
        self._raw_at_confirmation = self._raw_signal_started_at
        if self._raw_at_confirmation is not None:
            self.raw_to_confirmed.append(confirmed_at - self._raw_at_confirmation)

    def intervention_delivered(self, delivered_at: float) -> None:
        if not self.enabled or self._printed or self._confirmed_at is None:
            return
        self.confirmed_to_intervention.append(delivered_at - self._confirmed_at)
        if self._raw_at_confirmation is not None:
            self.raw_to_intervention.append(delivered_at - self._raw_at_confirmation)
        self._raw_signal_started_at = None
        self._raw_at_confirmation = None
        self._confirmed_at = None

    def fatigue_recovered(self) -> None:
        """Close an episode that ended without a recorded alarm delivery."""
        if not self.enabled or self._printed:
            return
        self._raw_signal_started_at = None
        self._raw_at_confirmation = None
        self._confirmed_at = None

    def record_frame(self, processing_seconds: float, loop_seconds: float) -> None:
        if not self.enabled or self._printed:
            return
        self.processing_latencies.append(processing_seconds)
        self.loop_latencies.append(loop_seconds)
        if len(self.processing_latencies) >= self.frame_count:
            self.print_summary()

    @staticmethod
    def _print_latency(label: str, samples: list[float]) -> None:
        stats = _latency_stats(samples)
        if stats is None:
            return
        print(f"{label}:")
        print(f"  Mean: {stats['mean']:.2f} ms")
        print(f"  Median: {stats['median']:.2f} ms")
        print(f"  P95: {stats['p95']:.2f} ms")
        print(f"  Min: {stats['min']:.2f} ms")
        print(f"  Max: {stats['max']:.2f} ms")

    def print_summary(self) -> None:
        if self._printed or not self.processing_latencies:
            return
        self._printed = True
        frames = len(self.processing_latencies)
        total_loop_time = sum(self.loop_latencies)

        print("\n=== EEPY BENCHMARK RESULTS ===")
        print("Vision Pipeline")
        print(f"Frames processed: {frames}")
        print(f"Total benchmark time: {total_loop_time:.2f} s")
        if total_loop_time:
            print(f"Average FPS: {frames / total_loop_time:.2f}")
        self._print_latency("Processing-only latency", self.processing_latencies)
        self._print_latency("End-to-end frame-loop latency", self.loop_latencies)

        print("Fatigue Detection")
        print(f"Events measured: {len(self.confirmed_to_intervention)}")
        self._print_latency(
            "Raw signal -> confirmed fatigue", self.raw_to_confirmed
        )
        self._print_latency(
            "Confirmed fatigue -> local alarm delivery",
            self.confirmed_to_intervention,
        )
        self._print_latency(
            "Raw signal -> local alarm delivery", self.raw_to_intervention
        )
