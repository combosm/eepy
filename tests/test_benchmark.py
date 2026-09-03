from __future__ import annotations

import unittest

from vision.benchmark import EepyBenchmark


class InterventionBenchmarkTests(unittest.TestCase):
    def test_successful_local_delivery_records_confirmation_latency(self) -> None:
        benchmark = EepyBenchmark(enabled=True, frame_count=500)
        benchmark.observe_raw_signal(True, 1.0)
        benchmark.confirm_fatigue(2.0)

        benchmark.intervention_delivered(2.25)

        self.assertEqual(benchmark.raw_to_confirmed, [1.0])
        self.assertEqual(benchmark.confirmed_to_intervention, [0.25])
        self.assertEqual(benchmark.raw_to_intervention, [1.25])

    def test_recovery_clears_failed_episode_before_next_confirmation(self) -> None:
        benchmark = EepyBenchmark(enabled=True, frame_count=500)
        benchmark.observe_raw_signal(True, 1.0)
        benchmark.confirm_fatigue(2.0)
        benchmark.fatigue_recovered()

        benchmark.observe_raw_signal(True, 10.0)
        benchmark.confirm_fatigue(11.0)
        benchmark.intervention_delivered(11.1)

        self.assertEqual(len(benchmark.raw_to_confirmed), 2)
        self.assertAlmostEqual(benchmark.confirmed_to_intervention[-1], 0.1)
        self.assertAlmostEqual(benchmark.raw_to_intervention[-1], 1.1)


if __name__ == "__main__":
    unittest.main()
