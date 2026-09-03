from __future__ import annotations

import unittest

from vision.awake_state_gate import (
    AwakeGateReason,
    AwakeStateGate,
)
from vision.calibration_eligibility import FrameEligibility, FrameQuality


VALID = FrameEligibility(FrameQuality.VALID, ())
DEGRADED = FrameEligibility(FrameQuality.DEGRADED, ())
REJECTED = FrameEligibility(FrameQuality.REJECTED, ())


def feed(
    gate: AwakeStateGate,
    start: float,
    end: float,
    *,
    step: float = 0.5,
    frame: FrameEligibility = VALID,
    ear: float = 0.36,
    mar: float = 0.32,
):
    update = None
    now = start
    while now <= end + 1e-9:
        update = gate.observe(
            frame=frame,
            ear=ear,
            mar=mar,
            fatigue_suspected=False,
            now=now,
        )
        now += step
    return update


class AwakeStateGateTests(unittest.TestCase):
    def test_startup_is_not_assumed_awake(self) -> None:
        update = feed(AwakeStateGate(), 0.0, 4.0)

        self.assertFalse(update.eligible)
        self.assertIn(AwakeGateReason.INSUFFICIENT_HISTORY, update.reasons)

    def test_stable_valid_history_becomes_eligible(self) -> None:
        update = feed(AwakeStateGate(), 0.0, 8.0)

        self.assertTrue(update.eligible)
        self.assertTrue(update.current_sample_approved)
        self.assertEqual(update.weighted_evidence_seconds, 8.0)

    def test_degraded_frames_provide_weaker_evidence(self) -> None:
        update = feed(AwakeStateGate(), 0.0, 8.0, frame=DEGRADED)

        self.assertFalse(update.eligible)
        self.assertEqual(update.weighted_evidence_seconds, 4.0)
        self.assertIn(
            AwakeGateReason.INSUFFICIENT_USABLE_EVIDENCE,
            update.reasons,
        )

    def test_short_rejected_gap_delays_but_does_not_reset_history(self) -> None:
        gate = AwakeStateGate()
        feed(gate, 0.0, 3.5)
        gate.observe(
            frame=REJECTED,
            ear=None,
            mar=None,
            fatigue_suspected=False,
            now=4.0,
        )
        update = feed(gate, 4.5, 8.5)

        self.assertTrue(update.eligible)

    def test_recent_gap_in_usable_observations_blocks_eligibility(self) -> None:
        gate = AwakeStateGate()
        feed(gate, 0.0, 8.0)
        gate.observe(
            frame=REJECTED,
            ear=None,
            mar=None,
            fatigue_suspected=False,
            now=8.5,
        )
        update = gate.observe(
            frame=REJECTED,
            ear=None,
            mar=None,
            fatigue_suspected=False,
            now=9.5,
        )

        self.assertFalse(update.eligible)
        self.assertIn(AwakeGateReason.RECENT_USABLE_GAP, update.reasons)

    def test_unstable_ear_blocks_eligibility(self) -> None:
        gate = AwakeStateGate()
        update = None
        for index in range(17):
            update = gate.observe(
                frame=VALID,
                ear=0.31 if index % 2 else 0.45,
                mar=0.32,
                fatigue_suspected=False,
                now=index * 0.5,
            )

        self.assertFalse(update.eligible)
        self.assertIn(AwakeGateReason.UNSTABLE_EAR, update.reasons)

    def test_meaningful_downward_ear_trend_blocks_eligibility(self) -> None:
        gate = AwakeStateGate()
        update = None
        for index in range(17):
            now = index * 0.5
            update = gate.observe(
                frame=VALID,
                ear=0.40 - 0.005 * now,
                mar=0.32,
                fatigue_suspected=False,
                now=now,
            )

        self.assertFalse(update.eligible)
        self.assertIn(AwakeGateReason.DOWNWARD_EAR_TREND, update.reasons)

    def test_blink_like_drop_recovers_without_quarantining_history(self) -> None:
        gate = AwakeStateGate()
        feed(gate, 0.0, 8.0)
        blink = gate.observe(
            frame=VALID,
            ear=0.25,
            mar=0.32,
            fatigue_suspected=False,
            now=8.2,
        )
        recovered = gate.observe(
            frame=VALID,
            ear=0.36,
            mar=0.32,
            fatigue_suspected=False,
            now=8.4,
        )

        self.assertFalse(blink.eligible)
        self.assertIn(AwakeGateReason.EYE_CLOSURE_IN_PROGRESS, blink.reasons)
        self.assertTrue(recovered.eligible)
        self.assertIsNone(recovered.quarantined_since)

    def test_prolonged_closure_quarantines_recent_history_and_freezes(self) -> None:
        gate = AwakeStateGate()
        feed(gate, 0.0, 8.0)
        gate.observe(
            frame=VALID,
            ear=0.25,
            mar=0.32,
            fatigue_suspected=False,
            now=8.1,
        )
        update = gate.observe(
            frame=VALID,
            ear=0.25,
            mar=0.32,
            fatigue_suspected=False,
            now=8.6,
        )

        self.assertFalse(update.eligible)
        self.assertAlmostEqual(update.quarantined_since, 6.6)
        self.assertIn(AwakeGateReason.PROLONGED_EYE_CLOSURE, update.reasons)
        self.assertIn(AwakeGateReason.RECOVERY_FREEZE, update.reasons)

    def test_stable_low_ear_is_not_accepted_as_proof_of_wakefulness(self) -> None:
        update = feed(AwakeStateGate(), 0.0, 8.0, ear=0.29)

        self.assertFalse(update.eligible)
        self.assertIn(AwakeGateReason.PROLONGED_EYE_CLOSURE, update.reasons)

    def test_confirmed_fatigue_quarantines_even_with_open_eye_measurement(self) -> None:
        gate = AwakeStateGate()
        feed(gate, 0.0, 8.0)
        update = gate.observe(
            frame=VALID,
            ear=0.36,
            mar=0.32,
            fatigue_suspected=True,
            now=8.1,
        )

        self.assertFalse(update.eligible)
        self.assertIn(AwakeGateReason.SUSPECTED_FATIGUE, update.reasons)
        self.assertFalse(update.current_sample_approved)

    def test_sustained_mouth_opening_is_suspicious(self) -> None:
        gate = AwakeStateGate()
        feed(gate, 0.0, 8.0)
        gate.observe(
            frame=VALID,
            ear=0.36,
            mar=0.7,
            fatigue_suspected=False,
            now=8.1,
        )
        update = gate.observe(
            frame=VALID,
            ear=0.36,
            mar=0.7,
            fatigue_suspected=False,
            now=9.6,
        )

        self.assertFalse(update.eligible)
        self.assertIn(AwakeGateReason.SUSPICIOUS_MOUTH_BEHAVIOUR, update.reasons)

    def test_repeated_mouth_opening_is_suspicious(self) -> None:
        gate = AwakeStateGate()
        feed(gate, 0.0, 8.0)
        for now in (8.1, 8.5, 8.9):
            gate.observe(
                frame=VALID,
                ear=0.36,
                mar=0.7,
                fatigue_suspected=False,
                now=now,
            )
            update = gate.observe(
                frame=VALID,
                ear=0.36,
                mar=0.32,
                fatigue_suspected=False,
                now=now + 0.1,
            )

        self.assertFalse(update.eligible)
        self.assertIn(AwakeGateReason.RECOVERY_FREEZE, update.reasons)

    def test_non_monotonic_time_resets_history(self) -> None:
        gate = AwakeStateGate()
        feed(gate, 0.0, 8.0)
        update = gate.observe(
            frame=VALID,
            ear=0.36,
            mar=0.32,
            fatigue_suspected=False,
            now=7.0,
        )

        self.assertFalse(update.eligible)
        self.assertIn(AwakeGateReason.NON_MONOTONIC_TIME, update.reasons)


if __name__ == "__main__":
    unittest.main()
