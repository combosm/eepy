from __future__ import annotations

import unittest

from vision.intervention import (
    InterventionAction,
    InterventionController,
    InterventionPolicy,
)


class InterventionControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = InterventionController(
            InterventionPolicy(repeat_cooldown_seconds=15.0)
        )

    def test_no_action_while_driver_is_not_drowsy(self) -> None:
        update = self.controller.update(is_drowsy=False, now=0.0)

        self.assertFalse(update.active)
        self.assertIsNone(update.episode_id)
        self.assertIsNone(update.action)

    def test_confirmation_starts_episode_and_alerts_once(self) -> None:
        initial = self.controller.update(is_drowsy=True, now=1.0)
        next_frame = self.controller.update(is_drowsy=True, now=1.1)

        self.assertEqual(initial.episode_id, 1)
        self.assertEqual(initial.escalation_level, 1)
        self.assertEqual(initial.action, InterventionAction.INITIAL_ALERT)
        self.assertTrue(initial.should_alert)
        self.assertIsNone(next_frame.action)

    def test_persistent_fatigue_repeats_only_after_cooldown(self) -> None:
        self.controller.update(is_drowsy=True, now=0.0)
        before = self.controller.update(is_drowsy=True, now=14.999)
        repeat = self.controller.update(is_drowsy=True, now=15.0)
        next_frame = self.controller.update(is_drowsy=True, now=15.1)

        self.assertIsNone(before.action)
        self.assertEqual(repeat.action, InterventionAction.REPEAT_ALERT)
        self.assertEqual(repeat.escalation_level, 2)
        self.assertIsNone(next_frame.action)

    def test_repeat_cooldown_runs_from_the_previous_alert(self) -> None:
        self.controller.update(is_drowsy=True, now=0.0)
        self.controller.update(is_drowsy=True, now=15.0)
        before = self.controller.update(is_drowsy=True, now=29.9)
        repeat = self.controller.update(is_drowsy=True, now=30.0)

        self.assertIsNone(before.action)
        self.assertEqual(repeat.action, InterventionAction.REPEAT_ALERT)
        self.assertEqual(repeat.escalation_level, 3)

    def test_recovery_is_emitted_once_and_closes_episode(self) -> None:
        self.controller.update(is_drowsy=True, now=0.0)
        recovered = self.controller.update(is_drowsy=False, now=2.0)
        next_frame = self.controller.update(is_drowsy=False, now=2.1)

        self.assertFalse(recovered.active)
        self.assertEqual(recovered.episode_id, 1)
        self.assertEqual(recovered.action, InterventionAction.RECOVERED)
        self.assertFalse(recovered.should_alert)
        self.assertIsNone(next_frame.action)
        self.assertIsNone(next_frame.episode_id)

    def test_missing_observation_does_not_claim_recovery(self) -> None:
        self.controller.update(is_drowsy=True, now=0.0)

        missing = self.controller.update(is_drowsy=None, now=2.0)

        self.assertTrue(missing.active)
        self.assertEqual(missing.episode_id, 1)
        self.assertIsNone(missing.action)

    def test_missing_observation_can_repeat_active_safety_alert(self) -> None:
        self.controller.update(is_drowsy=True, now=0.0)

        missing = self.controller.update(is_drowsy=None, now=15.0)

        self.assertEqual(missing.action, InterventionAction.REPEAT_ALERT)

    def test_new_fatigue_after_recovery_gets_new_episode_id(self) -> None:
        first = self.controller.update(is_drowsy=True, now=0.0)
        self.controller.update(is_drowsy=False, now=2.0)
        second = self.controller.update(is_drowsy=True, now=3.0)

        self.assertEqual(first.episode_id, 1)
        self.assertEqual(second.episode_id, 2)
        self.assertEqual(second.action, InterventionAction.INITIAL_ALERT)

    def test_non_finite_or_backward_time_is_rejected(self) -> None:
        self.controller.update(is_drowsy=False, now=2.0)

        with self.assertRaises(ValueError):
            self.controller.update(is_drowsy=False, now=1.0)
        with self.assertRaises(ValueError):
            self.controller.update(is_drowsy=False, now=float("nan"))

    def test_non_positive_cooldown_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            InterventionController(
                InterventionPolicy(repeat_cooldown_seconds=0.0)
            )


if __name__ == "__main__":
    unittest.main()
