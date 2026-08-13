import unittest

from scripts.drowsiness import FatigueState


def make_state():
    return FatigueState(
        eye_grace_seconds=0.5,
        eye_ramp_seconds=1.0,
        mouth_window_seconds=5.0,
        eye_weight=0.7,
        mouth_weight=0.3,
    )


class FatigueStateTests(unittest.TestCase):
    def test_normal_blink_is_ignored(self):
        state = make_state()

        state.update(1.0, True, 0.0, 0.0)
        blink = state.update(1.0, True, 0.0, 0.4)
        reopened = state.update(0.0, False, 0.0, 0.41)

        self.assertEqual(blink.eye_evidence, 0.0)
        self.assertFalse(blink.is_drowsy)
        self.assertFalse(reopened.is_drowsy)

    def test_full_eye_severity_confirms_after_grace_and_ramp(self):
        state = make_state()

        state.update(1.0, True, 0.0, 0.0)
        before = state.update(1.0, True, 0.0, 1.49)
        confirmed = state.update(1.0, True, 0.0, 1.5)

        self.assertFalse(before.is_drowsy)
        self.assertTrue(confirmed.is_drowsy)
        self.assertTrue(confirmed.just_confirmed)

    def test_moderate_eye_severity_takes_proportionally_longer(self):
        state = make_state()

        state.update(0.5, True, 0.0, 0.0)
        before = state.update(0.5, True, 0.0, 2.49)
        confirmed = state.update(0.5, True, 0.0, 2.5)

        self.assertFalse(before.is_drowsy)
        self.assertTrue(confirmed.is_drowsy)

    def test_mouth_evidence_cannot_confirm_without_closed_eyes(self):
        state = make_state()

        update = state.update(0.0, False, 1.0, 10.0)

        self.assertFalse(update.is_drowsy)

    def test_mouth_evidence_can_corroborate_eye_closure(self):
        state = make_state()

        state.update(0.5, True, 1.0, 0.0)
        update = state.update(0.5, True, 1.0, 1.65)

        self.assertTrue(update.is_drowsy)

    def test_confirmation_is_one_shot_until_recovery(self):
        state = make_state()

        state.update(1.0, True, 0.0, 0.0)
        confirmed = state.update(1.0, True, 0.0, 1.5)
        sustained = state.update(1.0, True, 0.0, 1.6)
        recovered = state.update(0.0, False, 0.0, 1.7)

        self.assertTrue(confirmed.just_confirmed)
        self.assertFalse(sustained.just_confirmed)
        self.assertTrue(recovered.just_recovered)

    def test_missing_face_clears_timers_history_and_fatigue(self):
        state = make_state()

        state.update(1.0, True, 1.0, 0.0)
        state.update(1.0, True, 1.0, 1.5)
        missing = state.face_missing()
        new_face = state.update(1.0, True, 0.0, 2.0)

        self.assertTrue(missing.just_recovered)
        self.assertFalse(new_face.is_drowsy)
        self.assertEqual(new_face.eye_evidence, 0.0)
        self.assertEqual(new_face.mouth_evidence, 0.0)


if __name__ == "__main__":
    unittest.main()
