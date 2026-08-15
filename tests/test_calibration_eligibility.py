from __future__ import annotations

import unittest

import numpy as np

from vision.calibration_eligibility import (
    FrameQuality,
    FrameQualityReason,
    HeadPose,
    assess_frame_eligibility,
    no_face_eligibility,
)


def make_landmarks(scale: float = 1.0) -> np.ndarray:
    landmarks = np.full((68, 2), (320.0, 260.0))
    landmarks[36:42] = [
        (240, 200), (250, 194), (260, 194),
        (270, 200), (260, 206), (250, 206),
    ]
    landmarks[42:48] = [
        (370, 200), (380, 194), (390, 194),
        (400, 200), (390, 206), (380, 206),
    ]
    landmarks[[48, 50, 52, 54, 56, 58]] = [
        (270, 320), (290, 310), (330, 310),
        (350, 320), (330, 330), (290, 330),
    ]
    landmarks[30] = (320, 260)
    landmarks[8] = (320, 400)
    return landmarks * scale


def assess(
    *,
    scale: float = 1.0,
    confidence: float = 0.9,
    raw_box: tuple[int, int, int, int] | None = None,
    clipped_box: tuple[int, int, int, int] | None = None,
    landmarks: np.ndarray | None = None,
    ear: float | None = 0.3,
    mar: float | None = 0.3,
    pose: HeadPose | None = HeadPose(0.0, 0.0, 0.0),
):
    raw_box = raw_box or tuple(int(value * scale) for value in (160, 100, 480, 440))
    clipped_box = clipped_box or raw_box
    return assess_frame_eligibility(
        frame_width=int(640 * scale),
        frame_height=int(480 * scale),
        detection_confidence=confidence,
        raw_face_box=raw_box,
        clipped_face_box=clipped_box,
        landmarks=make_landmarks(scale) if landmarks is None else landmarks,
        ear=ear,
        mar=mar,
        head_pose=pose,
    )


class FrameEligibilityTests(unittest.TestCase):
    def test_missing_face_is_rejected(self) -> None:
        result = no_face_eligibility()

        self.assertEqual(result.quality, FrameQuality.REJECTED)
        self.assertEqual(result.reasons, (FrameQualityReason.NO_FACE,))
        self.assertFalse(result.may_enter_rolling_gate)

    def test_technically_sound_frame_is_valid(self) -> None:
        result = assess()

        self.assertEqual(result.quality, FrameQuality.VALID)
        self.assertEqual(result.reasons, ())
        self.assertTrue(result.may_enter_rolling_gate)

    def test_equivalent_geometry_has_same_result_at_different_resolutions(self) -> None:
        standard = assess(scale=1.0)
        doubled = assess(scale=2.0)

        self.assertEqual(standard, doubled)

    def test_small_face_is_degraded_instead_of_rejected(self) -> None:
        result = assess(
            raw_box=(280, 180, 360, 300),
            clipped_box=(280, 180, 360, 300),
        )

        self.assertEqual(result.quality, FrameQuality.DEGRADED)
        self.assertIn(FrameQualityReason.FACE_SMALL_IN_FRAME, result.reasons)
        self.assertTrue(result.may_enter_rolling_gate)

    def test_low_confidence_is_degraded_instead_of_rejected(self) -> None:
        result = assess(confidence=0.49)

        self.assertEqual(result.quality, FrameQuality.DEGRADED)
        self.assertIn(FrameQualityReason.LOW_FACE_CONFIDENCE, result.reasons)

    def test_clipped_face_box_is_degraded_when_landmarks_remain_visible(self) -> None:
        result = assess(
            raw_box=(-100, 100, 480, 440),
            clipped_box=(0, 100, 480, 440),
        )

        self.assertEqual(result.quality, FrameQuality.DEGRADED)
        self.assertIn(FrameQualityReason.FACE_BOX_CLIPPED, result.reasons)

    def test_unusual_but_finite_ear_is_not_rejected(self) -> None:
        result = assess(ear=0.05)

        self.assertEqual(result.quality, FrameQuality.VALID)
        self.assertNotIn(FrameQualityReason.INVALID_EAR, result.reasons)

    def test_invalid_ratio_is_rejected(self) -> None:
        result = assess(ear=None, mar=float("nan"))

        self.assertEqual(result.quality, FrameQuality.REJECTED)
        self.assertIn(FrameQualityReason.INVALID_EAR, result.reasons)
        self.assertIn(FrameQualityReason.INVALID_MAR, result.reasons)

    def test_malformed_landmark_array_is_rejected(self) -> None:
        result = assess(landmarks=np.zeros((67, 2)))

        self.assertEqual(result.quality, FrameQuality.REJECTED)
        self.assertIn(FrameQualityReason.INVALID_LANDMARKS, result.reasons)

    def test_invalid_face_box_is_rejected(self) -> None:
        result = assess(
            raw_box=(100, 100, 100, 300),
            clipped_box=(100, 100, 100, 300),
        )

        self.assertEqual(result.quality, FrameQuality.REJECTED)
        self.assertIn(FrameQualityReason.INVALID_FACE_BOX, result.reasons)

    def test_required_landmark_outside_frame_is_rejected(self) -> None:
        landmarks = make_landmarks()
        landmarks[36] = (-1, 200)

        result = assess(landmarks=landmarks)

        self.assertEqual(result.quality, FrameQuality.REJECTED)
        self.assertIn(
            FrameQualityReason.REQUIRED_LANDMARKS_OUTSIDE_FRAME,
            result.reasons,
        )

    def test_impossible_vertical_landmark_order_is_rejected(self) -> None:
        landmarks = make_landmarks()
        landmarks[8] = (320, 250)

        result = assess(landmarks=landmarks)

        self.assertEqual(result.quality, FrameQuality.REJECTED)
        self.assertIn(
            FrameQualityReason.IMPLAUSIBLE_LANDMARK_ORDER,
            result.reasons,
        )

    def test_collapsed_eye_geometry_is_rejected(self) -> None:
        landmarks = make_landmarks()
        landmarks[39] = landmarks[36]

        result = assess(landmarks=landmarks)

        self.assertEqual(result.quality, FrameQuality.REJECTED)
        self.assertIn(
            FrameQualityReason.COLLAPSED_LANDMARK_GEOMETRY,
            result.reasons,
        )

    def test_unavailable_pose_is_degraded(self) -> None:
        result = assess(pose=None)

        self.assertEqual(result.quality, FrameQuality.DEGRADED)
        self.assertIn(FrameQualityReason.HEAD_POSE_UNAVAILABLE, result.reasons)

    def test_moderate_pose_is_degraded(self) -> None:
        result = assess(pose=HeadPose(0.0, 30.0, 0.0))

        self.assertEqual(result.quality, FrameQuality.DEGRADED)
        self.assertIn(FrameQualityReason.MODERATE_HEAD_POSE, result.reasons)

    def test_extreme_pose_is_rejected(self) -> None:
        result = assess(pose=HeadPose(0.0, 50.0, 0.0))

        self.assertEqual(result.quality, FrameQuality.REJECTED)
        self.assertIn(FrameQualityReason.EXTREME_HEAD_POSE, result.reasons)


if __name__ == "__main__":
    unittest.main()
