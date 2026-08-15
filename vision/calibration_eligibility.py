"""Per-frame technical quality checks for passive calibration.

This module does not decide whether the driver is awake. It only classifies
whether a frame is technically suitable for the future rolling awake-state
gate. Measurements are normalised to the frame or detected face so decisions
do not depend on a particular camera resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


LEFT_EYE = np.array([36, 37, 38, 39, 40, 41])
RIGHT_EYE = np.array([42, 43, 44, 45, 46, 47])
MOUTH = np.array([48, 50, 52, 54, 56, 58])
REQUIRED_LANDMARKS = np.concatenate((LEFT_EYE, RIGHT_EYE, MOUTH, [8, 30]))


class FrameQuality(str, Enum):
    VALID = "valid"
    DEGRADED = "degraded"
    REJECTED = "rejected"


class FrameQualityReason(str, Enum):
    NO_FACE = "no_face"
    INVALID_FRAME_SIZE = "invalid_frame_size"
    INVALID_FACE_BOX = "invalid_face_box"
    LOW_FACE_CONFIDENCE = "low_face_confidence"
    FACE_SMALL_IN_FRAME = "face_small_in_frame"
    FACE_BOX_CLIPPED = "face_box_clipped"
    INVALID_LANDMARKS = "invalid_landmarks"
    REQUIRED_LANDMARKS_OUTSIDE_FRAME = "required_landmarks_outside_frame"
    IMPLAUSIBLE_LANDMARK_ORDER = "implausible_landmark_order"
    COLLAPSED_LANDMARK_GEOMETRY = "collapsed_landmark_geometry"
    INVALID_EAR = "invalid_ear"
    INVALID_MAR = "invalid_mar"
    HEAD_POSE_UNAVAILABLE = "head_pose_unavailable"
    MODERATE_HEAD_POSE = "moderate_head_pose"
    EXTREME_HEAD_POSE = "extreme_head_pose"


@dataclass(frozen=True)
class HeadPose:
    pitch_degrees: float
    yaw_degrees: float
    roll_degrees: float


@dataclass(frozen=True)
class FrameEligibilityPolicy:
    """Named, resolution-independent boundaries for frame quality.

    The face-size boundaries describe preferred image occupancy, not a hard
    physiological requirement. Falling below them degrades the frame. Head
    pose has a preferred band and a wider hard-rejection band.
    """

    minimum_detection_confidence: float = 0.6
    preferred_face_width_fraction: float = 0.18
    preferred_face_height_fraction: float = 0.24
    preferred_visible_face_fraction: float = 0.90
    preferred_absolute_yaw_degrees: float = 25.0
    preferred_absolute_pitch_degrees: float = 20.0
    preferred_absolute_roll_degrees: float = 20.0
    maximum_absolute_yaw_degrees: float = 45.0
    maximum_absolute_pitch_degrees: float = 35.0
    maximum_absolute_roll_degrees: float = 35.0


@dataclass(frozen=True)
class FrameEligibility:
    quality: FrameQuality
    reasons: tuple[FrameQualityReason, ...]

    @property
    def may_enter_rolling_gate(self) -> bool:
        return self.quality is not FrameQuality.REJECTED


DEFAULT_FRAME_ELIGIBILITY_POLICY = FrameEligibilityPolicy()


def no_face_eligibility() -> FrameEligibility:
    return FrameEligibility(FrameQuality.REJECTED, (FrameQualityReason.NO_FACE,))


def _box_area(box: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)


def _has_valid_ratio(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value >= 0.0


def _head_pose_reasons(
    pose: HeadPose | None,
    policy: FrameEligibilityPolicy,
) -> tuple[list[FrameQualityReason], list[FrameQualityReason]]:
    degraded: list[FrameQualityReason] = []
    rejected: list[FrameQualityReason] = []

    if pose is None:
        degraded.append(FrameQualityReason.HEAD_POSE_UNAVAILABLE)
        return degraded, rejected

    angles = (pose.pitch_degrees, pose.yaw_degrees, pose.roll_degrees)
    if not all(math.isfinite(angle) for angle in angles):
        degraded.append(FrameQualityReason.HEAD_POSE_UNAVAILABLE)
        return degraded, rejected

    if (
        abs(pose.yaw_degrees) > policy.maximum_absolute_yaw_degrees
        or abs(pose.pitch_degrees) > policy.maximum_absolute_pitch_degrees
        or abs(pose.roll_degrees) > policy.maximum_absolute_roll_degrees
    ):
        rejected.append(FrameQualityReason.EXTREME_HEAD_POSE)
    elif (
        abs(pose.yaw_degrees) > policy.preferred_absolute_yaw_degrees
        or abs(pose.pitch_degrees) > policy.preferred_absolute_pitch_degrees
        or abs(pose.roll_degrees) > policy.preferred_absolute_roll_degrees
    ):
        degraded.append(FrameQualityReason.MODERATE_HEAD_POSE)

    return degraded, rejected


def assess_frame_eligibility(
    *,
    frame_width: int,
    frame_height: int,
    detection_confidence: float,
    raw_face_box: tuple[int, int, int, int],
    clipped_face_box: tuple[int, int, int, int],
    landmarks: np.ndarray,
    ear: float | None,
    mar: float | None,
    head_pose: HeadPose | None,
    policy: FrameEligibilityPolicy = DEFAULT_FRAME_ELIGIBILITY_POLICY,
) -> FrameEligibility:
    """Classify one detected driver frame as valid, degraded, or rejected."""
    degraded: list[FrameQualityReason] = []
    rejected: list[FrameQualityReason] = []

    if frame_width <= 0 or frame_height <= 0:
        rejected.append(FrameQualityReason.INVALID_FRAME_SIZE)

    raw_area = _box_area(raw_face_box)
    clipped_area = _box_area(clipped_face_box)
    if raw_area <= 0 or clipped_area <= 0:
        rejected.append(FrameQualityReason.INVALID_FACE_BOX)

    if (
        not math.isfinite(detection_confidence)
        or detection_confidence < policy.minimum_detection_confidence
    ):
        degraded.append(FrameQualityReason.LOW_FACE_CONFIDENCE)

    if frame_width > 0 and frame_height > 0 and clipped_area > 0:
        x0, y0, x1, y1 = clipped_face_box
        face_width_fraction = (x1 - x0) / frame_width
        face_height_fraction = (y1 - y0) / frame_height
        if (
            face_width_fraction < policy.preferred_face_width_fraction
            or face_height_fraction < policy.preferred_face_height_fraction
        ):
            degraded.append(FrameQualityReason.FACE_SMALL_IN_FRAME)

    if raw_area > 0:
        visible_fraction = clipped_area / raw_area
        if visible_fraction < policy.preferred_visible_face_fraction:
            degraded.append(FrameQualityReason.FACE_BOX_CLIPPED)

    landmarks_valid = (
        isinstance(landmarks, np.ndarray)
        and landmarks.shape == (68, 2)
        and np.isfinite(landmarks).all()
    )
    if not landmarks_valid:
        rejected.append(FrameQualityReason.INVALID_LANDMARKS)
    else:
        required = landmarks[REQUIRED_LANDMARKS]
        required_inside_frame = bool(
            np.all(required[:, 0] >= 0)
            and np.all(required[:, 0] < frame_width)
            and np.all(required[:, 1] >= 0)
            and np.all(required[:, 1] < frame_height)
        )
        if not required_inside_frame:
            rejected.append(FrameQualityReason.REQUIRED_LANDMARKS_OUTSIDE_FRAME)

        left_eye_center = landmarks[LEFT_EYE].mean(axis=0)
        right_eye_center = landmarks[RIGHT_EYE].mean(axis=0)
        mouth_center = landmarks[MOUTH].mean(axis=0)
        nose = landmarks[30]
        chin = landmarks[8]
        if not (
            left_eye_center[1] < mouth_center[1]
            and right_eye_center[1] < mouth_center[1]
            and nose[1] < mouth_center[1]
            and chin[1] > mouth_center[1]
        ):
            rejected.append(FrameQualityReason.IMPLAUSIBLE_LANDMARK_ORDER)

        if clipped_area > 0:
            x0, _, x1, _ = clipped_face_box
            face_width = x1 - x0
            feature_spans = (
                np.linalg.norm(landmarks[LEFT_EYE][0] - landmarks[LEFT_EYE][3]),
                np.linalg.norm(landmarks[RIGHT_EYE][0] - landmarks[RIGHT_EYE][3]),
                np.linalg.norm(landmarks[MOUTH][0] - landmarks[MOUTH][3]),
            )
            relative_spans = [span / face_width for span in feature_spans]
            if not all(math.isfinite(span) and span > 0.0 for span in relative_spans):
                rejected.append(FrameQualityReason.COLLAPSED_LANDMARK_GEOMETRY)

    if not _has_valid_ratio(ear):
        rejected.append(FrameQualityReason.INVALID_EAR)
    if not _has_valid_ratio(mar):
        rejected.append(FrameQualityReason.INVALID_MAR)

    pose_degraded, pose_rejected = _head_pose_reasons(head_pose, policy)
    degraded.extend(pose_degraded)
    rejected.extend(pose_rejected)

    if rejected:
        return FrameEligibility(FrameQuality.REJECTED, tuple(rejected + degraded))
    if degraded:
        return FrameEligibility(FrameQuality.DEGRADED, tuple(degraded))
    return FrameEligibility(FrameQuality.VALID, ())
