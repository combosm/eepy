from __future__ import annotations

from collections.abc import Iterator

import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from flask_socketio import SocketIO
from vision.benchmark import (
    BENCHMARK_ENABLED,
    BENCHMARK_FRAME_COUNT,
    EepyBenchmark,
)
from vision.calibration_eligibility import (
    FrameEligibility,
    FrameEligibilityPolicy,
    HeadPose,
    assess_frame_eligibility,
    no_face_eligibility,
)
from vision.awake_state_gate import AwakeGateUpdate, AwakeStateGate
from vision.drowsiness import FatigueState
from vision.intervention import (
    InterventionAction,
    InterventionController,
    InterventionPolicy,
    InterventionUpdate,
)
from vision.local_alarm import LocalAlarm
from vision.monitoring_worker import MonitoringWorker

socketio = SocketIO()
### TODO
# - Calibration: per-user EAR/MAR thresholds instead of the fixed constants below.
# - Glasses detection.
# - Detect a hand covering the mouth, and distinguish that from a hand just resting near it
#   (e.g. while laughing) - ties into emotion detection below.
# - Emotion detection: laughing can close the eyes and read as drowsiness; needs to be told
#   apart from real fatigue.
# - Validate the EAR/MAR normalization bounds against a wider range of eyes than they were
#   tuned on.

# OpenCV Face Detector (DNN)
face_net = cv2.dnn.readNetFromTensorflow("models/opencv_face_detector_uint8.pb", "models/opencv_face_detector.pbtxt")

# dlib face landmarks
predictor_path = "models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# indices for eye and mouth landmarks (0-indexed)
LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
MOUTH = [48, 50, 52, 54, 56, 58]

FACE_DETECTION_CONFIDENCE_THRESHOLD = 0.5
CALIBRATION_PREFERRED_FACE_CONFIDENCE = 0.6
FRAME_ELIGIBILITY_POLICY = FrameEligibilityPolicy(
    minimum_detection_confidence=CALIBRATION_PREFERRED_FACE_CONFIDENCE,
)

# Normalisation bounds are centered on the existing binary fatigue thresholds.
EAR_CLOSED_BOUND = 0.2
EAR_DROWSY_THRESHOLD = 0.3
EAR_OPEN_BOUND = 0.4
MAR_RESTING_BOUND = 0.3
MAR_DROWSY_THRESHOLD = 0.6
MAR_YAWN_BOUND = 0.9
EYE_SCORE_WEIGHT = 0.7
MOUTH_SCORE_WEIGHT = 0.3
EYE_CLOSURE_GRACE_SECONDS = 0.5
EYE_CLOSURE_RAMP_SECONDS = 1.0
MOUTH_EVIDENCE_WINDOW_SECONDS = 5.0
INTERVENTION_REPEAT_COOLDOWN_SECONDS = 15.0

data_store = {
    "EAR": 0,
    "MAR": 0,
    "is_drowsy": False,
    "calibration_frame_quality": "rejected",
    "calibration_frame_reasons": ["no_face"],
    "calibration_awake_eligible": False,
    "calibration_awake_reasons": ["insufficient_history"],
    "calibration_evidence_seconds": 0.0,
    "calibration_history_seconds": 0.0,
    "calibration_sample_approved": False,
    "intervention_active": False,
    "intervention_episode_id": None,
    "intervention_escalation_level": 0,
    "intervention_action": None,
    "intervention_message": "",
    "local_alarm_available": False,
    "local_alarm_error": None,
    "camera_running": False,
    "camera_error": None,
    "ai_response": "",
}


def eye_aspect_ratio(eye: np.ndarray) -> float | None:
    """calculate Eye Aspect Ratio (EAR)"""
    A = distance.euclidean(eye[1], eye[5])  # vertical
    B = distance.euclidean(eye[2], eye[4])  # vertical
    C = distance.euclidean(eye[0], eye[3])  # horizontal
    if not np.isfinite([A, B, C]).all() or C <= 0.0:
        return None
    ear = (A + B) / (2.0 * C)
    if not np.isfinite(ear):
        return None
    return round(float(ear), 3)


def mouth_aspect_ratio(mouth: np.ndarray) -> float | None:
    """calculate Mouth Aspect Ratio (MAR)"""
    A = distance.euclidean(mouth[1], mouth[5])  # vertical
    B = distance.euclidean(mouth[2], mouth[4])  # vertical
    C = distance.euclidean(mouth[0], mouth[3])  # horizontal
    if not np.isfinite([A, B, C]).all() or C <= 0.0:
        return None
    mar = (A + B) / (2.0 * C)
    if not np.isfinite(mar):
        return None
    return round(float(mar), 3)


HEAD_POSE_LANDMARKS = np.array([30, 8, 36, 45, 48, 54])
HEAD_POSE_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)


def estimate_head_pose(
    landmarks: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> HeadPose | None:
    """Estimate approximate face orientation from six 2D landmarks."""
    image_points = landmarks[HEAD_POSE_LANDMARKS].astype(np.float64)
    focal_length = float(frame_width)
    camera_matrix = np.array(
        [
            [focal_length, 0.0, frame_width / 2.0],
            [0.0, focal_length, frame_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.zeros((4, 1), dtype=np.float64)

    try:
        success, rotation_vector, _ = cv2.solvePnP(
            HEAD_POSE_MODEL_POINTS,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        angles, *_ = cv2.RQDecomp3x3(rotation_matrix)
    except cv2.error:
        return None

    pitch, yaw, roll = (float(angle) for angle in angles)
    if not np.isfinite([pitch, yaw, roll]).all():
        return None
    return HeadPose(pitch, yaw, roll)


def _store_frame_eligibility(quality: FrameEligibility) -> None:
    data_store["calibration_frame_quality"] = quality.quality.value
    data_store["calibration_frame_reasons"] = [
        reason.value for reason in quality.reasons
    ]


def _store_awake_gate(update: AwakeGateUpdate) -> None:
    data_store["calibration_awake_eligible"] = bool(update.eligible)
    data_store["calibration_awake_reasons"] = [
        reason.value for reason in update.reasons
    ]
    data_store["calibration_evidence_seconds"] = update.weighted_evidence_seconds
    data_store["calibration_history_seconds"] = update.history_seconds
    data_store["calibration_sample_approved"] = bool(
        update.current_sample_approved
    )


def _store_intervention(
    update: InterventionUpdate,
    alarm: LocalAlarm,
) -> bool:
    """Publish intervention state and synchronously start any local alarm."""
    data_store["intervention_active"] = bool(update.active)
    data_store["intervention_episode_id"] = update.episode_id
    data_store["intervention_escalation_level"] = update.escalation_level
    data_store["intervention_action"] = (
        update.action.value if update.action is not None else None
    )
    data_store["local_alarm_available"] = alarm.available

    if update.action is InterventionAction.INITIAL_ALERT:
        data_store["intervention_message"] = (
            "Fatigue detected. Wake up and pull over safely."
        )
    elif update.action is InterventionAction.REPEAT_ALERT:
        data_store["intervention_message"] = (
            "Fatigue persists. Pull over safely now."
        )
    elif update.action is InterventionAction.RECOVERED:
        data_store["intervention_message"] = "Driver appears responsive again."
        data_store["local_alarm_error"] = alarm.error
        return False
    elif not update.active:
        data_store["intervention_message"] = ""

    if not update.should_alert:
        data_store["local_alarm_error"] = alarm.error
        return False

    alarm_result = alarm.play()
    data_store["local_alarm_available"] = alarm.available
    data_store["local_alarm_error"] = alarm_result.error
    return alarm_result.delivered


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(value, upper))


def normalize_eye_closure(ear: float) -> float:
    """map open-to-closed EAR values onto a clamped [0, 1] score"""
    score = (EAR_OPEN_BOUND - ear) / (EAR_OPEN_BOUND - EAR_CLOSED_BOUND)
    return clamp(score)


def normalize_mouth_opening(mar: float) -> float:
    """map resting-to-yawning MAR values onto a clamped [0, 1] score"""
    score = (mar - MAR_RESTING_BOUND) / (MAR_YAWN_BOUND - MAR_RESTING_BOUND)
    return clamp(score)


class CameraPipeline:
    """Stateful inference pipeline owned by the single monitoring worker."""

    def __init__(self) -> None:
        self.benchmark = EepyBenchmark(BENCHMARK_ENABLED, BENCHMARK_FRAME_COUNT)
        self.fatigue_state = FatigueState(
            eye_grace_seconds=EYE_CLOSURE_GRACE_SECONDS,
            eye_ramp_seconds=EYE_CLOSURE_RAMP_SECONDS,
            mouth_window_seconds=MOUTH_EVIDENCE_WINDOW_SECONDS,
            eye_weight=EYE_SCORE_WEIGHT,
            mouth_weight=MOUTH_SCORE_WEIGHT,
        )
        self.awake_state_gate = AwakeStateGate()
        self.intervention_controller = InterventionController(
            InterventionPolicy(
                repeat_cooldown_seconds=INTERVENTION_REPEAT_COOLDOWN_SECONDS,
            )
        )
        self.local_alarm = LocalAlarm()
        data_store["local_alarm_available"] = self.local_alarm.available
        data_store["local_alarm_error"] = self.local_alarm.error

    def source_unavailable(self, observed_at: float) -> None:
        """Preserve active interventions while camera recovery is uncertain."""
        self.fatigue_state.face_missing()
        data_store["is_drowsy"] = False
        frame_eligibility = no_face_eligibility()
        _store_frame_eligibility(frame_eligibility)
        _store_awake_gate(
            self.awake_state_gate.observe(
                frame=frame_eligibility,
                ear=None,
                mar=None,
                fatigue_suspected=False,
                now=observed_at,
            )
        )
        intervention = self.intervention_controller.update(
            is_drowsy=None,
            now=observed_at,
        )
        local_alarm_delivered = _store_intervention(
            intervention,
            self.local_alarm,
        )
        if local_alarm_delivered:
            self.benchmark.intervention_delivered(self.benchmark.now())
        self.benchmark.observe_raw_signal(False, observed_at)
        socketio.emit("update_data", dict(data_store))

    def process(self, frame: np.ndarray, loop_started_at: float) -> bytes:
        """Run inference once, update local state, and return one annotated JPEG."""
        processing_started_at = self.benchmark.now()
        non_vision_seconds = 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame,
            1.0,
            (300, 300),
            (104, 177, 123),
            False,
            False,
        )
        face_net.setInput(blob)
        detections = face_net.forward()

        driver_detection = None
        driver_area = 0
        for index in range(detections.shape[2]):
            confidence = detections[0, 0, index, 2]
            if confidence <= FACE_DETECTION_CONFIDENCE_THRESHOLD:
                continue
            box = detections[0, 0, index, 3:7] * np.array([w, h, w, h])
            raw_box = tuple(int(value) for value in box)
            raw_x, raw_y, raw_x1, raw_y1 = raw_box
            x, y = max(0, raw_x), max(0, raw_y)
            x1, y1 = min(w - 1, raw_x1), min(h - 1, raw_y1)
            area = max(0, x1 - x) * max(0, y1 - y)
            if area > driver_area:
                driver_area = area
                driver_detection = {
                    "confidence": float(confidence),
                    "raw_box": raw_box,
                    "clipped_box": (x, y, x1, y1),
                }

        landmarks = None
        if driver_detection is None:
            self.fatigue_state.face_missing()
            data_store["is_drowsy"] = False
            frame_eligibility = no_face_eligibility()
            _store_frame_eligibility(frame_eligibility)
            observed_at = self.benchmark.now()
            _store_awake_gate(
                self.awake_state_gate.observe(
                    frame=frame_eligibility,
                    ear=None,
                    mar=None,
                    fatigue_suspected=False,
                    now=observed_at,
                )
            )
            intervention = self.intervention_controller.update(
                is_drowsy=None,
                now=observed_at,
            )
            self.benchmark.observe_raw_signal(False, observed_at)
        else:
            clipped_face_box = driver_detection["clipped_box"]
            face_rect = dlib.rectangle(*clipped_face_box)
            shape = predictor(gray, face_rect)
            landmarks = np.array(
                [[shape.part(i).x, shape.part(i).y] for i in range(68)]
            )

            left_ear = eye_aspect_ratio(landmarks[LEFT_EYE])
            right_ear = eye_aspect_ratio(landmarks[RIGHT_EYE])
            ear = (
                None
                if left_ear is None or right_ear is None
                else round((left_ear + right_ear) / 2.0, 3)
            )
            mar = mouth_aspect_ratio(landmarks[MOUTH])
            frame_eligibility = assess_frame_eligibility(
                frame_width=w,
                frame_height=h,
                detection_confidence=driver_detection["confidence"],
                raw_face_box=driver_detection["raw_box"],
                clipped_face_box=clipped_face_box,
                landmarks=landmarks,
                ear=ear,
                mar=mar,
                head_pose=estimate_head_pose(landmarks, w, h),
                policy=FRAME_ELIGIBILITY_POLICY,
            )
            _store_frame_eligibility(frame_eligibility)

            intervention_observation: bool | None = None
            if ear is None or mar is None:
                self.fatigue_state.face_missing()
                data_store["is_drowsy"] = False
                observed_at = self.benchmark.now()
                self.benchmark.observe_raw_signal(False, observed_at)
            else:
                data_store["EAR"] = ear
                data_store["MAR"] = mar
                eye_closure_score = normalize_eye_closure(ear)
                mouth_open_score = normalize_mouth_opening(mar)
                observed_at = self.benchmark.now()
                self.benchmark.observe_raw_signal(
                    ear < EAR_DROWSY_THRESHOLD or mar > MAR_DROWSY_THRESHOLD,
                    observed_at,
                )
                fatigue = self.fatigue_state.update(
                    eye_severity=eye_closure_score,
                    eye_closed=ear < EAR_DROWSY_THRESHOLD,
                    mouth_severity=mouth_open_score,
                    now=observed_at,
                )
                data_store["is_drowsy"] = fatigue.is_drowsy
                intervention_observation = fatigue.is_drowsy
                if fatigue.just_confirmed:
                    self.benchmark.confirm_fatigue(observed_at)

            _store_awake_gate(
                self.awake_state_gate.observe(
                    frame=frame_eligibility,
                    ear=ear,
                    mar=mar,
                    fatigue_suspected=bool(data_store["is_drowsy"]),
                    now=observed_at,
                )
            )
            intervention = self.intervention_controller.update(
                is_drowsy=intervention_observation,
                now=observed_at,
            )

        local_alarm_delivered = _store_intervention(
            intervention,
            self.local_alarm,
        )
        if local_alarm_delivered:
            self.benchmark.intervention_delivered(self.benchmark.now())
        elif intervention.action is InterventionAction.RECOVERED:
            self.benchmark.fatigue_recovered()

        if bool(data_store["is_drowsy"]):
            cv2.putText(
                frame,
                "DROWSY! Wake up!",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                4,
            )
        if landmarks is not None:
            for x, y in landmarks:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        emit_started_at = self.benchmark.now()
        socketio.emit("update_data", dict(data_store))
        non_vision_seconds += self.benchmark.now() - emit_started_at
        processing_finished_at = self.benchmark.now()

        encoded, buffer = cv2.imencode(".jpg", frame)
        if not encoded:
            raise RuntimeError("JPEG encoding failed")
        loop_finished_at = self.benchmark.now()
        self.benchmark.record_frame(
            processing_finished_at - processing_started_at - non_vision_seconds,
            loop_finished_at - loop_started_at,
        )
        return buffer.tobytes()


def _open_camera():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("unable to open camera index 0")
    return capture


def _store_camera_status(running: bool, error: str | None) -> None:
    data_store["camera_running"] = bool(running)
    data_store["camera_error"] = error


monitoring_worker = MonitoringWorker(
    capture_factory=_open_camera,
    processor_factory=CameraPipeline,
    status_callback=_store_camera_status,
)


def start_monitoring() -> None:
    monitoring_worker.start()


def stop_monitoring() -> None:
    monitoring_worker.stop()


def generate_frames() -> Iterator[bytes]:
    """Subscribe an HTTP client to the worker's latest encoded frame."""
    yield from monitoring_worker.stream()
