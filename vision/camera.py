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
from vision.drowsiness import FatigueState

socketio = SocketIO()
### TODO
# - Calibration: per-user EAR/MAR thresholds instead of the fixed constants below.
# - Glasses detection.
# - Detect a hand covering the mouth, and distinguish that from a hand just resting near it
#   (e.g. while laughing) - ties into emotion detection below.
# - Emotion detection: laughing can close the eyes and read as drowsiness; needs to be told
#   apart from real fatigue.
# - Wire up an actual intervention (voice prompt) when drowsiness is confirmed. Today
#   confirmation only updates state and benchmark data.
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

data_store = {
    "EAR": 0,
    "MAR": 0,
    "is_drowsy": False,
    "ai_response": "",
}


def eye_aspect_ratio(eye: np.ndarray) -> float:
    """calculate Eye Aspect Ratio (EAR)"""
    A = distance.euclidean(eye[1], eye[5])  # vertical
    B = distance.euclidean(eye[2], eye[4])  # vertical
    C = distance.euclidean(eye[0], eye[3])  # horizontal
    ear = (A + B) / (2.0 * C)
    return round(ear, 3)


def mouth_aspect_ratio(mouth: np.ndarray) -> float:
    """calculate Mouth Aspect Ratio (MAR)"""
    A = distance.euclidean(mouth[1], mouth[5])  # vertical
    B = distance.euclidean(mouth[2], mouth[4])  # vertical
    C = distance.euclidean(mouth[0], mouth[3])  # horizontal
    mar = (A + B) / (2.0 * C)
    return round(mar, 3)


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


def generate_frames() -> Iterator[bytes]:
    cap = cv2.VideoCapture(0)
    benchmark = EepyBenchmark(BENCHMARK_ENABLED, BENCHMARK_FRAME_COUNT)
    fatigue_state = FatigueState(
        eye_grace_seconds=EYE_CLOSURE_GRACE_SECONDS,
        eye_ramp_seconds=EYE_CLOSURE_RAMP_SECONDS,
        mouth_window_seconds=MOUTH_EVIDENCE_WINDOW_SECONDS,
        eye_weight=EYE_SCORE_WEIGHT,
        mouth_weight=MOUTH_SCORE_WEIGHT,
    )
    
    try:
        while True:
            # end-to-end timing includes blocking capture and JPEG encoding
            loop_started_at = benchmark.now()
            success, frame = cap.read()
            if not success:
                break

            # processing-only timing excludes camera capture and stream-consumer waits
            processing_started_at = benchmark.now()
            non_vision_seconds = 0.0

            # convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = frame.shape[:2]

            # detecting face
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123), False, False)
            face_net.setInput(blob)
            detections = face_net.forward()

            # Track one driver face per frame. The largest detected face is the
            # clearest available proxy for the closest/primary driver.
            driver_detection = None
            driver_area = 0
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > FACE_DETECTION_CONFIDENCE_THRESHOLD:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")
                    x, y = max(0, x), max(0, y)
                    x1, y1 = min(w - 1, x1), min(h - 1, y1)
                    area = max(0, x1 - x) * max(0, y1 - y)
                    if area > driver_area:
                        driver_area = area
                        driver_detection = (x, y, x1, y1)

            if driver_detection is None:
                fatigue_state.face_missing()
                data_store["is_drowsy"] = False
                benchmark.observe_raw_signal(False, benchmark.now())
            else:
                face_rect = dlib.rectangle(*driver_detection)

                # detecting landmarks facial landmarks
                shape = predictor(gray, face_rect)
                landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

                # EAR calculation for blink detection
                left_eye = landmarks[LEFT_EYE]
                right_eye = landmarks[RIGHT_EYE]
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                ear = round(ear, 3)
                data_store["EAR"] = ear

                # MAR calculation for yawning detection
                mouth = landmarks[MOUTH]
                mar = mouth_aspect_ratio(mouth)
                mar = round(mar, 3)
                data_store["MAR"] = mar

                # normalise differently scaled EAR/MAR values before aggregation
                eye_closure_score = normalize_eye_closure(ear)
                mouth_open_score = normalize_mouth_opening(mar)

                # start an episode at the first raw eye-closure or yawn signal
                observed_at = benchmark.now()
                benchmark.observe_raw_signal(
                    ear < EAR_DROWSY_THRESHOLD or mar > MAR_DROWSY_THRESHOLD,
                    observed_at,
                )
                fatigue = fatigue_state.update(
                    eye_severity=eye_closure_score,
                    eye_closed=ear < EAR_DROWSY_THRESHOLD,
                    mouth_severity=mouth_open_score,
                    now=observed_at,
                )

                if fatigue.is_drowsy:
                    cv2.putText(frame, "DROWSY! Wake up!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

                data_store["is_drowsy"] = fatigue.is_drowsy
                if fatigue.just_confirmed:
                    benchmark.confirm_fatigue(observed_at)

                # sending data to frontend
                data = dict(data_store)
                if fatigue.just_confirmed:
                    # no real intervention exists yet 
                    benchmark.intervention_invoked(benchmark.now())
                emit_started_at = benchmark.now()
                socketio.emit("update_data", data)
                non_vision_seconds += benchmark.now() - emit_started_at

                # drawing landmarks
                for (x, y) in landmarks:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            processing_finished_at = benchmark.now()

            # converting frame to bytes for streaming
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            loop_finished_at = benchmark.now()
            benchmark.record_frame(
                processing_finished_at - processing_started_at - non_vision_seconds,
                loop_finished_at - loop_started_at,
            )
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    # release the capture even if the client disconnects mid-stream
    finally:
        cap.release()
