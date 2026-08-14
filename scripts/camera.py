import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from flask_socketio import SocketIO
from scripts.benchmark import (
    BENCHMARK_ENABLED,
    BENCHMARK_FRAME_COUNT,
    EepyBenchmark,
)
from scripts.drowsiness import FatigueState

socketio = SocketIO()
### to do
# CALIBRATION !!!!!!!!!!!!!!!!

# covering mouth when yawning
# consider glasses detection
# ADJUST THRESHOLDS

# covering mouth when yawning
# consider glasses detection

# 0. if eyes are closed for too long, wake up
# 1. drowsiness eye detection is fixed i think -> check with everyones eyes??
# 2. add detection for coverign hand -> if possible, differentiate between covering mouth and just having it there cus of lauhging or smth
# 3. previous point ties in with 2, emotion detection needs to be added - laughing can close eyes and be mistaken for drowsiness

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

# Normalization bounds are centered on the existing binary fatigue thresholds.
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
}

# function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])  # vertical
    B = distance.euclidean(eye[2], eye[4])  # vertical
    C = distance.euclidean(eye[0], eye[3])  # horizontal
    ear = (A + B) / (2.0 * C)
    return round(ear, 3)

# function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[1], mouth[5])  # vertical
    B = distance.euclidean(mouth[2], mouth[4])  # vertical
    C = distance.euclidean(mouth[0], mouth[3])  # horizontal
    mar = (A + B) / (2.0 * C)
    return round(mar, 3)


def clamp(value, lower=0.0, upper=1.0):
    return max(lower, min(value, upper))


def normalize_eye_closure(ear):
    """Map open-to-closed EAR values onto a clamped [0, 1] score."""
    score = (EAR_OPEN_BOUND - ear) / (EAR_OPEN_BOUND - EAR_CLOSED_BOUND)
    return clamp(score)


def normalize_mouth_opening(mar):
    """Map resting-to-yawning MAR values onto a clamped [0, 1] score."""
    score = (mar - MAR_RESTING_BOUND) / (MAR_YAWN_BOUND - MAR_RESTING_BOUND)
    return clamp(score)


blink_count = 0
yawn_count = 0

def generate_frames():
    global blink_count, yawn_count
    cap = cv2.VideoCapture(0)
    benchmark = EepyBenchmark(BENCHMARK_ENABLED, BENCHMARK_FRAME_COUNT)
    fatigue_state = FatigueState(
        eye_grace_seconds=EYE_CLOSURE_GRACE_SECONDS,
        eye_ramp_seconds=EYE_CLOSURE_RAMP_SECONDS,
        mouth_window_seconds=MOUTH_EVIDENCE_WINDOW_SECONDS,
        eye_weight=EYE_SCORE_WEIGHT,
        mouth_weight=MOUTH_SCORE_WEIGHT,
    )
    
    while True:
        # End-to-end timing includes blocking capture and JPEG encoding.
        loop_started_at = benchmark.now()
        success, frame = cap.read()
        if not success:
            break

        # Processing-only timing excludes camera capture and stream-consumer waits.
        processing_started_at = benchmark.now()
        non_vision_seconds = 0.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert frame to grayscale
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
            if confidence > 0.5:
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
            blink_count = 0
            yawn_count = 0
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

                # Normalize differently scaled EAR/MAR values before aggregation.
                eye_closure_score = normalize_eye_closure(ear)
                mouth_open_score = normalize_mouth_opening(mar)

                # Start an episode at the first raw eye-closure or yawn signal.
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
                
                if ear < EAR_DROWSY_THRESHOLD:
                    blink_count += 1
                    if blink_count > 10:      # ADJUST THIS
                        cv2.putText(frame, "DROWSY! Wake up!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                else:
                    blink_count = 0

                # detecting yawning
                if mar > MAR_DROWSY_THRESHOLD:
                    yawn_count += 1
                    if yawn_count > 10:  # ADJUST THIS
                        cv2.putText(frame, "Yawning! Take a break!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)
                else:
                    yawn_count = 0

                data_store["is_drowsy"] = fatigue.is_drowsy
                if fatigue.just_confirmed:
                    benchmark.confirm_fatigue(observed_at)
                
                # sending data to frontend
                data = {"EAR": ear, "MAR": mar, "is_drowsy": data_store["is_drowsy"]}
                if fatigue.just_confirmed:
                    # Invocation, rather than Socket.IO delivery, is the local intervention boundary.
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
