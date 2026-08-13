import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from flask_socketio import SocketIO
import time
from collections import deque  # leetcode finally being useful!!!
from scripts.benchmark import (
    BENCHMARK_ENABLED,
    BENCHMARK_FRAME_COUNT,
    EepyBenchmark,
)

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

# indices for eye and mouth landmarks

# 1-indexed
# LEFT_EYE = [37, 38, 39, 40, 41, 42]
# RIGHT_EYE = [43, 44, 45, 46, 47, 48]
# MOUTH = [50, 52, 54, 56, 58, 59]
# 0-indexed
LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
MOUTH = [48, 50, 52, 54, 56, 58]

# ear_value = 0
# mar_value = 0
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

blink_count = 0
yawn_count = 0
drowsy_frame_count = 0  # counter to track the number of frames since is_drowsy was set to True

def generate_frames():
    global blink_count, yawn_count, drowsy_frame_count
    cap = cv2.VideoCapture(0)
    benchmark = EepyBenchmark(BENCHMARK_ENABLED, BENCHMARK_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_window = int(fps * 5)  # Number of frames in the last minute
    blink_scores = deque(maxlen=frame_window)
    yawn_scores = deque(maxlen=frame_window)
    
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

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                face_rect = dlib.rectangle(x, y, x1, y1)

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

                # detecting drowsiness
                if ear < 0.3:  # threshold
                    blink_scores.append(1)
                else:
                    blink_scores.append(0)
                    
                if mar > 0.6:  # threshold
                    yawn_scores.append(1)
                else:
                    yawn_scores.append(0)

                # Start an episode at the first raw eye-closure or yawn signal.
                benchmark.observe_raw_signal(
                    ear < 0.3 or mar > 0.6, benchmark.now()
                )
                    
                if len(blink_scores) >= frame_window:
                    blink_scores.popleft()
                if len(yawn_scores) >= frame_window:
                    yawn_scores.popleft()
                    
                blink_score = sum(blink_scores)
                yawn_score = sum(yawn_scores)
                drowsiness_score = blink_score * 0.7 + yawn_score * 0.3
                # print(f'len(blink_scores) {len(blink_scores)}')
                # print(f'len(yawn_scores) {len(yawn_scores)}')
                # print(f'Blink Score: {blink_score}')
                # print(f'Yawn Score: {yawn_score}')
                
                if ear < 0.3:  # threshold
                    blink_count += 1
                    if blink_count > 10:      # ADJUST THIS
                        cv2.putText(frame, "DROWSY! Wake up!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                else:
                    blink_count = 0

                # detecting yawning
                if mar > 0.6:  # threshold
                    yawn_count += 1
                    if yawn_count > 10:  # ADJUST THIS
                        cv2.putText(frame, "Yawning! Take a break!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)
                else:
                    yawn_count = 0

                # display warnings based on the scores
                # if blink_score > frame_window * 0.5:  # Adjust threshold as needed
                #     cv2.putText(frame, "TOO MUCH BLINKING", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
                # if yawn_score > frame_window * 0.3:  # Adjust threshold as needed
                #     cv2.putText(frame, "TOO MUCH YAWNING", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 4)
                
                fatigue_just_confirmed = False
                if drowsiness_score > frame_window * 0.5:  # ADJUST
                    # cv2.putText(frame, "DROWSINESS", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
                    fatigue_just_confirmed = not data_store["is_drowsy"]
                    if fatigue_just_confirmed:
                        benchmark.confirm_fatigue(benchmark.now())
                    data_store["is_drowsy"] = True
                    drowsy_frame_count += 1
                else:
                    if drowsy_frame_count > 140:  # ADJUSTABLE
                        is_drowsy = False
                        drowsy_frame_count = 0

                # print(f'Drowsiness Score: {drowsiness_score}')
                # print(f'is_drowsy: {data_store["is_drowsy"]}')
                
                # sending data to frontend
                data = {"EAR": ear, "MAR": mar, "is_drowsy": data_store["is_drowsy"]}
                if fatigue_just_confirmed:
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

cv2.destroyAllWindows()
