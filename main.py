import cv2
import torch
import mediapipe as mp
import numpy as np
import pyaudio
import threading
from ultralytics import YOLO

# Load YOLO model
yolo_model = YOLO('yolov8n.pt')

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Voice detection parameters
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
VOLUME_THRESHOLD = 0.01  # Lowered to detect soft speech
cheating_alert = False  # Global flag to track cheating
lock = threading.Lock()

# Initialize PyAudio
audio = pyaudio.PyAudio()

def audio_callback(in_data, frame_count, time_info, status):
    """Callback function for real-time audio processing."""
    global cheating_alert
    if status:
        print("PyAudio Status:", status)
    
    # Convert raw audio bytes to numpy array
    audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
    volume_norm = np.linalg.norm(audio_data) * 10  # Normalize volume

    if volume_norm > VOLUME_THRESHOLD:
        with lock:
            cheating_alert = True

    return (in_data, pyaudio.paContinue)

# Start PyAudio stream
try:
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
        stream_callback=audio_callback
    )
    stream.start_stream()
except Exception as e:
    print("⚠️ Microphone error:", e)

# Fast Video Capture Class
class FastVideoCapture:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Avoids lag
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        """Continuously updates the latest frame."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        """Returns the latest captured frame."""
        with self.lock:
            return self.frame

    def release(self):
        """Releases the video capture."""
        self.running = False
        self.thread.join()
        self.cap.release()

# Start ultra-fast video capture
cap = FastVideoCapture()

while True:
    frame = cap.read()
    if frame is None:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face detection
    results = face_detection.process(rgb_frame)
    faces_detected = 0

    if results.detections:
        faces_detected = len(results.detections)
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = (
                int(bboxC.xmin * w), int(bboxC.ymin * h),
                int(bboxC.width * w), int(bboxC.height * h)
            )
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Object detection using YOLO
    yolo_results = yolo_model(frame, verbose=False)
    cheating_detected = False  # Local flag for current frame

    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            
            if class_id in result.names:
                label = result.names[class_id]
                if label in ['cell phone', 'book'] or (label == 'person' and faces_detected > 1):
                    cheating_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, 'ALERT: ' + label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Update cheating alert only when cheating is detected
    with lock:
        if cheating_detected:
            cheating_alert = True
        else:
            cheating_alert = False  # Reset when no cheating is detected

    # Display alert only if cheating is detected
    if cheating_alert:
        cv2.putText(frame, '🚨 CHEATING DETECTED!', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    # Show video
    cv2.imshow('Exam Proctoring', frame)
    
    # Quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
stream.stop_stream()
stream.close()
audio.terminate()
cv2.destroyAllWindows()
