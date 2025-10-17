import cv2
import os
import time
import csv
from datetime import datetime
import tkinter as tk
#import mediapipe as mp   # for face alignment
from PIL import Image, ImageTk
from threading import Thread
from smile_model_test import SmileDetectionModel
from age_model_test import AgePredictionModel
from emotion_model_test import EmotionDetectionModel
from deepface import DeepFace
import numpy as np
from collections import deque
prev_time = time.time()
fps = 0
from collections import OrderedDict
class CentroidTracker:
    def __init__(self, max_disappeared=30):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cX = int(x + w / 2.0)
            cY = int(y + h / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.linalg.norm(np.array(object_centroids)[:, None] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects

# Buffers for smoothing (store last 5 predictions)
smile_history = deque(maxlen=5)
age_history = deque(maxlen=5)
emotion_history = deque(maxlen=5)


# ------------------- Load Models -------------------
smile_model = SmileDetectionModel(model_name="dataset")
age_model = AgePredictionModel()
emotion_model = EmotionDetectionModel(model_name="dataset")
# Create folder to save captured images
os.makedirs("captured_smiles", exist_ok=True)

# Initialize CSV logging
log_file = "captured_smiles/results.csv"
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "FileName", "Smile", "Smile_Conf", "Age", "Age_Conf", "Emotion"])
# ------------------- GUI Variables -------------------
manual_capture = False
quit_app = False
capture_count = 0

smile_prev_faces = {}  # track smile per face to avoid multiple captures

# ------------------- Tkinter GUI -------------------
root = tk.Tk()
root.title("Smile, Age & Emotion Detection")

# Canvas for video feed
video_label = tk.Label(root)
video_label.pack()

# Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=5)

def take_photo():
    global manual_capture
    manual_capture = True

def quit_program():
    global quit_app
    quit_app = True
    root.destroy()

btn_photo = tk.Button(btn_frame, text="📸 Take Photo", command=take_photo, width=20, height=2)
btn_photo.pack(side="left", padx=10)

btn_quit = tk.Button(btn_frame, text="❌ Quit", command=quit_program, width=20, height=2, fg="red")
btn_quit.pack(side="left", padx=10)

# Status label
status_label = tk.Label(root, text="Status: Waiting...", font=("Arial", 12))
status_label.pack(pady=5)

# ------------------- Webcam -------------------
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
tracker = CentroidTracker()
# ------------------- Gender Prediction -------------------
# def predict_gender(face_img):
#     try:
#         result = DeepFace.analyze(face_img, actions=['gender'], enforce_detection=False)
#         if isinstance(result, list):
#             result = result[0]
#         return result.get("dominant_gender", "N/A")
#     except:
#         return "N/A"
# ------------------- Helper Functions -------------------
def detect_and_update():
    global manual_capture, capture_count, smile_prev_faces, prev_time
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        root.after(10, detect_and_update)
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    objects = tracker.update(faces)

    current_faces_smile = {}
    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    for i, (x, y, w, h) in enumerate(faces):
        face_img = frame_rgb[y:y+h, x:x+w]
        cv2.putText(frame, f"ID {i}", (x, y - 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

       # --- Smile detection (using model) ---
        smile_result = smile_model.predict_from_array(face_img)
        smile_label = smile_result.get("label", "N/A")
        smile_conf = smile_result.get("confidence", 0.0)
        smile_history.append(smile_label)
        smile_label = max(set(smile_history), key=smile_history.count)  # smoothing

        # --- Age prediction ---
        age_result = age_model.predict_from_array(face_img)
        age_label = str(age_result.get("age", "N/A"))
        # Gender prediction
        #gender_label = predict_gender(face_img)
        age_history.append(age_label)
        age_conf = float(age_result.get("confidence", 0.0))   # <-- make sure this exists
        # Use last seen age most often
        age_label = max(set(age_history), key=age_history.count)

        # Emotion prediction using HuggingFace
        emotion_result = emotion_model.predict_from_array(face_img)
        dominant_emotion = emotion_result.get("label", "N/A")
        emotion_conf = emotion_result.get("confidence", 0.0)

        # Confidence threshold
        if emotion_conf < 0.5:
            dominant_emotion = "Uncertain"

        # Add to smoothing buffer
        emotion_history.append(dominant_emotion)
        dominant_emotion = max(set(emotion_history), key=emotion_history.count)

        # Apply threshold for emotion confidence
        if emotion_conf < 0.3:
            dominant_emotion = "Uncertain"

        emotion_history.append(dominant_emotion)
        dominant_emotion = max(set(emotion_history), key=emotion_history.count)

        # Overlay text without rectangle background
        cv2.putText(frame, f"Smile: {smile_label}", (x, y - 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Age: {age_label}", (x, y - 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # cv2.putText(frame, f"Gender: {gender_label}", (x, y - 20),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Auto capture if smiling
        if smile_label.lower() == "smiling":
            capture_count += 1
            filename = f"captured_smiles/auto_{capture_count}.jpg"
            cv2.imwrite(filename, frame)

            # Save metadata .txt
            meta_file = filename.replace(".jpg", ".txt")
            with open(meta_file, "w") as f:
                f.write(f"Timestamp: {datetime.now()}\n")
                f.write(f"Smile: {smile_label} ({smile_conf:.2f})\n")
                f.write(f"Age: {age_label} ({age_conf:.2f})\n")
                f.write(f"Emotion: {dominant_emotion}\n")

            # Append to CSV log (ensure `log_file` and `csv` imported)
            with open(log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now(), filename,
                    smile_label, f"{smile_conf:.2f}",
                    age_label, f"{age_conf:.2f}",
                    dominant_emotion
                ])

        current_faces_smile[i] = smile_label.lower() == "smiling"

    smile_prev_faces = current_faces_smile

    # Manual capture
    if manual_capture:
        capture_count += 1
        filename = f"captured_smiles/manual_{capture_count}.jpg"
        cv2.imwrite(filename, frame)
        meta_file = filename.replace(".jpg", ".txt")
        with open(meta_file, "w") as f:
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Captured Faces: {len(faces)}\n")
        with open(log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), filename, smile_label, f"{smile_conf:.2f}", age_label, f"{age_conf:.2f}", dominant_emotion])

        manual_capture = False
    # FPS and processing time
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    processing_time = (curr_time - start_time) * 1000
    prev_time = curr_time

    (h, w) = frame.shape[:2]
    cv2.putText(frame, f"FPS: {fps:.2f}", (w-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(frame, f"Proc Time: {processing_time:.1f} ms", (w-200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Convert to ImageTk for Tkinter
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Update status
    status_label.config(text=f"Captured Images: {capture_count} | Faces detected: {len(faces)}")

    if not quit_app:
        root.after(10, detect_and_update)

# ------------------- Start Detection -------------------
root.after(0, detect_and_update)
root.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()