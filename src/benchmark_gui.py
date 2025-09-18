import cv2
import os
import time
from datetime import datetime
import customtkinter as ctk
from PIL import Image, ImageTk
from threading import Thread
from smile_model_test import SmileDetectionModel
from age_model_test import AgePredictionModel
from deepface import DeepFace
import numpy as np
import subprocess
import platform
camera_index = 0  # start with default camera

# ------------------- Load Models -------------------
smile_model = SmileDetectionModel(model_name="dataset")
age_model = AgePredictionModel()

# Create folder to save captured images
os.makedirs("captured_smiles", exist_ok=True)

# ------------------- GUI Setup -------------------
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Smile, Age & Emotion Detection")
root.geometry("1200x700")

# Video Feed
video_label = ctk.CTkLabel(root, text="")
video_label.grid(row=0, column=0, padx=10, pady=10)

# Right Panel for Predictions + Captured Preview
right_panel = ctk.CTkFrame(root, width=350, height=600)
right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="n")

prediction_label = ctk.CTkLabel(right_panel, text="Predictions", font=("Arial", 18, "bold"))
prediction_label.pack(pady=10)

smile_text = ctk.CTkLabel(right_panel, text="Smile: N/A", font=("Arial", 14))
smile_text.pack(pady=5)

age_text = ctk.CTkLabel(right_panel, text="Age: N/A", font=("Arial", 14))
age_text.pack(pady=5)

emotion_text = ctk.CTkLabel(right_panel, text="Emotion: N/A", font=("Arial", 14))
emotion_text.pack(pady=5)

fps_text = ctk.CTkLabel(right_panel, text="FPS: 0", font=("Arial", 14))
fps_text.pack(pady=5)

proc_text = ctk.CTkLabel(right_panel, text="Proc Time: 0 ms", font=("Arial", 14))
proc_text.pack(pady=5)

status_label = ctk.CTkLabel(right_panel, text="Status: Waiting...", font=("Arial", 14))
status_label.pack(pady=10)

# Captured Image Preview
captured_preview_label = ctk.CTkLabel(right_panel, text="Last Captured Image", font=("Arial", 14, "bold"))
captured_preview_label.pack(pady=10)

captured_img_label = ctk.CTkLabel(right_panel, text="")
captured_img_label.pack(pady=10)

# Buttons
btn_frame = ctk.CTkFrame(right_panel)
btn_frame.pack(pady=20)
def open_gallery():
    folder = os.path.abspath("captured_smiles")
    if platform.system() == "Windows":
        os.startfile(folder)
    elif platform.system() == "Darwin":  # macOS
        subprocess.Popen(["open", folder])
    else:  # Linux
        subprocess.Popen(["xdg-open", folder])

btn_gallery = ctk.CTkButton(btn_frame, text="🖼️ Open Gallery", command=open_gallery, width=150)
btn_gallery.grid(row=0, column=2, padx=10)

def take_photo():
    global manual_capture
    manual_capture = True

def quit_program():
    global quit_app
    quit_app = True
    root.destroy()

btn_photo = ctk.CTkButton(btn_frame, text="📸 Take Photo", command=take_photo, width=150)
btn_photo.grid(row=0, column=0, padx=10)

btn_quit = ctk.CTkButton(btn_frame, text="❌ Quit", command=quit_program, fg_color="red", width=150)
btn_quit.grid(row=0, column=1, padx=10)

# ------------------- Webcam -------------------
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

manual_capture = False
quit_app = False
capture_count = 0
prev_time = time.time()

# ------------------- Helper Function -------------------
def detect_and_update():
    global manual_capture, capture_count, prev_time

    ret, frame = cap.read()
    if not ret:
        root.after(10, detect_and_update)
        return

    start_time = time.time()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    smile_label, age_label, dominant_emotion = "N/A", "N/A", "N/A"
    smile_conf, age_conf = 0.0, 0.0

    for (x, y, w, h) in faces:
        face_img = frame_rgb[y:y+h, x:x+w]

        # Smile detection
        smile_result = smile_model.predict_from_array(face_img)
        smile_label = smile_result.get("label", "N/A")
        smile_conf = smile_result.get("confidence", 0.0)

        # Age prediction
        age_result = age_model.predict_from_array(face_img)
        age_label = str(age_result.get("age", "N/A"))
        age_conf = age_result.get("confidence", 0.0)

        # Emotion detection
        try:
            emotion_result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            if isinstance(emotion_result, list):
                emotion_result = emotion_result[0]
            dominant_emotion = emotion_result.get("dominant_emotion", "N/A")
        except Exception:
            dominant_emotion = "N/A"

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Auto Capture on Smile
        if smile_label.lower() == "smiling":
            capture_count += 1
            filename = f"captured_smiles/auto_{capture_count}.jpg"
            cv2.imwrite(filename, frame)

            # Save metadata
            meta_file = filename.replace(".jpg", ".txt")
            with open(meta_file, "w") as f:
                f.write(f"Timestamp: {datetime.now()}\n")
                f.write(f"Smile: {smile_label} ({smile_conf:.2f})\n")
                f.write(f"Age: {age_label} ({age_conf:.2f})\n")
                f.write(f"Emotion: {dominant_emotion}\n")

            # Update preview
            update_captured_preview(filename)

    # Manual Capture
    if manual_capture:
        capture_count += 1
        filename = f"captured_smiles/manual_{capture_count}.jpg"
        cv2.imwrite(filename, frame)
        update_captured_preview(filename)
        manual_capture = False

    # Calculate FPS and Proc Time
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time
    proc_time = (curr_time - start_time) * 1000

    # Update GUI Labels
    smile_text.configure(text=f"Smile: {smile_label} ({smile_conf:.2f})")
    age_text.configure(text=f"Age: {age_label} ({age_conf:.2f})")
    emotion_text.configure(text=f"Emotion: {dominant_emotion}")
    fps_text.configure(text=f"FPS: {fps:.2f}")
    proc_text.configure(text=f"Proc Time: {proc_time:.1f} ms")
    status_label.configure(text=f"Captured: {capture_count} | Faces: {len(faces)}")

    # Update Video Frame
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    if not quit_app:
        root.after(10, detect_and_update)

def update_captured_preview(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 150))
    imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
    captured_img_label.imgtk = imgtk
    captured_img_label.configure(image=imgtk)

# ------------------- Start Detection -------------------
root.after(0, detect_and_update)
root.mainloop()

# Release resources
cap.release()
cv2.destroyAllWindows()
