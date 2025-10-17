import cv2
import os
import time
from datetime import datetime
import customtkinter as ctk
from PIL import Image, ImageTk
from smile_model_test import SmileDetectionModel
from age_model_test import AgePredictionModel
from deepface import DeepFace
import numpy as np
import csv
import subprocess

# ------------------- Load Models -------------------
smile_model = SmileDetectionModel(model_name="dataset")
age_model = AgePredictionModel()

# Create folder to save captured images
os.makedirs("captured_smiles", exist_ok=True)

# ------------------- CSV Setup -------------------
csv_filename = "benchmarks.csv"
def setup_csv():
    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Filename", "Smile_Confidence", "Age_Confidence", "Emotion", "FPS", "Processing_Time_ms", "Faces_Detected", "Capture_Type"])

def log_benchmarks(filename, smile_conf, age_conf, dominant_emotion, fps, proc_time, faces_detected, capture_type):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, os.path.basename(filename), smile_conf, age_conf, dominant_emotion, fps, proc_time, faces_detected, capture_type])

def open_benchmarks_file():
    try:
        if os.path.exists(csv_filename):
            subprocess.Popen(['notepad.exe', csv_filename])
        else:
            print(f"File not found: {csv_filename}")
    except FileNotFoundError:
        print("Could not open the file. Please check your system's default text editor.")

# ------------------- GUI Setup -------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Smart Selfie Capture")
root.state("zoomed")
root.configure(fg_color="#36454F")

# ------------------- Menu Bar (TOP) -------------------
menubar = ctk.CTkFrame(root, height=30, fg_color="#202020", corner_radius=0)
menubar.pack(side="top", fill="x")

menu_font = ("Arial", 14)

btn_menu_gallery = ctk.CTkButton(menubar, text="🖼️ Gallery", command=lambda: os.startfile("captured_smiles"), font=menu_font, fg_color="transparent", hover_color="#303030")
btn_menu_gallery.pack(side="left", padx=5)

btn_menu_benchmarks = ctk.CTkButton(menubar, text="📊 Benchmarks", command=open_benchmarks_file, font=menu_font, fg_color="transparent", hover_color="#303030")
btn_menu_benchmarks.pack(side="left", padx=5)

btn_menu_home = ctk.CTkButton(menubar, text="🏠 Home", command=lambda: go_home(), font=menu_font, fg_color="transparent", hover_color="#303030")
btn_menu_home.pack(side="left", padx=5)

# ------------------- Main App Frames -------------------
app_frame = ctk.CTkFrame(root, fg_color="#36454F")
app_frame.pack_forget()

# Video feed (left half)
video_frame = ctk.CTkFrame(app_frame, fg_color="#36454F",width=690,height=500)
#video_frame.pack_propagate(False)
video_label = ctk.CTkLabel(video_frame, text="", fg_color="black")
video_label.pack(expand=True, fill="both")

# Last captured preview (right half)
preview_frame = ctk.CTkFrame(app_frame, fg_color="#36454F",width=460,height=460)
#preview_frame.pack_propagate(False)
preview_heading = ctk.CTkLabel(preview_frame, text="Last Captured", font=("Arial", 18, "bold"))
preview_heading.pack(pady=10)
captured_img_label = ctk.CTkLabel(preview_frame, text="(No Image Yet)", fg_color="gray", text_color="white")
captured_img_label.pack(expand=True, fill="both", padx=10, pady=10)

# Separator Line
separator_line = ctk.CTkFrame(root, height=2, fg_color="white")
separator_line.pack_forget()

# Bottom Layout
bottom_frame = ctk.CTkFrame(root, fg_color="#36454F")
# bottom_frame.pack_propagate(False)
bottom_frame.pack_forget()

# Left: predictions (Smile, Age, Emotion)
pred_frame_left = ctk.CTkFrame(bottom_frame, fg_color="#36454F")
pred_frame_left.pack(side="left", anchor="nw", padx=20, pady=10)  # add pady for spacing

pred_heading = ctk.CTkLabel(pred_frame_left, text="Predictions", font=("Arial", 16, "bold"))
pred_heading.pack(anchor="w", pady=(0, 5))
smile_text = ctk.CTkLabel(pred_frame_left, text="Smile: N/A", font=("Arial", 14), text_color="white")
smile_text.pack(anchor="w")
age_text = ctk.CTkLabel(pred_frame_left, text="Age: N/A", font=("Arial", 14), text_color="white")
age_text.pack(anchor="w")
emotion_text = ctk.CTkLabel(pred_frame_left, text="Emotion: N/A", font=("Arial", 14), text_color="white")
emotion_text.pack(anchor="w")

# Right: FPS, Proc Time, Faces
pred_frame_right = ctk.CTkFrame(bottom_frame, fg_color="#36454F")

benchmark_heading = ctk.CTkLabel(pred_frame_right, text="Benchmarks", font=("Arial", 16, "bold"))
benchmark_heading.pack(anchor="e", pady=(0, 5))
fps_text = ctk.CTkLabel(pred_frame_right, text="FPS: 0", font=("Arial", 14), text_color="white")
fps_text.pack(anchor="e")
proc_text = ctk.CTkLabel(pred_frame_right, text="Proc Time: 0 ms", font=("Arial", 14), text_color="white")
proc_text.pack(anchor="e")
faces_text = ctk.CTkLabel(pred_frame_right, text="Faces Detected: 0", font=("Arial", 14), text_color="white")
faces_text.pack(anchor="e")
captured_count_text = ctk.CTkLabel(pred_frame_right, text="Captured Images: 0", font=("Arial", 14), text_color="white")
captured_count_text.pack(anchor="e")


# ------------------- Button Frame (Unified) -------------------
btn_frame = ctk.CTkFrame(video_frame, fg_color="#36454F")
btn_frame.pack(side="bottom", pady=10)

# ------------------- Home Page -------------------
home_frame = ctk.CTkFrame(root, fg_color="#36454F")
home_frame.pack(fill="both", expand=True)

# ---------------- Home Page ----------------
# Container for heading + description
text_container = ctk.CTkFrame(home_frame, fg_color="transparent")
text_container.pack(expand=True)  # This will center everything inside

# Heading (centered)
home_label = ctk.CTkLabel(
    text_container,
    text="Welcome to Smart Selfie Capture",
    font=("Arial", 48, "bold")
)
home_label.pack(pady=(10, 10))

# Description (will stay just below the heading, centered)
description_label = ctk.CTkLabel(
    text_container,
    text="An AI-powered app that captures your best moments with a smile.\n" 
    "This application uses AI to detect smiles, predict age, and recognize emotions in real time.\n"
    "Designed to be fast, interactive, and user-friendly for a smart photo experience.",
    font=("Arial", 18)
)
description_label.pack(pady=(0, 20))


btn_start = ctk.CTkButton(home_frame, text="🚀 Start", command=lambda: start_app(), width=200, height=50, font=("Arial", 20, "bold"))
btn_start.pack(pady=20)

# ------------------- Global Variables -------------------
manual_capture = False
quit_app = False
camera_on = True
capture_count = 0
prev_time = time.time()
preview_mode = False
current_captured_frame = None

# ------------------- Core Functions -------------------
def start_app():
    global capture_count, prev_time
    home_frame.pack_forget()
    app_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)
    separator_line.pack(fill="x", padx=10)
    bottom_frame.pack(side="bottom", fill="x", padx=10, pady=10)
    video_frame.pack(side="left", expand=True, fill="both", padx=5, pady=5)
    preview_frame.pack(side="left", expand=True, fill="both", padx=5, pady=5)
    pred_frame_left.pack(side="left", anchor="w", padx=20)
    pred_frame_right.pack(side="right", anchor="e", padx=20)
    btn_frame.pack(side="bottom", pady=10)
    
    # Show main buttons
    btn_photo.pack(side="left", padx=10)
    btn_switch.pack(side="left", padx=10)
    # btn_gallery.pack(side="left", padx=10)
    # btn_benchmarks.pack(side="left", padx=10)
    # btn_home.pack(side="left", padx=10)
    btn_quit.pack(side="left", padx=10)
    
    # Hide preview buttons
    btn_save.pack_forget()
    btn_discard.pack_forget()
    
    capture_count = 0
    prev_time = time.time()
    
    detect_and_update()

def go_home():
    global preview_mode, current_captured_frame, camera_on
    
    preview_mode = False
    current_captured_frame = None
    camera_on = True
    
    app_frame.pack_forget()
    separator_line.pack_forget()
    bottom_frame.pack_forget()
    
    home_frame.pack(fill="both", expand=True)

def save_photo():
    global preview_mode, current_captured_frame, capture_count
    
    if current_captured_frame is not None:
        capture_count += 1
        filename = f"captured_smiles/manual_{capture_count}.jpg"
        cv2.imwrite(filename, current_captured_frame)
        update_captured_preview(filename)
        
        log_benchmarks(filename, 0, 0, "N/A", 0, 0, 0, "manual")
        
    preview_mode = False
    current_captured_frame = None
    show_main_buttons()

def discard_photo():
    global preview_mode, current_captured_frame
    
    preview_mode = False
    current_captured_frame = None
    show_main_buttons()

def show_main_buttons():
    btn_photo.pack(side="left", padx=10)
    btn_switch.pack(side="left", padx=10)
    # btn_gallery.pack(side="left", padx=10)
    # btn_benchmarks.pack(side="left", padx=10)
    # btn_home.pack(side="left", padx=10)
    btn_quit.pack(side="left", padx=10)
    btn_save.pack_forget()
    btn_discard.pack_forget()
    detect_and_update()

def show_preview_buttons():
    btn_photo.pack_forget()
    btn_switch.pack_forget()
    # btn_gallery.pack_forget()
    # btn_benchmarks.pack_forget()
    # btn_home.pack_forget()
    btn_quit.pack_forget()
    btn_save.pack(side="left", padx=10)
    btn_discard.pack(side="left", padx=10)

def take_photo():
    global preview_mode, current_captured_frame
    
    ret, frame = cap.read()
    if ret:
        current_captured_frame = frame
        preview_mode = True
        
        show_preview_buttons()

        frame_rgb = cv2.cvtColor(current_captured_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

def quit_program():
    global quit_app
    quit_app = True
    root.destroy()

def toggle_camera():
    global camera_on
    camera_on = not camera_on
    if not camera_on:
        video_label.configure(image="", text="📷 Camera OFF", font=("Arial", 18, "bold"))
    else:
        video_label.configure(text="")

# ------------------- Main App Buttons -------------------
btn_photo = ctk.CTkButton(btn_frame, text="📸 Take Photo", command=take_photo, width=150)
btn_switch = ctk.CTkButton(btn_frame, text="🔄 Turn Camera", command=toggle_camera, width=150)
# btn_gallery = ctk.CTkButton(btn_frame, text="🖼️ Gallery", command=lambda: os.startfile("captured_smiles"), width=150)
# btn_benchmarks = ctk.CTkButton(btn_frame, text="📊 Benchmarks", command=open_benchmarks_file, width=150)
# btn_home = ctk.CTkButton(btn_frame, text="🏠 Home", command=go_home, width=150)
btn_quit = ctk.CTkButton(btn_frame, text="❌ Quit", command=quit_program, width=150, fg_color="red")

# ------------------- Preview Buttons -------------------
btn_save = ctk.CTkButton(btn_frame, text="✅ Save", command=save_photo, width=150)
btn_discard = ctk.CTkButton(btn_frame, text="🗑️ Discard", command=discard_photo, width=150, fg_color="red")

# ------------------- Webcam & Main Loop -------------------
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def update_captured_preview(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (captured_img_label.winfo_width(), captured_img_label.winfo_height()))
    imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))
    captured_img_label.imgtk = imgtk
    captured_img_label.configure(image=imgtk, text="")

def detect_and_update():
    global manual_capture, capture_count, prev_time, preview_mode

    if not camera_on or preview_mode:
        root.after(10, detect_and_update)
        return

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
    faces_detected = len(faces)
    if faces_detected == 0:
        cv2.putText(frame, "No face detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    
    log_smile_conf, log_age_conf, log_dominant_emotion = 0.0, 0.0, "N/A"

    for (x, y, w, h) in faces:
        face_img = frame_rgb[y:y+h, x:x+w]
        
        smile_result = smile_model.predict_from_array(face_img)
        smile_label = smile_result.get("label", "N/A")
        smile_conf = smile_result.get("confidence", 0.0)
        log_smile_conf = smile_conf

        age_result = age_model.predict_from_array(face_img)
        age_label = str(age_result.get("age", "N/A"))
        age_conf = age_result.get("confidence", 0.0)
        log_age_conf = age_conf

        try:
            emotion_result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            if isinstance(emotion_result, list):
                emotion_result = emotion_result[0]
            dominant_emotion = emotion_result.get("dominant_emotion", "N/A")
            log_dominant_emotion = dominant_emotion
        except Exception:
            dominant_emotion = "N/A"
            log_dominant_emotion = "N/A"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if smile_label.lower() == "smiling":
            capture_count += 1
            filename = f"captured_smiles/auto_{capture_count}.jpg"
            cv2.imwrite(filename, frame)
            update_captured_preview(filename)
            curr_time_log = time.time()
            fps_log = 1 / (curr_time_log - prev_time) if curr_time_log != prev_time else 0
            proc_time_log = (curr_time_log - start_time) * 1000
            log_benchmarks(filename, log_smile_conf, log_age_conf, log_dominant_emotion, fps_log, proc_time_log, faces_detected, "auto")

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time
    proc_time = (curr_time - start_time) * 1000

    smile_text.configure(text=f"Smile: {smile_label} ({smile_conf:.2f})")
    age_text.configure(text=f"Age: {age_label} ({age_conf:.2f})")
    emotion_text.configure(text=f"Emotion: {dominant_emotion}")
    fps_text.configure(text=f"FPS: {fps:.2f}")
    proc_text.configure(text=f"Proc Time: {proc_time:.1f} ms")
    faces_text.configure(text=f"Faces Detected: {faces_detected}")
    captured_count_text.configure(text=f"Captured Images: {capture_count}")
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    if not quit_app:
        root.after(10, detect_and_update)

# ------------------- Start App -------------------
setup_csv()
root.mainloop()

cap.release()
cv2.destroyAllWindows()