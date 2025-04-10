import streamlit as st
import os
import cv2
import pickle
import numpy as np
import hashlib
import csv
import time
import pyttsx3
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# === Config ===
IMG_SIZE = 50
MAX_SAMPLES = 100
DATA_DIR = "data"
PREVIEW_DIR = os.path.join(DATA_DIR, "previews")
ATTENDANCE_DIR = "Attendance"
LOG_PATH = os.path.join(DATA_DIR, 'log.csv')
FACES_PATH = os.path.join(DATA_DIR, 'faces_data.pkl')
NAMES_PATH = os.path.join(DATA_DIR, 'names.pkl')
CASCADE_PATH = os.path.join(DATA_DIR, 'haarcascade_frontalface_default.xml')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# === TTS ===
def speak(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    except RuntimeError as e:
        st.warning("⚠️ Voice engine error. Try restarting the app.")
        print(f"TTS Error: {e}")

# === Streamlit UI ===
st.set_page_config(page_title="Face Recognition Attendance", layout="centered")
st.title("📸 Face Recognition Attendance System")

menu = ["🏷️ Add Face", "🧑‍💼 Mark Attendance", "📊 View Attendance Log"]
choice = st.sidebar.selectbox("Navigation", menu)

# === Add Face ===
if choice == "🏷️ Add Face":
    st.header("➕ Add New Face")
    name = st.text_input("Enter Your Name:").strip().lower()
    start = st.button("Start Capture")

    if start and name:
        facedetect = cv2.CascadeClassifier(CASCADE_PATH)
        video = cv2.VideoCapture(0)
        faces_data, session_hashes, i = [], set(), 0

        # Load existing
        faces = np.empty((0, IMG_SIZE * IMG_SIZE * 3))
        names = []
        existing_hashes = set()
        if os.path.exists(FACES_PATH):
            with open(FACES_PATH, 'rb') as f:
                faces = pickle.load(f)
                for face in faces:
                    h = hashlib.md5(face.tobytes()).hexdigest()
                    existing_hashes.add(h)

        if os.path.exists(NAMES_PATH):
            with open(NAMES_PATH, 'rb') as f:
                names = pickle.load(f)

        def get_hash(img):
            return hashlib.md5(img.tobytes()).hexdigest()

        saved_preview = False
        stframe = st.empty()
        while len(faces_data) < MAX_SAMPLES:
            ret, frame = video.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_rects = facedetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces_rects:
                crop = frame[y:y+h, x:x+w, :]
                resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                img_hash = get_hash(resized)
                if img_hash not in session_hashes and img_hash not in existing_hashes:
                    faces_data.append(resized)
                    session_hashes.add(img_hash)
                    if not saved_preview:
                        cv2.imwrite(os.path.join(PREVIEW_DIR, f"preview_{name}.jpg"), resized)
                        saved_preview = True
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                break
            cv2.putText(frame, f"{len(faces_data)}/{MAX_SAMPLES}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            stframe.image(frame, channels="BGR")
        video.release()

        if not faces_data:
            st.error("No unique face data captured!")
        else:
            faces_data = np.asarray(faces_data).reshape(len(faces_data), -1)
            if faces.shape[0] + len(faces_data) != len(names) + len(faces_data):
                st.error("Mismatch between faces and labels count!")
            else:
                faces = np.concatenate([faces, faces_data], axis=0)
                names += [name] * len(faces_data)
                with open(FACES_PATH, 'wb') as f:
                    pickle.dump(faces, f)
                with open(NAMES_PATH, 'wb') as f:
                    pickle.dump(names, f)
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(LOG_PATH, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if os.stat(LOG_PATH).st_size == 0:
                        writer.writerow(['Name', 'Samples Added', 'DateTime'])
                    writer.writerow([name, len(faces_data), ts])
                st.success(f"{len(faces_data)} faces saved for '{name}'")

# === Mark Attendance ===
elif choice == "🧑‍💼 Mark Attendance":
    st.header("✅ Mark Attendance")
    if st.button("Start Face Recognition"):
        with open(NAMES_PATH, 'rb') as f:
            LABELS = pickle.load(f)
        with open(FACES_PATH, 'rb') as f:
            FACES = pickle.load(f)

        if FACES.shape[0] != len(LABELS):
            st.error("Model training error: Faces and Labels mismatch!")
            raise ValueError("Faces and Labels count mismatch.")

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(FACES, LABELS)

        facedetect = cv2.CascadeClassifier(CASCADE_PATH)
        video = cv2.VideoCapture(0)
        stframe = st.empty()
        start_time = time.time()
        recognized = False

        while True:
            ret, frame = video.read()
            if not ret:
                break
            stframe.image(frame, channels="BGR")
            if time.time() - start_time < 2:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                crop = frame[y:y+h, x:x+w]
                resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE)).flatten().reshape(1, -1)
                name = knn.predict(resized)[0]
                timestamp = datetime.now().strftime("%H:%M:%S")
                date = datetime.now().strftime("%d-%m-%Y")
                att_file = os.path.join(ATTENDANCE_DIR, f"Attendance_{date}.csv")
                if not os.path.exists(att_file):
                    with open(att_file, 'w', newline='') as f:
                        csv.writer(f).writerow(['NAME', 'TIME'])
                already_marked = False
                with open(att_file, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        if row and row[0] == name:
                            already_marked = True
                            break
                if already_marked:
                    speak(f"{name}, you are already marked present.")
                    st.warning(f"{name} is already marked present.")
                else:
                    with open(att_file, 'a', newline='') as f:
                        csv.writer(f).writerow([name, timestamp])
                    speak(f"Hello {name}, your attendance has been marked.")
                    st.success(f"Attendance marked for {name}")
                break
        video.release()

# === View Attendance ===
elif choice == "📊 View Attendance Log":
    st.header("📅 Attendance Records")
    date_str = datetime.now().strftime("%d-%m-%Y")
    att_file = os.path.join(ATTENDANCE_DIR, f"Attendance_{date_str}.csv")
    if os.path.exists(att_file):
        df = pd.read_csv(att_file)
        st.success(f"Attendance for {date_str}")
        st.dataframe(df)
    else:
        st.warning("No attendance marked for today yet.")

    st.subheader("📜 Add Face Log")
    if os.path.exists(LOG_PATH):
        log_df = pd.read_csv(LOG_PATH)
        st.dataframe(log_df)
    else:
        st.warning("No faces added yet.")
