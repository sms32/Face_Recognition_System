# Face Recognition Attendance System

This project is a real-time face recognition attendance system built using Python, OpenCV, and Streamlit. It allows you to register faces, mark attendance, and view logs via a simple local web app.

## FEATURES

- Real-time face detection & recognition using webcam
- Add new faces with duplicate prevention (via MD5 hash)
- Mark attendance with voice confirmation
- Daily attendance saved in CSV format
- Preview image saved during face registration
- View attendance logs and registration history
- Runs fully offline
- Built with Streamlit for UI

## PROJECT STRUCTURE
--------------------

FaceRecognitionAttendance/
├── data/
│   ├── faces_data.pkl              # Saved face encodings
│   ├── names.pkl                   # Saved names
│   ├── log.csv                     # Registration logs
│   ├── haarcascade_frontalface_default.xml  # Face detection model
│   └── previews/                   # Captured preview images
├── Attendance/                     # Daily attendance CSVs
├── app.py                          # Main Streamlit application
├── requirements.txt                # Required packages
├── README.md                       # Project description
