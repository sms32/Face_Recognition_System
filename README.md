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

```bash
## PROJECT STRUCTURE

FaceRecognitionAttendance/
├── data/
│   ├── faces_data.pkl              # Saved face encodings
│   ├── names.pkl                   # Saved names
│   ├── log.csv                     # Registration logs
│   ├── haarcascade_frontalface_default.xml  # Face detection model
│   └── previews/                   # Captured preview images
├── Attendance/                     # Daily attendance CSVs
├── main.py                          # Main Streamlit application
├── requirements.txt                # Required packages
├── README.md                       # Project description
```

## SETUP INSTRUCTIONS

 1. Clone the repository
```bash
git clone https://github.com/your-username/face-attendance-app.git
cd face-attendance-app
```
 2. (Optional) Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate
# On Windows: venv\Scripts\activate
```
 3. Install dependencies
```bash
pip install -r requirements.txt
```
 4. Run the Streamlit app
```bash
streamlit run app.py
```

## NOTES

- Uses Haarcascade XML for face detection (OpenCV)
- Face data stored in .pkl format
- Attendance saved per day in CSV files
- Uses KNN classifier with 5 neighbors
- Face hashing prevents duplicates when adding new entries
- App UI is fully local and simple to use

## LICENSE

This project is open-source and free to use for personal, academic, or non-commercial purposes.
