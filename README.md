# FaceRocognitionAttendanceSystem

A Python-based real-time facial recognition that marks attendance using a webcam.
![face_recognition_video](https://github.com/user-attachments/assets/9fd3d01c-15c6-4728-a54e-9d39842877b5)

## Requirements

- Python 3.7 or higher
- Webcam
- Required Python packages:
  - face_recognition
  - opencv-python
  - numpy
  - pandas

## Installation

1. Clone this repository to your computer:
```bash
git clone https://github.com/hanktchen18/FaceRocognitionAttendanceSystem.git
cd FaceRocognitionAttendanceSystem
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

## How to Use

1. Place face images of people you want to recognize in the `ImagesBasic` folder (preferably clear front-facing photos). There are some example images in the folder. You can either test the system using them or upload your own images!
2. Run the program:
```bash
python main.py
```
3. The system will automatically open your webcam and start facial recognition (get ready to show the image of the person in front of your camera)
4. When a known face is recognized, the system will automatically record attendance in the `Attendance.csv` file, including name and time recorded (if the person already marked as attendance there won't be new record)

## Features

* Real-time facial recognition
* Automatic attendance logging
* Support for multiple face recognition
* Export attendance records

# What I Learned

* Use face_recognition and OpenCV libraries for facial detection and recognition
* Automate image processing and facial encoding to match with live webcam input
* Extract and store user data and file handling for attendance records
