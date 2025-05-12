import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Create an automatic workflow that will convert images into encoding form, which will be compared with the webcam
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)

# Append names into ClassNames, as well as for images
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

# Encoding images into encodeLists
def findEncodings(images):
    encodeLists = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeLists.append(encode)
    return encodeLists

encodeListKnown = findEncodings(images)
print('Encoding Complete!')

# Mark attendance if webcam identifies that person, and write the time in Attendance.csv
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [] 
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0]) # Append only names in nameList
        if name not in nameList: # If somebody is arrived, we don't want repeat it
            now = datetime.now()
            dtStr = now.strftime('%m/%d/%Y, %H:%M:%S')
            f.writelines(f'\n{name},{dtStr}') # Write it in a new line with name and time


# Open the webcam
cap = cv2.VideoCapture(0)

# Get each frame one by one, convert it to RGB
while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25) # Reducing size of input helps us in speeding up the process (1/4 of size)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    # find face location in webcam and encode it
    faceCurFrame = face_recognition.face_locations(imgS) # Find all faces in webcam
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    # Iterate through all the faces identified in webcam, and compare them to all of our encoding targets
    # We want them in the same loop so we use zip - find one frame in webcam that match to its location
    for encodeFace,faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        # If find matches, display on webcam
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # Calculate similarity percentage with adjusted formula
            similarity = max(0, min(100, (1 - faceDis[matchIndex]) * 200))
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4 # We have scaled down to 0.25, so all location coordinates times 4 is needed
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(img,f'{name} {similarity:.1f}%',(x1, y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)