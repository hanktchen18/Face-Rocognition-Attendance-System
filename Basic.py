import cv2
import numpy as np
import face_recognition
import os

# import images and covert to RGB
imgElon = face_recognition.load_image_file('ImagesBasic/Elon Musk.png')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/Elon test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

# Identify face location and encode, use rectangle to showcase the face area
faceLoc = face_recognition.face_locations(imgElon)[0] # Return a list of tuples (top, right, bottom, left)
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2) # cv2.rectangle(target, (left, top), (right, bottom), color, thickness)

# Same for test images
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

# Compare two images
res = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
bool_res = bool(res)
print(bool_res)
print(faceDis)

# Display the results on the test image
cv2.putText(imgTest, f'{bool_res} {round(faceDis[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2 )

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon test', imgTest)
cv2.waitKey(0)
cv2.destroyAllWindows()
