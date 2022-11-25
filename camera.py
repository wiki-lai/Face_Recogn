import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# from PIL import ImageGrab

path = 'base'
pics = os.listdir(path)
print(pics)
images = []
classNames = []

for people in pics:
    curImg = cv2.imread(f'{path}/{people}')
    images.append(curImg)
    classNames.append(os.path.splitext(people)[0])
    print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()

    nameList = []
    for line in myDataList:
        entry = line.split(',')
        nameList.append(entry[0])

    if name not in nameList:
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        f.writelines(f'n{name},{dtString}')


#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    # img = captureScreen()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces_loc = face_recognition.face_locations(frame)
    encodes = face_recognition.face_encodings(frame, faces_loc)

    for faceEncode, faceLoc in zip(encodes, faces_loc):
        matches = face_recognition.compare_faces(encodeListKnown, faceEncode)
        faceDis = face_recognition.face_distance(encodeListKnown, faceEncode)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
        # print(name)
        y1, x2, y2, x1 = faceLoc
        # y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1 + 20), (x2, y1), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name + ' ' + str(round(1-faceDis[matchIndex], 3)), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # markAttendance(name)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
