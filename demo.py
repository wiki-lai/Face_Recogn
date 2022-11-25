import cv2
import face_recognition

imgWK = face_recognition.load_image_file('base/WUJING.jpg')
imgWK = cv2.cvtColor(imgWK, cv2.COLOR_BGR2RGB)

encodeWK = face_recognition.face_encodings(imgWK)[0]
faceLoc1 = face_recognition.face_locations(imgWK)[0]
cv2.rectangle(imgWK, (faceLoc1[3], faceLoc1[0]), (faceLoc1[1], faceLoc1[2]), (255, 0, 255), 2)


imgWKT = face_recognition.load_image_file('test/1.jpg')
imgWKT = cv2.cvtColor(imgWKT, cv2.COLOR_BGR2RGB)

encodeWKT = face_recognition.face_encodings(imgWKT)[0]
faceLoc2 = face_recognition.face_locations(imgWKT)[0]
cv2.rectangle(imgWKT, (faceLoc2[3], faceLoc2[0]), (faceLoc2[1], faceLoc2[2]), (255, 0, 255), 2)


results = face_recognition.compare_faces([encodeWK], encodeWKT)
faceDis = face_recognition.face_distance([encodeWK], encodeWKT)


cv2.putText(imgWKT, f'{results} {round(1-faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)


print(results, 1-faceDis)
cv2.imshow('WU KING', imgWKT)
cv2.waitKey(0)
