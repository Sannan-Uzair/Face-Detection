from cv2 import cv2
import numpy as np
import face_recognition

img2 = face_recognition.load_image_file('2.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img3 = face_recognition.load_image_file('1.jpg')
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

fca_loc = face_recognition.face_locations(img2)[0]
encode_img = face_recognition.face_encodings(img2)[0]
cv2.rectangle(img2, (fca_loc[3],fca_loc[0]),(fca_loc[1],fca_loc[2]),(255,0,255),2)

fac_loc_test = face_recognition.face_locations(img3)[0]
encode_test = face_recognition.face_encodings(img3)[0]
cv2.rectangle(img3, (fac_loc_test[3],fac_loc_test[0]),(fac_loc_test[1],fac_loc_test[2]),(255,0,255),2)

result = face_recognition.compare_faces([encode_img], encode_test)
face_dist = face_recognition.face_distance([encode_img], encode_test)
print(result,face_dist)

cv2.putText(img2, f'{result} {round(face_dist[0]),2}', (50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

cv2.imshow('img2', img2)
cv2.imshow('img3', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
