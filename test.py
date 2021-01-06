import cv2 as cv
import matplotlib.pyplot as plt
img=cv.imread('E:\\IMG_2351.jpg')
img=cv.resize(img,(700,500),interpolation=cv.INTER_AREA)
cv.imshow('image',img)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

cv.imshow('gray',gray)

harr_cascade=cv.CascadeClassifier('haarcase_face_detect.xml')
face_rect=harr_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
print('number of faces',str(len(face_rect)))

for (x,y,w,h) in face_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('detected',img)

cv.waitKey(0)
