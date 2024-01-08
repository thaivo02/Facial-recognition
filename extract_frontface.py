import numpy as np
import cv2
from matplotlib import pyplot as plt

# folder = "./test_img/"
# img_path = 'selfie1.jpg'
# img = cv2.imread(folder+img_path)

"""Using Haar Cascade to get face of image"""
def get_face(img):
    face_cascade = cv2.CascadeClassifier('./other_files/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img)
    roi = img
    for (x,y,w,h) in faces:
        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi = img[y:y+w, x:x+h]
    return roi
    
# cv2.imshow('img',roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()