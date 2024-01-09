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
    
    #if no faces are detected then return original img
    if (len(faces) == 0):
        return []

    list_img = []
    for (x,y,w,h) in faces:
        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        list_img.append(img[y:y+w, x:x+h])
        #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return list_img
    
# cv2.imshow('img',roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()