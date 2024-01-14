import sys
import dlib
import cv2
import numpy as np
import tensorflow as tf
import time

from extract_frontface import get_face
from gender.gender_predict import predict_gender
from age.age_predict import predict_age
from ethnicity.ethnicity_predict import predict_race
from skintone.skintone_predict import predict_skintone
from mask.mask_predict import predict_mask

cam = cv2.VideoCapture(0)
color_green = (0,255,0)
line_width = 3
# Modify display
font_size = 0.5
font_thickness = 1
font_color = (36,255,12)
font = cv2.FONT_HERSHEY_SIMPLEX
# load model
gender_model = tf.keras.models.load_model('gender/models/pred_gender_model.keras')
race_model = tf.keras.models.load_model('ethnicity/models/pred_ethnicity_model.keras')
age_model = tf.keras.models.load_model('age/models/pred_age_model.keras')
mask_model = tf.keras.models.load_model("mask\models\pred_mask_model.keras")
# used to record the time when we processed last frame 
prev_frame_time = 0
  
# used to record the time at which we processed current frame 
new_frame_time = 0
while True:
    ret_val, img = cam.read()

    #Calc FPS
    new_frame_time = time.time() 
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 

    cv2.putText(img, "FPS: " +str(fps) , (0, 0),font, font_size, font_color, font_thickness)
    faces = get_face(img)
    for face in faces[1]:
        x = face[0]
        y = face[1]
        cv2.rectangle(img,(face[0], face[1]), (face[0]+face[2], face[1]+face[3]), color_green, line_width)
    for i in faces[0]:
        # Predict
        img_resized = np.array([cv2.resize(i, (64,64))])
        cv2.putText(img,"Race: "+predict_race(img_resized, race_model), (x, y-10), font, font_size, font_color, font_thickness)
        cv2.putText(img,"Age: "+predict_age(img_resized, age_model), (x, y-20), font, font_size, font_color, font_thickness)
        cv2.putText(img,"Gender: "+predict_gender(img_resized, gender_model), (x, y-30), font, font_size, font_color, font_thickness)
        cv2.putText(img,"Skintone: "+predict_skintone(i), (x, y-40), font, font_size, font_color, font_thickness)
        cv2.putText(img,"Mask: "+predict_mask(np.array([cv2.resize(i, (128,128))]), mask_model), (x, y-50), font, font_size, font_color, font_thickness)
    cv2.imshow('my webcam', img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()