from gender.gender_predict import predict_gender
from gender.gender_train import train_gender_model
from age.age_train import train_age_model
from age.age_predict import predict_age
from ethnicity.ethnicity_train import train_race_model
from ethnicity.ethnicity_predict import predict_race
from skintone.skintone_train import train_skintone_model
from skintone.skintone_predict import predict_skintone
from mask.mask_train import train_mask_model
from mask.mask_predict import predict_mask
from emotion.emotion_train import train_emotion_model
from emotion.emotion_predict import predict_emotion, emo_model
from balance_data.down_sapling import under_sampling
from extract_frontface import get_face, detect_skin_in_color
from extract_skin import extractSkin
import tensorflow as tf
import numpy as np
import cv2
from deepface import DeepFace

# # Balance data
# under_sampling("emotion")

# Train data
# # train_age_model()
# train_emotion_model()
# train_gender_model()
# train_mask_model()
# train_race_model()
# train_skintone_model()

# Predict data
path_to_image = "D:\Python_project\data\\95357413.jpg"
# gender_model = tf.keras.models.load_model('gender/models/pred_gender_model1.keras')
# race_model = tf.keras.models.load_model('ethnicity/models/pred_ethnicity_model.keras')
# age_model = tf.keras.models.load_model('age/models/pred_age_model1.keras')
mask_model = tf.keras.models.load_model("mask\models\pred_mask_model.keras")
# emotion_model = emo_model("other_files\\facial_expression_model_weights.h5")
for face in get_face(cv2.imread(path_to_image), 0.8):
    img_resized = np.array([cv2.resize(face[0], (64,64))])
    bbox = face[1]
    # print(predict_race(img_resized, race_model))
    # print(predict_age(img_resized, age_model))
    # print(predict_emotion(cv2.cvtColor(cv2.resize(face[0], (48,48)), cv2.COLOR_BGR2GRAY).reshape(-1, 48,48, 1) , emotion_model))
    # print(predict_gender(img_resized, gender_model))
    # print(predict_skintone(face[0]))
    print(predict_mask(np.array([cv2.resize(face[0], (128,128))]), mask_model))
    cv2.imshow("img", face[0])
    cv2.waitKey(0)