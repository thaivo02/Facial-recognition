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
# # train_gender_model()
train_mask_model()
# train_race_model()

# Predict data
# model = tf.keras.models.load_model("emotion/models/pred_emotion_model1.keras")
# # for i in get_face(cv2.imread(r"C:\\Users\ACER\AI\\hackathon\\test_img\\masked.jpg")):
# #for filename in glob.glob('D:\Python_project\data\\58837884.jpg'):
# for i in get_face(cv2.imread(r"D:\Python_project\data\10492403.jpg")):
#     img_resized = np.array([cv2.resize(i[0], (64,64))])
#     print(predict_emotion(img_resized))
#     cv2.imshow("img", i[0])
#     cv2.waitKey(0)