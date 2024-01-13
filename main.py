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
from balance_data.down_sapling import under_sampling
from extract_frontface import get_face, detect_skin_in_color
from extract_skin import extractSkin
import tensorflow as tf
import numpy as np
import cv2
import glob

# # Balance data
# under_sampling("masked")

# Train data
# train_mask_model()

# Predict data
model = tf.keras.models.load_model("mask\models\pred_mask_model.keras")
for i in get_face(cv2.imread(r"C:\\Users\ACER\AI\\hackathon\\test_img\\masked.jpg"))[0]:
# for filename in glob.glob('D:\Python_project\data\*.jpg'):
#     for i in get_face(cv2.imread(filename))[0]:
        img_resized = np.array([cv2.resize(i, (128,128))])
        print(predict_mask(img_resized, model))
        cv2.imshow("img", i)
        cv2.waitKey(0)