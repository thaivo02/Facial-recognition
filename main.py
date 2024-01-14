from balance_data.down_sapling import under_sampling
from extract_frontface import get_face, detect_skin_in_color
from emotion.emotion_predict import predict_emotion
import tensorflow as tf
import numpy as np
import cv2
import glob


# Balance data
# under_sampling("emotion")

# Train data
# train_emotion_model()

# Predict data
model = tf.keras.models.load_model("emotion\models\pred_emotion_model.keras")
for i in get_face(
    cv2.imread(r"D:\Work\Learn\python\hackathon\Facial-recognition\data\13922435.jpg")
):
    # for filename in glob.glob('D:\Python_project\data\*.jpg'):
    #     for i in get_face(cv2.imread(filename))[0]:
    img_resized = cv2.resize(i, (128, 128))
    img_expanded = np.expand_dims(img_resized, axis=0)
    print(predict_emotion(img_expanded, model))
    cv2.imshow("img", i)
    cv2.waitKey(0)
