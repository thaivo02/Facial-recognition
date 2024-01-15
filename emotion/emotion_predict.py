import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Flatten,
    Dense,
    Dropout,
)

emotion_label = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]
data = pd.read_csv(r"D:\Python_project\labels.csv")
emotions = sorted(set(data["emotion"]))
emotion_dict = {emotion: index for index, emotion in enumerate(emotions)}

def emo_model(emotion_pretrained_weight_path):

    num_classes = 7

    model = Sequential()

    # 1st convolution layer
    model.add(Conv2D(64, (5, 5), activation="relu", input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    # 2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    # fully connected neural networks
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation="softmax"))

    # ----------------------------

    model.load_weights(emotion_pretrained_weight_path)

    return model

def predict_emotion(img, model):
    # data = pd.read_csv("D:\Python_project\labels.csv", nrows=10000)
    # df = pd.DataFrame(data, columns=['file_name', 'gender', 'age'])
    # for ind  in df.index:
    #     img = cv2.imread(os.path.join("D:\\Python_project\\data\\",df['file_name'][ind]))
    #     pred = new_model.predict(np.array([cv2.resize(get_face(img), (64,64))]))
    #     gend = np.argmax(pred[0])
    #     print("Original Gender", df['gender'][ind])
    #     print("Predict Gender:", list(gender_dict.keys())[list(gender_dict.values()).index(gend)])

    pred = model(img)
    # print("Predict Gender:", list(gender_dict.keys())[list(gender_dict.values()).index(np.argmax(pred[0]))])
    # cv2.imshow("img",i)
    # cv2.waitKey(0)
    return emotion_label[np.argmax(pred)]
