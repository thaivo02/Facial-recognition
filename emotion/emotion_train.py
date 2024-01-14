import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from keras.layers import Conv2D
import keras.layers as L
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten,BatchNormalization
from tensorflow.keras.layers import Dense, MaxPooling2D,Conv2D, MaxPool2D
from tensorflow.keras.layers import Input,Activation,Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from keras.preprocessing.image import ImageDataGenerator


def load_images_from_folder(list_data):
    df = pd.DataFrame(list_data, columns=['file_name'])
    images = []
    i = 0
    for ind  in df.index:
        print(i)
        i = i + 1
        img = cv2.imread(os.path.join("D:\\Python_project\\crop_img_data\\",df['file_name'][ind]))
        
        img = cv2.resize(img, (64,64))
        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        if img is not None:
            images.append(img)
    return np.array(images)


def create_emotion_model():
    input_shape=(64,64,3)

    model=Sequential([
                  Conv2D(64,3,activation='relu',kernel_initializer='he_normal',input_shape=(64,64,3)),
                  MaxPooling2D(3),
                  Conv2D(128,3,activation='relu',kernel_initializer='he_normal'),
                  Conv2D(256,3,activation='relu',kernel_initializer='he_normal'),
                  MaxPooling2D(3),
                  Flatten(),
                  Dense(256,activation='relu'),
                  Dense(7,activation='softmax',kernel_initializer='glorot_normal')
                  
])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    print(model.summary())
    return model


def train_emotion_model(rows=0):
    data = (
        pd.read_csv(r"D:\Python_project\file_emotion.csv")
        if rows == 0
        else pd.read_csv(r"D:\Python_project\file_emotion.csv", nrows=rows)
    )
    pd.options.display.max_columns = None
    emotion_dict = {'Anger':0,'Disgust':1,'Fear':2,'Happiness':3,'Neutral':4,'Sadness':5, 'Surprise': 6}

    y_emotion = data["emotion"].replace(emotion_dict)
    print("emotion: ", y_emotion)
    x = load_images_from_folder(data)

    print("start training")
    x_emotion_train, x_emotion_test, y_emotion_train, y_emotion_test = train_test_split(x, y_emotion, test_size=0.2, random_state=42)
    print(x.shape, y_emotion.shape)

    # construct the training image generator for data augmentation
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    model = create_emotion_model()
    history = model.fit(
        x_emotion_train, y_emotion_train,
        validation_data=(x_emotion_train, y_emotion_train),
        batch_size=32,
        epochs=30,
        validation_split=0.2,
    )

    model.save("emotion/models/pred_emotion_model.keras")
    print(model.evaluate(x_emotion_test, y_emotion_test, verbose=0))
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig("plot/emotion_CNN_plot_acc.png")
    plt.show()

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig("plot/emotion_CNN_plot_loss.png")
    plt.show()
