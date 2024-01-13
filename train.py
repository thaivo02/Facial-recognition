import numpy as np
import pandas as pd
import cv2
from keras.layers import Conv2D
import keras.layers as L
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    Flatten,
)
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from keras.preprocessing.image import ImageDataGenerator

train_dir = "data/train"
test_dir = "data/test"


def load_images_from_folder(list_data):
    df = pd.DataFrame(list_data, columns=["file_name"])
    images = []
    i = 0
    for ind in df.index:
        print(i)
        i = i + 1
        img_path = os.path.join(train_dir, df["file_name"][ind])
        if os.path.exists(img_path):
            img = cv2.imread(os.path.join(train_dir, df["file_name"][ind]))
        else:
            img = cv2.imread(os.path.join(test_dir, df["file_name"][ind]))
        img = cv2.resize(img, (128, 128))
        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        if img is not None:
            images.append(img)
    return np.array(images).astype("float32")


def create_emotion_model():
    # Create the model
    model = Sequential()

    model.add(
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1))
    )

    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(256, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.2))

    model.add(Dense(7, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())
    return model


def train_emotion_model(rows=0):
    emotion_dict = {
        "Anger": 0,
        "Disgust": 1,
        "Fear": 2,
        "Happiness": 3,
        "Neutral": 4,
        "Sadness": 5,
        "Suprise": 6,
    }
    data = (
        pd.read_csv(r"data/labels.csv")
        if rows == 0
        else pd.read_csv(r"data/labels.csv", nrows=rows)
    )
    pd.options.display.max_columns = None

    y_emotion = data["emotion"].replace(emotion_dict)
    print("emotion: ", y_emotion)
    x = load_images_from_folder(data)
    # x_age = load_images_from_folder(data)
    # print("x: " ,x)

    print("start training")
    # x_age_train, x_age_test, y_age_train, y_age_test = train_test_split(x, y_age, test_size=0.22, random_state=37)
    x_emotion_train, x_emotion_test, y_emotion_train, y_emotion_test = train_test_split(
        x, y_emotion, test_size=0.2, random_state=42
    )

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
        aug.flow(x_emotion_train, y_emotion_train, batch_size=32),
        validation_data=(x_emotion_train, y_emotion_train),
        batch_size=32,
        epochs=15,
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
    plt.savefig("emotion/plot/emotion_CNN_plot_acc.png")
    plt.show()

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig("emotion/plot/emotion_CNN_plot_loss.png")
    plt.show()


train_emotion_model(10)
