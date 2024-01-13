import numpy as np 
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow.keras.layers import Conv2D, AveragePooling2D
import keras.layers as L
from tensorflow.keras.optimizers import Adam, Adadelta
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPooling2D,AvgPool2D,GlobalAveragePooling2D,Dense,Dropout,BatchNormalization,Flatten,Input
from tensorflow.keras.layers import Input,Activation,Add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import MobileNetV2
from keras.layers import MaxPool2D, GlobalMaxPool2D
from keras.optimizers import SGD
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from extract_frontface import get_face
import glob
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
    return np.array(images).astype('float32')

def create_mask_model():
    # # model=Sequential([
    # #  tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    # #  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    # #  tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2,0.2),
    # #  ])
    # baseModel = MobileNetV2(weights="imagenet", include_top=False,input_shape=(128,128,3))

    # # construct the head of the model that will be placed on top of the
    # # the base model
    # headModel = baseModel.output
    # headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    # headModel = Flatten(name="flatten")(headModel)
    # headModel = Dense(128, activation="relu")(headModel)
    # headModel = Dropout(0.5)(headModel)
    # headModel = Dense(2, activation="sigmoid")(headModel)

    # # place the head FC model on top of the base model (this will become
    # # the actual model we will train)
    # model = Model(inputs=baseModel.input, outputs=headModel)

    # # loop over all layers in the base model and freeze them so they will
    # # *not* be updated during the first training process
    # for layer in baseModel.layers:
    #     layer.trainable = False

    # # compile our model
    # print("[INFO] compiling model...")
    # model.compile(loss="sparse_categorical_crossentropy", optimizer='adam',
    #     metrics=["accuracy"])
    # return model

    # Create the model
    model = Sequential()

    model.add(
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(64, 64, 3))
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

def train_mask_model(rows = 0):
    #mask_dict = {'masked':0,'unmasked':1}
    # data = pd.read_csv(r"D:\\Python_project\\file_masked.csv") if rows == 0 else pd.read_csv(r"D:\\Python_project\\file_masked.csv", nrows=rows)
    data = pd.read_csv(r"D:\\Python_project\\labels.csv") if rows == 0 else pd.read_csv(r"D:\\Python_project\\labels.csv", nrows=rows)
    pd.options.display.max_columns = None

    # y_mask = data['masked'].replace('unmasked',1)
    # y_mask = y_mask.replace('masked',0)
    emotion_dict = {
        "Anger": 0,
        "Disgust": 1,
        "Fear": 2,
        "Happiness": 3,
        "Neutral": 4,
        "Sadness": 5,
        "Suprise": 6,
    }
    y_mask = data["emotion"].replace(emotion_dict)
    print("mask: ",y_mask)
    x = load_images_from_folder(data)
    #x_age = load_images_from_folder(data)
    # print("x: " ,x)

    print("start training")
    #x_age_train, x_age_test, y_age_train, y_age_test = train_test_split(x, y_age, test_size=0.22, random_state=37)
    x_mask_train, x_mask_test, y_mask_train, y_mask_test = train_test_split(x, y_mask, test_size=0.2, random_state=42)

    # construct the training image generator for data augmentation
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    model = create_mask_model()
    history = model.fit(aug.flow(x_mask_train, y_mask_train, batch_size=32),validation_data=(x_mask_train,y_mask_train), batch_size=32, epochs=15, validation_split=0.2)


    # model.save('mask/models/pred_mask_model.keras')
    # print(model.evaluate(x_mask_test, y_mask_test,verbose=0))
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.savefig("mask/plot/mask_CNN_plot_acc.png")
    # plt.show()

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.savefig("mask/plot/mask_CNN_plot_loss.png")
    # plt.show()
