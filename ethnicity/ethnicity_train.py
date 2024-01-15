import numpy as np 
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow.keras.layers import Conv2D
import keras.layers as L
from tensorflow.keras.optimizers import Adam, Adadelta
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPooling2D,AvgPool2D,GlobalAveragePooling2D,Dense,Dropout,BatchNormalization,Flatten,Input
from tensorflow.keras.layers import Input,Activation,Add
from tensorflow.keras.regularizers import l2
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
    return np.array(images)

def conv_block(inp, filters=32, bn=True, pool=True):
    _ = Conv2D(filters=filters, kernel_size=3, activation='relu')(inp)
    if pool:
        _ = MaxPooling2D()(_)
    if bn:
        _ = BatchNormalization()(_)
    return _

def create_race_model():
    # CNN Architecture Used is similar to VGG-16
    inp = Input(shape=(64, 64, 3))

    net = Conv2D(filters=16, strides=(2,2), kernel_size=(3,3))(inp)
    net = BatchNormalization()(net)
    net = Activation('elu')(net)
    net = Dropout(0.5)(net)

    net = Conv2D(filters=32, strides=(2,2), kernel_size=(3,3))(net)
    net = BatchNormalization()(net)
    net = Activation('elu')(net)
    net = Dropout(0.5)(net)

    net = Conv2D(filters=64, strides=(2,2), kernel_size=(3,3))(net)
    net = BatchNormalization()(net)
    net = Activation('elu')(net)
    net = Dropout(0.5)(net)

    net = Conv2D(filters=64, strides=(2,2), kernel_size=(3,3))(net)
    net = BatchNormalization()(net)
    net = Activation('elu')(net)
    net = Dropout(0.5)(net)

    net = Conv2D(filters=64, strides=(2,2), kernel_size=(3,3))(net)
    net = BatchNormalization()(net)
    net = Activation('elu')(net)
    net = Dropout(0.5)(net)

    net = Flatten()(net)

    net = Dense(512, activation='relu')(net)
    out = Dense(5, activation='softmax')(net)

    model = Model(inputs=[inp], outputs=[out])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Compiling the above created CNN architecture.
    #model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def train_race_model(rows = 0):
    race_dict = {'Caucasian':0,'Mongoloid':1,'Negroid':2}
    data = pd.read_csv(r"D:\\Python_project\\file_race.csv") if rows == 0 else pd.read_csv(r"D:\\Python_project\\file_race.csv", nrows=rows)
    pd.options.display.max_columns = None

    y_race = data['race'].replace(race_dict)
    print("race: ",y_race)
    x = load_images_from_folder(data)
    #x_age = load_images_from_folder(data)
    # print("x: " ,x)
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest")

    print("start training")
    #x_age_train, x_age_test, y_age_train, y_age_test = train_test_split(x, y_age, test_size=0.22, random_state=37)
    x_race_train, x_race_test, y_race_train, y_race_test = train_test_split(x, y_race, test_size=0.22, random_state=37)

    model = create_race_model()
    history = model.fit(aug.flow(x_race_train,y_race_train, batch_size=32),validation_data=(x_race_train,y_race_train), batch_size=32, epochs=50, validation_split=0.2)


    model.save('ethnicity/models/pred_ethnicity_model1.keras')
    print(model.evaluate(x_race_test, y_race_test,verbose=0))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("ethnicity/plot/ethnicity_CNN_plot_acc1.png")
    #plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("ethnicity/plot/ethnicity_CNN_plot_loss1.png")
    #plt.show()