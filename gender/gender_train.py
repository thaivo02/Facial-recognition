import numpy as np 
import pandas as pd
import tensorflow as tf
import cv2
from keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, AveragePooling2D
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

def create_gender_model():
    # CNN Architecture
    input = Input(shape = (64,64,3))
    conv1 = Conv2D(32,(3, 3), padding = 'same', strides=(1, 1), kernel_regularizer=l2(0.001))(input)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size = (2,2)) (conv1)
    conv2 = Conv2D(64,(3, 3), padding = 'same', strides=(1, 1), kernel_regularizer=l2(0.001))(pool1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size = (2,2)) (conv2)
    conv3 = Conv2D(128,(3, 3), padding = 'same', strides=(1, 1), kernel_regularizer=l2(0.001))(pool2)
    conv3 = Dropout(0.1)(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size = (2,2)) (conv3)
    conv4 = Conv2D(256,(3, 3), padding = 'same', strides=(1, 1), kernel_regularizer=l2(0.001))(pool3)
    conv4 = Dropout(0.1)(conv4)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size = (2,2)) (conv4)
    flatten = Flatten()(pool4)
    dense_1 = Dense(128,activation='relu')(flatten)
    drop_1 = Dropout(0.2)(dense_1)
    output = Dense(2,activation="sigmoid")(drop_1)

    # Model compile
    model = Model(inputs=input,outputs=output)
    model.compile(optimizer="adam",loss=["sparse_categorical_crossentropy"],metrics=['accuracy'])

    return model

def train_gender_model(rows = 0):
    gender_dict = {'Male':0, 'Female':1}
    #age_dict = {'20-30s':0,'40-50s':1,'Baby':2,'Kid':3,'Senior':4,'Teenager':5}
    data = pd.read_csv(r"D:\\Python_project\\file_gender.csv") if rows == 0 else pd.read_csv(r"D:\\Python_project\\file_gender.csv", nrows=rows)
    pd.options.display.max_columns = None

    #y_age = data['age'].replace(age_dict)
    y_gender = data['gender'].replace(gender_dict)
    #print("age: ",y_age)
    x = load_images_from_folder(data)
    #x_gender = load_images_from_folder(data)
    # print("x: " ,x)

    print("start training")
    #x_age_train, x_age_test, y_age_train, y_age_test = train_test_split(x, y_age, test_size=0.22, random_state=37)
    x_gender_train, x_gender_test, y_gender_train, y_gender_test = train_test_split(x, y_gender, test_size=0.22, random_state=37)

    model = create_gender_model()
    history = model.fit(x_gender_train,y_gender_train,validation_data=(x_gender_train,y_gender_train), batch_size=32, epochs=15, validation_split=0.2)


    model.save('gender/models/pred_gender_model.keras')
    print(model.evaluate(x_gender_test, y_gender_test,verbose=0))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("./plot/gender_CNN_plot_acc.png")
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("./plot/gender_CNN_plot_loss.png")
    plt.show()