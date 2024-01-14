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

def create_age_model():
    final_cnn = Sequential()
    # Input layer with 32 filters, followed by an AveragePooling2D layer.
    final_cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(64, 64, 3)))    # 3rd dim = 1 for grayscale images.
    final_cnn.add(AveragePooling2D(pool_size=(2,2)))
    # Three Conv2D layers with filters increasing by a factor of 2 for every successive Conv2D layer.
    final_cnn.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    final_cnn.add(AveragePooling2D(pool_size=(2,2)))
    final_cnn.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
    final_cnn.add(AveragePooling2D(pool_size=(2,2)))
    final_cnn.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
    final_cnn.add(AveragePooling2D(pool_size=(2,2)))
    # A GlobalAveragePooling2D layer before going into Dense layers below.
    # GlobalAveragePooling2D layer gives no. of outputs equal to no. of filters in last Conv2D layer above (256).
    final_cnn.add(GlobalAveragePooling2D())
    # One Dense layer with 132 nodes so as to taper down the no. of nodes from no. of outputs of GlobalAveragePooling2D layer above towards no. of nodes in output layer below (7).
    final_cnn.add(Dense(132, activation='relu'))
    # Output layer with 7 nodes (equal to the no. of classes).
    final_cnn.add(Dense(6, activation='softmax'))
    final_cnn.summary()

    # Compiling the above created CNN architecture.
    final_cnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return final_cnn

def train_age_model(rows = 0):
    age_dict = {'20-30s':0,'40-50s':1,'Baby':2,'Kid':3,'Senior':4,'Teenager':5}
    data = pd.read_csv(r"D:\\Python_project\\file_age.csv") if rows == 0 else pd.read_csv(r"D:\\Python_project\\file_age.csv", nrows=rows)
    pd.options.display.max_columns = None

    y_age = data['age'].replace(age_dict)
    print("age: ",y_age)
    x = load_images_from_folder(data)
    #x_age = load_images_from_folder(data)
    # print("x: " ,x)

    print("start training")
    #x_age_train, x_age_test, y_age_train, y_age_test = train_test_split(x, y_age, test_size=0.22, random_state=37)
    x_age_train, x_age_test, y_age_train, y_age_test = train_test_split(x, y_age, test_size=0.22, random_state=37)

    model = create_age_model()
    history = model.fit(x_age_train,y_age_train,validation_data=(x_age_train,y_age_train), batch_size=32, epochs=15, validation_split=0.2)


    model.save('age/models/pred_age_model.keras')
    print(model.evaluate(x_age_test, y_age_test,verbose=0))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("age/plot/age_CNN_plot_acc.png")
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("age/plot/age_CNN_plot_loss.png")
    plt.show()