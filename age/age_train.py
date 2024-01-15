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

def create_age_model():
    # Define the input layer
    input_layer = Input(shape=(64,64,3))

    # First convolutional block
    x = Conv2D(32, (5,5), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2,2))(x)
    x = BatchNormalization()(x)

    # Second convolutional block
    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = BatchNormalization()(x)

    # Third convolutional block
    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    # Flatten the output
    x = Flatten()(x)
    # Age regression branch
    age_branch = Dense(32, activation='relu')(x)
    #age_branch = Dropout(0.5)(age_branch)
    age_branch = Dense(32, activation='relu')(age_branch)
    #age_branch = Dropout(0.5)(age_branch)
    age_branch = Dense(7, activation='softmax', name='age_output')(age_branch)

    # Define the multi-output model
    model = Model(inputs=input_layer, outputs=age_branch)

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics='accuracy')

    return model

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

    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")

    model = create_age_model()
    history = model.fit(aug.flow(x_age_train,y_age_train, batch_size=32),validation_data=(x_age_train,y_age_train), batch_size=32, epochs=50, validation_split=0.2)


    model.save('age/models/pred_age_model1.keras')
    print(model.evaluate(x_age_test, y_age_test,verbose=0))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("age/plot/age_CNN_plot_acc1.png")
    #plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("age/plot/age_CNN_plot_loss1.png")
    #plt.show()