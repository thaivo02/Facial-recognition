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

mask_dict = {'masked':0,'unmasked':1}

def predict_mask(img, model):
    # data = pd.read_csv("D:\Python_project\labels.csv", nrows=10000)
    # df = pd.DataFrame(data, columns=['file_name', 'gender', 'age'])
    # for ind  in df.index:
    #     img = cv2.imread(os.path.join("D:\\Python_project\\data\\",df['file_name'][ind]))
    #     pred = new_model.predict(np.array([cv2.resize(get_face(img), (64,64))]))
    #     gend = np.argmax(pred[0])
    #     print("Original Gender", df['gender'][ind])
    #     print("Predict Gender:", list(gender_dict.keys())[list(gender_dict.values()).index(gend)])

    
    pred = model(img)
    #print("Predict Gender:", list(gender_dict.keys())[list(gender_dict.values()).index(np.argmax(pred[0]))])
    #cv2.imshow("img",i)
    #cv2.waitKey(0)
    return list(mask_dict.keys())[list(mask_dict.values()).index(np.argmax(pred[0]))]