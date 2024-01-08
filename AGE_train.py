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

gender_dict = {'Male':0, 'Female':1}
age_dict = {'20-30s':0,'40-50s':1,'Baby':2,'Kid':3,'Senior':4,'Teenager':5}
# data = pd.read_csv("D:\Python_project\labels.csv")
# pd.options.display.max_columns = None

# print(data.head())
# print('Total rows: {}'.format(len(data)))

# def load_images_from_folder(list_data):
#     df = pd.DataFrame(list_data, columns=['file_name', 'bbox'])
#     images = []
#     i = 0
#     for ind  in df.index:
#         print(i)
#         i = i + 1
#         img = cv2.imread(os.path.join("D:\\Python_project\\data\\",df['file_name'][ind]))
#         bbox = df['bbox'][ind][1:-1].split(',')
#         x = int(float(bbox[0]))-1 if int(float(bbox[0]))>1 else int(float(bbox[0]))
#         y = int(float(bbox[1]))-1 if int(float(bbox[1]))>1 else int(float(bbox[1]))
#         w = int(float(bbox[2]))+1 if int(float(bbox[2]))< img.shape[1]-1 else int(float(bbox[2]))
#         h = int(float(bbox[3]))+1 if int(float(bbox[3]))< img.shape[0]-1 else int(float(bbox[3]))
#         # print("x:", x)
#         # print("y:", y)
#         # print("w:", w)
#         # print("h:", h)
        
#         img = cv2.resize(img[y:y+h, x:x+w], (64,64))
#         # cv2.imshow("img",img)
#         # cv2.waitKey(0)
#         if img is not None:
#             images.append(img)
#     return np.array(images)

# y_age = data['age'].replace(age_dict)
# y_gender = data['gender'].replace(gender_dict)
# print("age: ",y_age)
# x = load_images_from_folder(data)
# #x_gender = load_images_from_folder(data)
# # print("x: " ,x)

# print("start training")
# #x_age_train, x_age_test, y_age_train, y_age_test = train_test_split(x, y_age, test_size=0.22, random_state=37)
# x_gender_train, x_gender_test, y_gender_train, y_gender_test = train_test_split(x, y_gender, test_size=0.22, random_state=37)

# def create_gender_model():
#     # Define the input layer
#     input_layer = Input(shape=(64, 64, 3))

#     # First convolutional block
#     x = Conv2D(32, (3,3), activation='relu', padding='same')(input_layer)
#     x = MaxPooling2D((2,2))(x)
#     x = BatchNormalization()(x)

#     # Second convolutional block
#     x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
#     x = MaxPooling2D((2,2))(x)
#     x = BatchNormalization()(x)

#     # Third convolutional block
#     x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
#     x = MaxPooling2D((2,2))(x)
#     x = BatchNormalization()(x)

#     # Flatten the output
#     x = Flatten()(x)

#     # Gender classification branch
#     gender_branch = Dense(64, activation='relu')(x)
#     gender_branch = Dropout(0.5)(gender_branch)
#     gender_branch = Dense(2, activation='sigmoid', name='gender_output')(gender_branch)

#     # Age regression branch
#     # age_branch = Dense(64, activation='relu')(x)
#     # age_branch = Dropout(0.5)(age_branch)
#     # age_branch = Dense(32, activation='relu')(age_branch)
#     # age_branch = Dropout(0.5)(age_branch)
#     # age_branch = Dense(6, activation='softmax', name='age_output')(age_branch)

#     # Define the multi-output model
#     model = Model(inputs=input_layer, outputs=gender_branch)

#     # Compile the model
#     model.compile(optimizer='adam',
#                 loss={'gender_output': 'sparse_categorical_crossentropy'},
#                 metrics={'gender_output': 'accuracy'})

#     return model

# model = create_gender_model()
# history = model.fit(x_gender_train,y_gender_train,validation_data=(x_gender_train,y_gender_train), batch_size=32, epochs=15, validation_split=0.2)


# model.save('./trained_models/pred_gender_model.keras')
# print(model.evaluate(x_gender_test, y_gender_test,verbose=0))
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.savefig("./plot/gender_CNN_plot_acc.png")
# plt.show()

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.savefig("./plot/gender_CNN_plot_loss.png")
# plt.show()


new_model = tf.keras.models.load_model('./trained_models/pred_gender_model.keras')
# data = pd.read_csv("D:\Python_project\labels.csv", nrows=10000)
# df = pd.DataFrame(data, columns=['file_name', 'gender', 'age'])
# for ind  in df.index:
#     img = cv2.imread(os.path.join("D:\\Python_project\\data\\",df['file_name'][ind]))
#     pred = new_model.predict(np.array([cv2.resize(get_face(img), (64,64))]))
#     gend = np.argmax(pred[0])
#     print("Original Gender", df['gender'][ind])
#     print("Predict Gender:", list(gender_dict.keys())[list(gender_dict.values()).index(gend)])

img = cv2.imread(os.path.join(r"C:\Users\ACER\AI\hackathon\test_img\\male4.jpg"))
pred = new_model.predict(np.array([cv2.resize(get_face(img), (64,64))]))
print("Predict Gender:", list(gender_dict.keys())[list(gender_dict.values()).index(np.argmax(pred[0]))])
cv2.imshow("img",get_face(img))
cv2.waitKey(0)