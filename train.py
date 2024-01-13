from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2

import warnings

# Suppress deprecated optimizer warning
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

def load_images_from_folder(list_data):
    df = pd.DataFrame(list_data, columns=['file_name'])
    images = []
    i = 0
    for ind  in df.index:
        print(i)
        i = i + 1
        try:
            img = cv2.imread(os.path.join("data/train/",df['file_name'][ind]))
        except:
            img = cv2.imread(os.path.join("data/test/",df['file_name'][ind]))
        
        img = cv2.resize(img, (64,64))
        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        if img is not None:
            images.append(img)
    return np.array(images)


IMG_HEIGHT = 64
IMG_WIDTH = 64
batch_size = 32

train_data_dir = "data/train/"
validation_data_dir = "data/test/"

labels = pd.read_csv("data/labels.csv")
class_labels = labels["emotion"].unique()
num_classes = len(class_labels)

emo_dict = {'Anger':0, 'Disgust':1, 'Fear':2, 'Happiness':3, 'Neutral':4, 'Sadness':5, 'Suprise': 6}
x = load_images_from_folder(labels)
y = labels['emotion'].replace(emo_dict)
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42)

data_aug = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode="nearest",
)



def load_and_preprocess_image(img_path, target_size):
    # Placeholder implementation using PIL
    img = Image.open(img_path)
    img = img.resize(target_size)
    img = np.array(img) / 255.0  # Normalize pixel values
    return img


# Custom data generator for training
def custom_data_generator(df, datagen, directory, batch_size, target_size, num_classes):
    label_encoder = LabelEncoder()
    label_encoder.fit(class_labels)

    while True:
        batch = df.sample(batch_size)
        images = []
        labels = []

        for index, row in batch.iterrows():
            # Load and preprocess the image
            img_path = os.path.join(directory, row["file_name"])
            if not os.path.exists(img_path):
                continue
            img = load_and_preprocess_image(img_path, target_size)
            images.append(img)

            # Add the corresponding label
            labels.append(row["emotion"])

        # Convert string labels to numerical labels
        labels = label_encoder.transform(labels)

        # Convert numerical labels to one-hot encoding
        labels = to_categorical(labels, num_classes=num_classes)

        yield (np.array(images), np.array(labels))


# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     color_mode="grayscale",
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=batch_size,
#     class_mode="categorical",
#     shuffle=True,
# )

# validation_generator = validation_datagen.flow_from_directory(
#     validation_data_dir,
#     color_mode="grayscale",
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=batch_size,
#     class_mode="categorical",
#     shuffle=True,
# )

train_generator = custom_data_generator(
    train_df,
    train_datagen,
    train_data_dir,
    batch_size,
    (IMG_HEIGHT, IMG_WIDTH),
    num_classes,
)
validation_generator = custom_data_generator(
    val_df,
    validation_datagen,
    validation_data_dir,
    batch_size,
    (IMG_HEIGHT, IMG_WIDTH),
    num_classes,
)

img, label = train_generator.__next__()

import random

i = random.randint(0, (img.shape[0]) - 1)
image = img[i]
labl = class_labels[label[i].argmax()]
plt.imshow(image[:, :, 0], cmap="gray")
plt.title(labl)
plt.show()
##########################################################


###########################################################
# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 1)))

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

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print(model.summary())


train_path = "data/train/"
test_path = "data/test"

num_train_imgs = 0
for root, dirs, files in os.walk(train_path):
    num_train_imgs += len(files)

num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
    num_test_imgs += len(files)


epochs = 50

history = model.fit(
    data_aug.flow(x_train, y_train, batch_size=batch_size),,
    steps_per_epoch=num_train_imgs // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=num_test_imgs // batch_size,
)

model.save("emotion_detection_model_100epochs.h5")

# plot the training and validation accuracy and loss at each epoch
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "y", label="Training loss")
plt.plot(epochs, val_loss, "r", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

acc = history.history["accuracy"]
# acc = history.history['accuracy']
val_acc = history.history["val_accuracy"]
# val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, "y", label="Training acc")
plt.plot(epochs, val_acc, "r", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

####################################################################
from keras.models import load_model


# Test the model
my_model = load_model("emotion_detection_model_100epochs.h5", compile=False)

# Generate a batch of images
test_img, test_lbl = validation_generator.__next__()
predictions = my_model.predict(test_img)

predictions = np.argmax(predictions, axis=1)
test_labels = np.argmax(test_lbl, axis=1)

from sklearn import metrics

print("Accuracy = ", metrics.accuracy_score(test_labels, predictions))

# Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, predictions)
# print(cm)
import seaborn as sns

sns.heatmap(cm, annot=True)

class_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
# Check results on a few select images
n = random.randint(0, test_img.shape[0] - 1)
image = test_img[n]
orig_labl = class_labels[test_labels[n]]
pred_labl = class_labels[predictions[n]]
plt.imshow(image[:, :, 0], cmap="gray")
plt.title("Original label is:" + orig_labl + " Predicted is: " + pred_labl)
plt.show()
