import numpy as np
import pandas as pd

data = pd.read_csv(r"labels.csv")
emotions = sorted(set(data["emotion"]))
emotion_dict = {emotion: index for index, emotion in enumerate(emotions)}


def predict_emotion(img, model):
    # data = pd.read_csv("D:\Python_project\labels.csv", nrows=10000)
    # df = pd.DataFrame(data, columns=['file_name', 'gender', 'age'])
    # for ind  in df.index:
    #     img = cv2.imread(os.path.join("D:\\Python_project\\data\\",df['file_name'][ind]))
    #     pred = new_model.predict(np.array([cv2.resize(get_face(img), (64,64))]))
    #     gend = np.argmax(pred[0])
    #     print("Original Gender", df['gender'][ind])
    #     print("Predict Gender:", list(gender_dict.keys())[list(gender_dict.values()).index(gend)])

    pred = model(img)
    # print("Predict Gender:", list(gender_dict.keys())[list(gender_dict.values()).index(np.argmax(pred[0]))])
    # cv2.imshow("img",i)
    # cv2.waitKey(0)
    return list(emotion_dict.keys())[
        list(emotion_dict.values()).index(np.argmax(pred[0]))
    ]
