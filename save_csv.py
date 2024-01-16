import numpy as np
import pandas as pd
import os
import cv2
import json 
import csv
import time
import tensorflow as tf
import glob

from gender.gender_predict import predict_gender
from age.age_predict import predict_age
from ethnicity.ethnicity_predict import predict_race
from skintone.skintone_predict import predict_skintone
from mask.mask_predict import predict_mask
from extract_frontface import get_face, detect_skin_in_color
from emotion.emotion_predict import predict_emotion, emo_model
from extract_skin import extractSkin

def load_image_for_pred(folder_path):
    df = pd.DataFrame(pd.read_csv(r"D:\\Python_project\\answer.csv"), columns=['file_name'])
    i = 0
    with open(r"D:\Python_project\file_name_to_image_id.json", 'r') as fp:
        data = json.load(fp)
    # load model
    gender_model = tf.keras.models.load_model('gender/models/pred_gender_model.keras')
    race_model = tf.keras.models.load_model('ethnicity/models/pred_ethnicity_model.keras')
    age_model = tf.keras.models.load_model('age/models/pred_age_model.keras')
    mask_model = tf.keras.models.load_model("mask\models\pred_mask_model.keras")
    #emo_model = tf.keras.models.load_model('emotion/models/pred_emotion_model.keras')
    emotion_model = emo_model("other_files\\facial_expression_model_weights.h5")

    # Write header of csv
    with open(r"D:\Python_project\answer\answer1.csv", 'w', newline='') as f_object:
        #writer_object = csv.writer(f_object)
        #writer_object.writerow(['file_name','bbox','image_id','race','age','emotion','gender','skintone','masked'])
    #print(data['hjahdjahsgda'])
        #for ind  in df.index:
        for file in glob.glob(r'D:/Python_project/private_test (1)/data/*.jpg'):
            try:
                start_time = time.time()
                # _ = data[df['file_name'][ind]] #test if detect face

                # path = os.path.join(folder_path,df['file_name'][ind])
                img = cv2.imread(file)
                for face in get_face(img):
                    csv_list = []
                    #print("Load image No." + str(data[df['file_name'][ind]]) + "; File: "+ path)

                    #file_name
                    #csv_list.append(df['file_name'][ind])

                    # bounding box
                    print("bbox: ", face[1])
                    csv_list.append(face[1])

                    # image id
                    print("image id: ", os.path.basename(file))
                    #csv_list.append(data[df['file_name'][ind]])
                    csv_list.append(os.path.basename(file))

                    img_resized = np.array([cv2.resize(face[0], (64,64))])
                    img_resized_mask = np.array([cv2.resize(face[0], (128,128))])

                    # race predict
                    print("race: ", predict_race(img_resized, race_model))
                    csv_list.append(predict_race(img_resized, race_model))

                    #age predict
                    print("age: ", predict_age(img_resized, age_model))
                    csv_list.append(predict_age(img_resized, age_model))

                    # emotion predict
                    print("emotion: ", "Neural")
                    #csv_list.append(predict_emo(img_path))
                    csv_list.append(predict_emotion(cv2.cvtColor(cv2.resize(face[0], (48,48)), cv2.COLOR_BGR2GRAY).reshape(-1, 48,48, 1) , emotion_model))

                    print("gender: ", predict_gender(img_resized, gender_model))
                    csv_list.append(predict_gender(img_resized, gender_model))
                    print("skintone: ", predict_skintone(face[0]))
                    csv_list.append(predict_skintone(face[0]))
                    print("masked: ", predict_mask(img_resized_mask, mask_model))
                    csv_list.append(predict_mask(img_resized_mask, mask_model))
                    writer_object = csv.writer(f_object)
                    writer_object.writerow(csv_list)
                print("--- %s seconds ---" % (time.time() - start_time))

            except Exception as e:
                print(e)
    f_object.close()
    # dict = {'file_name':csv_file_name, 
    #     'bbox':csv_bbox, 
    #     'image_id':csv_image_id,
    #     'race':csv_race,
    #     'age':csv_age,
    #     'emotion':csv_emotion,
    #     'gender':csv_gender,
    #     'skintone':csv_skintone,
    #     'masked':csv_mask,
    #    } 
    # df2 = pd.DataFrame(dict).to_csv(r"D:\Python_project\answer\answer.csv")
        

        #prediction_ = pd.DataFrame(predictions, columns=['predictions']).to_csv('prediction.csv')

load_image_for_pred(r"D:\Python_project\public_test\public_test")
