import numpy as np
import pandas as pd
import os
import cv2
import json 
import csv
import time
import tensorflow as tf

from gender.gender_predict import predict_gender
from age.age_predict import predict_age
from ethnicity.ethnicity_predict import predict_race
from skintone.skintone_predict import predict_skintone
from mask.mask_predict import predict_mask
from extract_frontface import get_face, detect_skin_in_color
from emotion.emotion_predict import predict_emotion, emo_model
from extract_skin import extractSkin

def load_image_for_pred(folder_path):
    df = pd.DataFrame(pd.read_csv(r"D:\Python_project\answer.csv"), columns=['file_name'])
    with open(r"D:\Python_project\file_name_to_image_id.json", 'r') as fp:
        data = json.load(fp)
    #id_by_name = dict([(p['file_name'], p['id']) for p in data['images']])
    id_by_name = data
    # load model
    gender_model = tf.keras.models.load_model('gender/models/pred_gender_model1.keras')
    #gender_model =cv2.dnn.readNet(r"other_files\gender_net.caffemodel", r"other_files\gender_deploy.prototxt")
    race_model = tf.keras.models.load_model('ethnicity/models/pred_ethnicity_model.keras')
    age_model = tf.keras.models.load_model('age/models/pred_age_model1.keras')
    mask_model = tf.keras.models.load_model("mask\models\pred_mask_model.keras")
    emotion_model = emo_model("other_files\\facial_expression_model_weights.h5")

    # Write header of csv
    with open(r"D:\Python_project\answer\answer_public_test.csv", 'w', newline='') as f_object:
        writer_object = csv.writer(f_object)
        writer_object.writerow(['file_name','bbox','image_id','race','age','emotion','gender','skintone','masked'])
        for ind  in df.index:
            try:
                start_time = time.time()
                #_ = data[df['file_name'][ind]] #test if detect face

                path = os.path.join(folder_path,df['file_name'][ind])
                img = cv2.imread(path)
                faces = get_face(img, 0.9)
                if len(faces)==0:
                    faces = get_face(img, 0.8)
                    if len(faces)==0:
                        faces = get_face(img, 0.7)
                        if len(faces)==0:
                            faces = get_face(img, 0.6)
                            if len(faces)==0:
                                faces = get_face(img, 0.5)
                                if len(faces)==0:
                                    faces = get_face(img, 0.4)
                                    if len(faces)==0:
                                        faces = get_face(img, 0.3)
                                        if len(faces)==0:
                                            faces = get_face(img, 0.2)
                                            if len(faces)==0:
                                                faces = get_face(img, 0.1)
                for face in faces:
                    csv_list = []
                    print("Load image No." + str(id_by_name[df['file_name'][ind]]) + "; File: "+ path)
                    
                    # predict
                    img_resized = np.array([cv2.resize(face[0], (64,64))])
                    file_name = df['file_name'][ind]
                    img_id = id_by_name[df['file_name'][ind]]
                    bbox = face[1]
                    race = predict_race(img_resized, race_model)
                    age = predict_age(img_resized, age_model)
                    emo = predict_emotion(cv2.cvtColor(cv2.resize(face[0], (48,48)), cv2.COLOR_BGR2GRAY).reshape(-1, 48,48, 1) , emotion_model)
                    gender = predict_gender(face[0], gender_model)
                    skintone = predict_skintone(face[0])
                    mask = predict_mask(np.array([cv2.resize(face[0], (128,128))]), mask_model)

                    #file_name
                    print("file name: ", file_name)
                    csv_list.append(file_name)

                    # bounding box
                    print("bbox: ", bbox)
                    csv_list.append(bbox)

                    # image id
                    print("image id: ", img_id)
                    csv_list.append(img_id)

                    # race predict
                    print("race: ", race)
                    csv_list.append(race)

                    #age predict
                    print("age: ", age)
                    csv_list.append(age)

                    # emotion predict
                    print("emotion: ", emo)
                    csv_list.append(emo)

                    # gender predict
                    print("gender: ", gender)
                    csv_list.append(gender)

                    # skintone predict
                    print("skintone: ", skintone)
                    csv_list.append(skintone)

                    # masked predict
                    print("masked: ", mask)
                    csv_list.append(mask)

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
