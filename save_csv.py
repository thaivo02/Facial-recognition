import numpy as np
import pandas as pd
import os
import cv2
import json 
import csv
import time

from gender.gender_predict import predict_gender
from age.age_predict import predict_age
from ethnicity.ethnicity_predict import predict_race
from skintone.skintone_predict import predict_skintone
from extract_frontface import get_face, detect_skin_in_color
from extract_skin import extractSkin

def load_image_for_pred(folder_path):
    df = pd.DataFrame(pd.read_csv(r"D:\\Python_project\\answer.csv"), columns=['file_name'])
    images = []
    i = 0
    path_answer_csv = "D:\\Python_project\\answer\\answer.csv"
    with open(r"D:\Python_project\file_name_to_image_id.json", 'r') as fp:
        data = json.load(fp)
    csv_file_name = []
    csv_bbox = []
    csv_image_id= []
    csv_race = []
    csv_age = []
    csv_emotion = []
    csv_gender = []
    csv_skintone = []
    csv_mask = []
    with open(r"D:\Python_project\answer\answer.csv", 'w', newline='') as f_object:
        writer_object = csv.writer(f_object)
        writer_object.writerow(['file_name','bbox','image_id','race','age','emotion','gender','skintone','masked'])
    #print(data['hjahdjahsgda'])
        for ind  in df.index:
            try:
                start_time = time.time()
                csv_list = []
                _ = data[df['file_name'][ind]]
                csv_file_name.append(df['file_name'][ind])
                print("Load image No." + str(i) + "; File: "+ os.path.join(folder_path,df['file_name'][ind]))
                csv_list.append(df['file_name'][ind])
                i = i + 1
                img_path = os.path.join(folder_path,df['file_name'][ind])
                img = cv2.imread(img_path)
                print("bbox: ", get_face(img)[1][0])
                csv_list.append(get_face(img)[1][0])
                print("image id: ", data[df['file_name'][ind]])
                csv_list.append(data[df['file_name'][ind]])
                print("race: ", predict_race(img_path))
                csv_list.append(predict_race(img_path))
                print("age: ", predict_age(img_path))
                csv_list.append(predict_age(img_path))
                print("emotion: ", "Neural")
                #csv_list.append(predict_emo(img_path))
                csv_list.append("Neural")
                print("gender: ", predict_gender(img_path))
                csv_list.append(predict_gender(img_path))
                print("skintone: ", predict_skintone(img_path))
                csv_list.append(predict_skintone(img_path))
                print("masked: ", "unmasked")
                #csv_list.append(predict_mask(img_path))
                csv_list.append("unmasked")
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
