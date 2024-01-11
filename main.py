from gender.gender_predict import predict_gender
from gender.gender_train import train_gender_model
from age.age_train import train_age_model
from age.age_predict import predict_age
from ethnicity.ethnicity_train import train_race_model
from ethnicity.ethnicity_predict import predict_race
from skintone.skintone_train import train_skintone_model
from skintone.skintone_predict import predict_skintone
from balance_data.down_sapling import under_sampling
from extract_frontface import get_face, detect_skin_in_color
from extract_skin import extractSkin
import cv2


# for i in get_face(cv2.imread(r"C:\\Users\ACER\AI\\hackathon\\test_img\\test_oldman.jpg")):
#get_face(cv2.imread(r"D:\Python_project\data\\84882983.jpg"))
#get_face(cv2.imread(r"C:\\Users\ACER\AI\\hackathon\\test_img\\test1.jpg"))
predict_skintone(r"D:\Python_project\data\2645861.jpg")