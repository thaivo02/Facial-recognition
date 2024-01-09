from gender.gender_predict import predict_gender
from gender.gender_train import train_gender_model
from age.age_train import train_age_model
from age.age_predict import predict_age
from ethnicity.ethnicity_train import train_race_model
from ethnicity.ethnicity_predict import predict_race
from balance_data.down_sapling import under_sampling
from extract_frontface import get_face

# train_gender_model()

path = r"C:\Users\ACER\AI\hackathon\test_img\\female4.jpg"
#path = r"D:\\Python_project\\data\\73876047.jpg"
predict_gender(path)

# col = "gender"
# under_sampling(col)