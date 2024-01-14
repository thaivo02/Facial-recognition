import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

def under_sampling(col):
    #read data
    data = pd.read_csv("D:\Python_project\labels.csv")

    x, _ = RandomUnderSampler(sampling_strategy= 'all',random_state=42).fit_resample(data, data[col])

    pd.DataFrame(x).to_csv(r'D:\\Python_project\\file_'+col+'.csv')