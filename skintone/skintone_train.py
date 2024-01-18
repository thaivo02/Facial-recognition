import numpy as np
import pandas as pd
import cv2
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from extract_skin import extractSkin
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV 

def load_images_from_folder(list_data):
    df = pd.DataFrame(list_data, columns=['file_name'])
    images = []
    i = 0
    for ind  in df.index:
        print(i)
        i = i + 1
        img = extractSkin(cv2.imread(os.path.join("D:\\Python_project\\crop_img_data\\",df['file_name'][ind])))
        
        img = cv2.resize(img, (64,64))
        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        if img is not None:
            images.append(img.flatten())
    return np.array(images)

def skintone_model(x_train, x_test, y_train, y_test):
    # # defining parameter range 
    # param_grid={'C':[100], 
    #         'gamma':['auto'], 
    #         'kernel':['linear', 'rbf', 'sigmoid']} 
    # grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
    # # fitting the model for grid search 
    # grid.fit(x_train, y_train) 
    

    # # print best parameter after tuning 
    # print(grid.best_params_) 
    
    # # print how our model looks after hyper-parameter tuning 
    # print(grid.best_estimator_) 

    # Create a kernel support vector machine model
    ksvm = SVC(C=1, gamma='scale', kernel='rbf').fit(x_train, y_train)
    
    # Evaluate the model on the test data
    y_pred = ksvm.predict(x_test) 
    print(classification_report(y_test, y_pred, target_names=['light', 'mid-light', 'dark', 'mid-dark']))
    confusion_matrix_plot=confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_plot, display_labels = ['light', 'mid-light', 'dark', 'mid-dark'])
    cm_display.plot()
    plt.savefig('skintone\plot\confusion_matrix1.jpg')
    plt.show()  
    # save
    with open('skintone/models/pred_skintone_model1.pkl','wb') as f:
        pickle.dump(ksvm,f)

def train_skintone_model():
    data = pd.read_csv(r"D:\\Python_project\\file_skintone.csv")
    y_skintone = data['skintone']
    x = load_images_from_folder(data)
    x_skintone_train, x_skintone_test, y_skintone_train, y_skintone_test = train_test_split(x, y_skintone, test_size=0.22, random_state=37)
    # Import the necessary modules and libraries
    # Scale the features using standardization
    print("start training")
    skintone_model(x_skintone_train,x_skintone_test, y_skintone_train, y_skintone_test)