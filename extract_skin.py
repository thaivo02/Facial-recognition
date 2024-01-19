import numpy as np
import cv2
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from extract_frontface import get_face
import pandas as pd

def get_hsv_mask(img, debug=False):
    assert isinstance(img, np.ndarray), 'image must be a np array'
    assert img.ndim == 3, 'skin detection can only work on color images'

    lower_thresh = np.array([0, 50, 0], dtype=np.uint8)
    upper_thresh = np.array([120, 150, 255], dtype=np.uint8)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    msk_hsv = cv2.inRange(img_hsv, lower_thresh, upper_thresh)

    msk_hsv[msk_hsv < 128] = 0
    msk_hsv[msk_hsv >= 128] = 1

    return msk_hsv.astype(float)


def get_rgb_mask(img, debug=False):
    assert isinstance(img, np.ndarray), 'image must be a np array'
    assert img.ndim == 3, 'skin detection can only work on color images'

    lower_thresh = np.array([45, 52, 108], dtype=np.uint8)
    upper_thresh = np.array([255, 255, 255], dtype=np.uint8)

    mask_a = cv2.inRange(img, lower_thresh, upper_thresh)
    mask_b = 255 * ((img[:, :, 2] - img[:, :, 1]) / 20)
    mask_c = 255 * ((np.max(img, axis=2) - np.min(img, axis=2)) / 20)
    # msk_rgb = cv2.bitwise_and(mask_c, cv2.bitwise_and(mask_a, mask_b))
    mask_d = np.bitwise_and(np.uint64(mask_a), np.uint64(mask_b))
    msk_rgb = np.bitwise_and(np.uint64(mask_c), np.uint64(mask_d))

    msk_rgb[msk_rgb < 128] = 0
    msk_rgb[msk_rgb >= 128] = 1

    return msk_rgb.astype(float)


def get_ycrcb_mask(img, debug=False):
    assert isinstance(img, np.ndarray), 'image must be a np array'
    assert img.ndim == 3, 'skin detection can only work on color images'

    lower_thresh = np.array([90, 100, 130], dtype=np.uint8)
    upper_thresh = np.array([230, 120, 180], dtype=np.uint8)

    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    msk_ycrcb = cv2.inRange(img_ycrcb, lower_thresh, upper_thresh)

    msk_ycrcb[msk_ycrcb < 128] = 0
    msk_ycrcb[msk_ycrcb >= 128] = 1

    return msk_ycrcb.astype(float)

def extractSkin(image):
    # Converting from BGR Colors Space to HSV
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Defining skin Thresholds
    low_hsv = np.array([0, 20, 70], dtype=np.uint8)
    high_hsv = np.array([20, 255, 255], dtype=np.uint8)

    skin_mask = cv2.inRange(img, low_hsv, high_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    #skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.GaussianBlur(skin_mask, ksize=(3, 3), sigmaX=0)

    skin = cv2.bitwise_and(image, image, mask=skin_mask)
    if cv2.countNonZero(cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)) == 0:
    # print(global_result.shape)
    # Return the Skin image
        return image
    else:
        return skin


#for i in get_face(cv2.imread(r"C:\\Users\ACER\AI\\hackathon\\test_img\\male3.jpg")):
# cv2.imshow("img", extractSkin(cv2.imread(r"C:\\Users\ACER\AI\\hackathon\\test_img\\senior.jpg")))
# cv2.waitKey(0)

