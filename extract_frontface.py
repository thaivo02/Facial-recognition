import numpy as np
import cv2
from matplotlib import pyplot as plt
import dlib

# folder = "./test_img/"
# img_path = 'selfie1.jpg'
# img = cv2.imread(folder+img_path)


def convert_and_trim_bb(image, rect):
    # extract the starting and ending (x, y)-coordinates of the
    # bounding box
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    # ensure the bounding box coordinates fall within the spatial
    # dimensions of the image
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    # compute the width and height of the bounding box
    w = endX - startX
    h = endY - startY
    # return our bounding box coordinates
    return [startX, startY, w, h]


"""Using Haar Cascade to get face of image"""


def get_face(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # gray = cv2.equalizeHist(gray)
    # face_cascade = cv2.CascadeClassifier(
    #     "./other_files/haarcascade_frontalface_default.xml"
    # )
    # faces = face_cascade.detectMultiScale(gray)
    list_img = []
    # if len(faces) != 0:
    #     # if no faces are detected then return original img
    #     # if (len(faces) == 0):
    #     #     return []

    #     for x, y, w, h in faces:
    #         # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #         # list_img.append([img[y:y+w, x:x+h], [x,y, w, h]]) if is_face((x,y,w,h), img) else 1
    #         list_img.append([img[y : y + w, x : x + h], [x, y, w, h]])
    #         # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #     return list_img
    # else:
    # cnn_face_detector = dlib.cnn_face_detection_model_v1("other_files/mmod_human_face_detector.dat")
    # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = dlib.get_frontal_face_detector()(img, 1)
    # #print(faces)
    # list_img = []
    # #bbox = []

    for r in faces:
        list_img.append(img[r.top() : r.bottom(), r.left() : r.right()])
        # cv2.imshow('img',img[r.top():r.bottom(), r.left():r.right()])
        # cv2.waitKey(0)
        # bbox.append(convert_and_trim_bb(img, r))
    return list_img


def is_face(face_coord, image, threshold=0.3):
    """
    Check if the face is a real face.
    Method: detect the skin area in the "face" and check if the skin area is larger than the threshold
    :param face_coord:
    :param image:
    :param is_bw:
    :param threshold:
    :return:
    """
    x, y, w, h = face_coord
    face_image = image[y : y + w, x : x + h]
    _, skin_mask = detect_skin_in_color(face_image)
    skin_pixels = cv2.countNonZero(skin_mask)
    total_pixels = face_image.shape[0] * face_image.shape[1]
    skin_ratio = skin_pixels / total_pixels
    return skin_ratio >= threshold


def detect_skin_in_color(image):
    # Converting from BGR Colors Space to HSV
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Defining skin Thresholds
    low_hsv = np.array([0, 48, 80], dtype=np.uint8)
    high_hsv = np.array([20, 255, 255], dtype=np.uint8)

    skin_mask = cv2.inRange(img, low_hsv, high_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.GaussianBlur(skin_mask, ksize=(3, 3), sigmaX=0)

    skin = cv2.bitwise_and(image, image, mask=skin_mask)
    return skin, skin_mask


# for i in get_face(cv2.imread(r"C:\\Users\ACER\AI\\hackathon\\test_img\\male3.jpg")):
#     cv2.imshow('img',i)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# cv2.imshow('img',detect_skin_in_color(cv2.imread(r"C:\\Users\ACER\AI\\hackathon\\test_img\\senior.jpg"))[0])
# cv2.waitKey(0)
