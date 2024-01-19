import numpy as np
import cv2
from matplotlib import pyplot as plt

# folder = "./test_img/"
# img_path = 'selfie1.jpg'
# img = cv2.imread(folder+img_path)

"""Using Haar Cascade to get face of image"""
def get_face(img, confidence = 0.9):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # #gray = cv2.equalizeHist(gray)
    # list_img = []
    # if camera == False:
    #     faces = dlib.get_frontal_face_detector()(img, 1)
    
    #     if (len(faces) == 0):
    #         #if no faces are detected then return original img
    #         # if (len(faces) == 0):
    #         #     return []

    #         face_cascade = cv2.CascadeClassifier('./other_files/haarcascade_frontalface_default.xml')
    #         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30))
    #         for (x,y,w,h) in faces:
    #             # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #             # list_img.append([img[y:y+w, x:x+h], [x,y, w, h]]) if is_face((x,y,w,h), img) else 1
    #             list_img.append([img[y:y+w, x:x+h], [x,y, w, h]])
    #             #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #         return list_img
    #     else:
    #     #cnn_face_detector = dlib.cnn_face_detection_model_v1("other_files/mmod_human_face_detector.dat")
    #     #rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
    #     # #print(faces)
    #         list_img = []
    #     # #bbox = []
        
    #         for r in faces:
    #             list_img.append([img[r.top():r.bottom(), r.left():r.right()], convert_and_trim_bb(img, r)])
    #             #cv2.imshow('img',img[r.top():r.bottom(), r.left():r.right()])
    #             #cv2.waitKey(0)
    #             #bbox.append(convert_and_trim_bb(img, r))
    #         return list_img
    # else:
    #     face_cascade = cv2.CascadeClassifier('./other_files/haarcascade_frontalface_default.xml')
    #     faces = face_cascade.detectMultiScale(gray)
    #     for (x,y,w,h) in faces:
    #         # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #         # list_img.append([img[y:y+w, x:x+h], [x,y, w, h]]) if is_face((x,y,w,h), img) else 1
    #         list_img.append([img[y:y+w, x:x+h], [x,y, w, h]])
    #         #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #     return list_img
        list_img = []
        try:
                #face_cascade = cv2.CascadeClassifier('./other_files/haarcascade_frontalface_default.xml')
                #faces, _, scores = face_cascade.detectMultiScale3(img, 1.1, 10, outputRejectLevels=True)
                resized = False
                height, width = img.shape[0], img.shape[1]
                image = img.copy()
                if height > 640 or width > 640:
                        r = 640.0 / max(height, width)
                        image = cv2.resize(img, (int(width * r), int(height * r)))
                        height, width = image.shape[0], image.shape[1]
                        resized = True
                faces_detector = cv2.FaceDetectorYN_create(r"other_files\face_detection_yunet_2023mar.onnx", "", (0, 0))
                faces_detector.setInputSize((width, height))
                faces_detector.setScoreThreshold(confidence)
                _, faces = faces_detector.detect(image)
                for face in faces:
                        (x, y, w, h, x_re, y_re, x_le, y_le) = list(map(int, face[:8]))
                        x = max(x, 0)
                        y = max(y, 0)
                        if resized:
                                x, y, w, h = int(x / r), int(y / r), int(w / r), int(h / r)
                        list_img.append([img[int(y) : int(y + h), int(x) : int(x + w)], [x,y, w, h]])
                        # cv2.imshow("img", img[int(y) : int(y + h), int(x) : int(x + w)])
                        # cv2.waitKey(0)
                return list_img
        except Exception as e:
                return []

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