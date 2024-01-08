import cv2 as cv
from extract_frontface import get_face

def age_pred_cvDNN(img):
    img = get_face(img)
    blob = cv.dnn.blobFromImage(img, 1.0, (227, 227), [104, 117, 123], True, False)

    ageProto = "./other_files/age_deploy.prototxt"
    ageModel = "./other_files/age_net.caffemodel"
    ageNet = cv.dnn.readNet(ageModel, ageProto)
    
    ageList = ['(0 - 2)', '(4 - 6)', '(8 - 12)', '(15 - 20)', '(25 - 32)', '(38 - 43)', '(48 - 53)', '(60 - 100)']
    
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]
    return age