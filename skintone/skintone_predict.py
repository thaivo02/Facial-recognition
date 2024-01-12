import pickle
import cv2
import matplotlib.pyplot as plt
from extract_frontface import get_face,detect_skin_in_color
from extract_skin import extractSkin
import numpy as np
import numpy
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor

Categories =['dark', 'mid-dark', 'mid-light', 'light']
def predict_skintone(path):
    
#     # load
    model = pickle.load(open('skintone/models/pred_skintone_model.pkl', 'rb'))
    for i in get_face(extractSkin(cv2.imread(path)))[0]:
        #print(classify(i))
        l=[cv2.resize(i, (64,64)).flatten()] 
        #print("The predicted image is : "+model.predict(l)[0])
        #cv2.imshow("img", i)
        #cv2.waitKey(0)
        return model.predict(l)[0]
#     plt.imshow(img) 
#     plt.show() 
    

def patch_asscalar(a):
    return a.item()

setattr(numpy, "asscalar", patch_asscalar)

DEFAULT_TONE_PALETTE = {
    "color": [
        "#8d5524",
        "#ac7949",
        "#c29363",
        "#d8ad7d",
    ]
}
DEFAULT_TONE_LABELS = {
    "color": [Categories[i] for i in range(len(DEFAULT_TONE_PALETTE["color"]))]}

def dominant_colors(image, n_clusters=2):
    data = image
    data = np.reshape(data, (-1, 3))
    data = data[np.all(data != 0, axis=1)]
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, colors = cv2.kmeans(data, n_clusters, None, criteria, 10, flags)
    labels, counts = np.unique(labels, return_counts=True)

    order = (-counts).argsort()
    colors = colors[order]
    counts = counts[order]

    percents = counts / counts.sum()

    return colors, percents

def skin_tone(colors, percents, skin_tone_palette, tone_labels):
    lab_tones = [convert_color(sRGBColor.new_from_rgb_hex(rgb), LabColor) for rgb in skin_tone_palette]
    lab_colors = [convert_color(sRGBColor(rgb_r=r, rgb_g=g, rgb_b=b, is_upscaled=True), LabColor) for b, g, r in colors]
    distances = [np.sum([delta_e_cie2000(c, label) * p for c, p in zip(lab_colors, percents)]) for label in lab_tones]
    tone_id = np.argmin(distances)
    distance: float = distances[tone_id]
    tone_hex = skin_tone_palette[tone_id].upper()
    tone_label = tone_labels[tone_id]
    return tone_id, tone_hex, tone_label, distance

def classify(
    image,
    skin_tone_palette = DEFAULT_TONE_PALETTE["color"],
    tone_labels = DEFAULT_TONE_LABELS["color"],
    n_dominant_colors=1,
):
    """
    Classify the skin tone of the image
    :param image: Entire image or image with non-face areas masked
    :param is_bw: Whether the image is black and white
    :param to_bw: Whether to convert the image to black and white
    :param skin_tone_palette:
    :param tone_labels:
    :param n_dominant_colors:
    :param verbose: Whether to output the report image
    :param report_image: The image to draw the report on
    :param use_face: whether to use face area for detection
    :return:
    """
    skin, _ = detect_skin_in_color(image)
    dmnt_colors, dmnt_pcts = dominant_colors(skin, n_dominant_colors)
    # # Generate readable strings
    # hex_colors = ["#%02X%02X%02X" % tuple(np.around([r, g, b]).astype(int)) for b, g, r in dmnt_colors]
    # pct_strs = ["%.2f" % p for p in dmnt_pcts]
    # result = {"dominant_colors": [{"color": color, "percent": pct} for color, pct in zip(hex_colors, pct_strs)]}
    result = {}
    # Calculate skin tone
    tone_id, tone_hex, tone_label, distance = skin_tone(dmnt_colors, dmnt_pcts, skin_tone_palette, tone_labels)
    accuracy = round(100 - distance, 2)
    #result["skin_tone"] = tone_hex
    result["tone_label"] = tone_label
    result["accuracy"] = accuracy
    return result