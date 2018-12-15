import math
import os
from pathlib import Path

import cv2
import dlib
import numpy as np

from keras_lib.WideResNet import WideResNet
from utils import get_meta


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.8, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (0, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)


def yield_local_img(img_dir=""):
    img_dir = Path(img_dir)

    img = cv2.imread(str(img_dir), 1)

    if img is not None:
        h, w, _ = img.shape
        r = 640 / max(w, h)
        yield cv2.resize(img, (int(w * r), int(h * r)))


# Flags
verbose = False
cv2_show = True

# depth of network
depth = 16
# width of network
k = 8
# path to weight file
weight_file = "model/weights.20-3.79.hdf5"
# margin around detected face for age-gender estimation
margin = 0.5

# for face detection
detector = dlib.get_frontal_face_detector()

# load model and weights
img_size = 64
model = WideResNet(img_size, depth=depth, k=k)()
model.load_weights(weight_file)

db = "wiki"
mat_path = "data/{}_crop/{}.mat".format(db, db)
full_path_, dob_, gender_, photo_taken_, face_score_, second_face_score_, age_ \
    = get_meta(mat_path, db)

MAE_a = 0
g = 0
t = 0
for idx, path in enumerate(full_path_):
    if face_score_[idx] > 4.0:
        continue

    if (~np.isnan(second_face_score_[idx])) and second_face_score_[idx] > 0.0:
        continue

    if ~(0 <= age_[idx] <= 100):
        continue

    if np.isnan(gender_[idx]):
        continue

    img_path = (os.getcwd() + "/data/wiki_crop/" + path[0])
    img_generator = yield_local_img(img_path)

    for img in img_generator:
        # opencv uses BGR
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        # The 1 in the second argument indicates that we should up-sample the image
        # 1 time. This will make everything bigger and allow us to detect more
        # faces.
        detected = detector(input_img, 1)
        # [(x1,y1,x2,y2)]=left diagonal edge co-ordinates
        if verbose:
            print(detected)

        # print("Number of faces detected: {}".format(len(detected)))
        faces = np.empty((len(detected), img_size, img_size, 3))

        if len(detected) == 1:
            x1, y1, x2, y2, w, h = detected[0].left(), detected[0].top(), detected[0].right() + 1, detected[
                0].bottom() + 1, detected[0].width(), detected[0].height()
            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), img_w - 1)
            yw2 = min(int(y2 + margin * h), img_h - 1)
            # rectangle w/o margin
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            # rectangle with margin
            # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (0, 255, 255), 2)
            faces[0, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

            # predict ages and genders of the detected faces
            results = model.predict(faces)
            predicted_genders = results[0]
            # ages array [0].[1],[2]....[101]
            ages = np.arange(0, 101).reshape(101, 1)
            # take dot product and make a single list out of lists
            predicted_ages = results[1].dot(ages).flatten()

            real_gender = "F" if gender_[idx] == 0.0 else "M"
            predicted_gender = "F" if predicted_genders[0][0] > 0.5 else "M"

            # account for the assumption that photo was taken in the middle of the year
            if abs(int(predicted_ages[0]) - age_[idx]) < abs(math.ceil(predicted_ages[0]) - age_[idx]):
                diff = abs(int(predicted_ages[0]) - age_[idx])
            else:
                diff = abs(math.ceil(predicted_ages[0]) - age_[idx])

            MAE_a += diff
            if predicted_gender == real_gender:
                g += 1
            t += 1
            print(MAE_a / t, g / t, t)

            if cv2_show:
                # draw results
                tmp_age = "{0:.2f}".format(predicted_ages[0])
                label = "P:{}, P:{}, R:{}, R:{}".format(tmp_age, predicted_gender, age_[idx],
                                                        real_gender)
                draw_label(img, (detected[0].left(), detected[0].top()), label)
                cv2.imshow("DEXTER", img)
                key = cv2.waitKey(-1)

                if key == 27:  # ESC
                    break
