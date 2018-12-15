import cv2
import numpy as np
import scipy.io
from tqdm import tqdm

from utils import get_meta

''' clean-up noisy labels and create database for training '''

# path to output db.mat file
output_path = "data/wiki_db.mat"
# dataset wiki or imdb
db = "wiki"
# output image size
img_size = 64
# minimum face_score
min_score = 1.0

root_path = "data/{}_crop/".format(db)
mat_path = root_path + "{}.mat".format(db)
full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)

out_genders = []
out_ages = []
out_imgs = []

for i in tqdm(range(len(face_score))):
    if face_score[i] < min_score:
        continue

    if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
        continue

    if ~(0 <= age[i] <= 100):
        continue

    if np.isnan(gender[i]):
        continue

    out_genders.append(int(gender[i]))
    out_ages.append(age[i])
    img = cv2.imread(root_path + str(full_path[i][0]))
    out_imgs.append(cv2.resize(img, (img_size, img_size)))

output = {"image": np.array(out_imgs), "gender": np.array(out_genders), "age": np.array(out_ages),
          "db": db, "img_size": img_size, "min_score": min_score}
scipy.io.savemat(output_path, output)
