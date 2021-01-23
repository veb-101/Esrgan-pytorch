import gc
import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

ROOT_DIR_HR = r"images/hr"
ROOT_DIR_LR = r"images/lr"

HR_TRAIN = r"images/train/hr"
LR_TRAIN = r"images/train/lr"

HR_VALID = r"images/valid/hr"
LR_VALID = r"images/valid/lr"


def create_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


create_folder(HR_TRAIN)
create_folder(HR_VALID)
create_folder(LR_TRAIN)
create_folder(LR_VALID)


image_names = os.listdir(ROOT_DIR_HR)
length = len(image_names)

images_train, images_valid = train_test_split(
    image_names, shuffle=True, random_state=41, test_size=64
)

print(len(images_train), len(images_valid))


for image_name in images_train:
    image_path_hr = os.path.join(ROOT_DIR_HR, image_name)
    dest_path_hr = os.path.join(HR_TRAIN, image_name)

    image_path_lr = os.path.join(ROOT_DIR_LR, image_name)
    dest_path_lr = os.path.join(LR_TRAIN, image_name)

    shutil.copy(image_path_hr, dest_path_hr)
    shutil.copy(image_path_lr, dest_path_lr)

gc.collect()

for image_name in images_valid:
    image_path_hr = os.path.join(ROOT_DIR_HR, image_name)
    dest_path_hr = os.path.join(HR_VALID, image_name)

    image_path_lr = os.path.join(ROOT_DIR_LR, image_name)
    dest_path_lr = os.path.join(LR_VALID, image_name)

    shutil.copy(image_path_hr, dest_path_hr)
    shutil.copy(image_path_lr, dest_path_lr)

gc.collect()
