import os
import shutil
import cv2
import random


HR_FOLDER = r"images/hr"
LR_FOLDER = r"images/lr"

ROOT_DIR = r"images_og"
HR_SIZE = 512
TAKE = 4
start = 0

folder = os.listdir(ROOT_DIR)
length = len(folder)


def create_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


create_folder(HR_FOLDER)
create_folder(LR_FOLDER)

for idx, image_name in enumerate(sorted(folder)):
    # if idx == 10:
    #     break
    image_path = os.path.join(ROOT_DIR, image_name)
    raw_image = cv2.imread(image_path)

    raw_w, raw_h, _ = raw_image.shape

    h_limit = raw_h - HR_SIZE
    w_limit = raw_w - HR_SIZE
    # print(raw_w, raw_h, w_limit, h_limit)

    for c in range(TAKE):
        h_start = random.randint(0, h_limit)
        w_start = random.randint(0, w_limit)
        # print(w_start, h_start)

        w_end = w_start + HR_SIZE
        h_end = h_start + HR_SIZE

        image_ = raw_image[w_start:w_end, h_start:h_end]
        assert image_.shape == (
            512,
            512,
            3,
        ), f"lol noob, {image_name}, {w_end - w_start, h_end - h_start}"

        new_image_name = f"{start:05}.png"
        bicubic_image = cv2.resize(image_, (128, 128), cv2.INTER_CUBIC)

        cv2.imwrite(os.path.join(HR_FOLDER, new_image_name), image_)
        cv2.imwrite(os.path.join(LR_FOLDER, new_image_name), bicubic_image)

        start += 1
        del image_

    del raw_image

    print(f"Images Completed: {idx}/{length}", end="\r")
