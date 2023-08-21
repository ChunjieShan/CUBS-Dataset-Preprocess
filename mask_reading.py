import shutil

import cv2 as cv
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


class DataReader:
    def __init__(self, root, save_root=None, split_ratio=0.8):
        self.root = root
        self.img_root = os.path.join(root, "IMAGES")
        self.anno_root = os.path.join(root, "SEGMENTATIONS/Manual-A1")
        self.split_ratio = split_ratio

        if save_root is None:
            save_root = "masks"

        self.save_root = os.path.join(root, save_root)

        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)

    def _split_dataset(self):
        images_list = os.listdir(self.img_root)
        random.shuffle(images_list)
        total_samples_num = len(images_list)
        training_num = int(total_samples_num * self.split_ratio)

        training_list = images_list[:training_num]
        val_list = images_list[training_num:]

        if not os.path.exists(os.path.join(self.img_root, "train")):
            os.mkdir(os.path.join(self.img_root, "train"))

        if not os.path.exists(os.path.join(self.img_root, "valid")):
            os.mkdir(os.path.join(self.img_root, "valid"))

        for file_name in training_list:
            shutil.copyfile(os.path.join(self.img_root, file_name),
                            os.path.join(self.img_root, "train", file_name))

        for file_name in val_list:
            shutil.copyfile(os.path.join(self.img_root, file_name),
                            os.path.join(self.img_root, "valid", file_name))

        return training_list, val_list

    def call_generate_mask(self):
        files_list = os.listdir(self.img_root)
        pbar = tqdm(files_list)

        for file_name in pbar:
            mask = self.generate_single_mask(file_name)
            img_name = file_name.split(".tiff")[0]
            cv.imwrite(os.path.join(self.save_root, img_name + ".png"), mask)

    def generate_single_mask(self, img_path):
        img = self.read_single_image(img_path)
        img_wh = img.shape[:2]
        ma_coords, li_coords = self.read_single_mask(img_path)
        li_coords = li_coords[::-1]

        mask = np.zeros(img_wh, dtype=np.uint8)

        cv.fillPoly(mask, [np.concatenate([ma_coords, li_coords])], color=1)

        return mask

    def read_single_image(self, img_path):
        curr_img_path = os.path.join(self.img_root, img_path)
        img = cv.imread(curr_img_path, cv.IMREAD_UNCHANGED)

        return img

    def read_single_mask(self, img_path):
        img = self.read_single_image(img_path)
        img_name = img_path.split(".tiff")[0]
        curr_li_mask_path = os.path.join(self.anno_root, img_name + "-LI.txt")
        curr_ma_mask_path = os.path.join(self.anno_root, img_name + "-MA.txt")

        ma_coords = []
        li_coords = []

        with open(curr_ma_mask_path, 'r') as f:
            _ma_coords = f.readlines()

        for coord in _ma_coords:
            coord = coord.strip()
            x, y = coord.split(' ')
            ma_coords.append([int(float(x)), int(float(y))])

            cv.circle(img, [int(float(x)), int(float(y))], 2, (0, 255, 0))

        with open(curr_li_mask_path, 'r') as f:
            _li_coords = f.readlines()

        for coord in _li_coords:
            coord = coord.strip()
            x, y = coord.split(' ')
            li_coords.append([int(float(x)), int(float(y))])

            cv.circle(img, [int(float(x)), int(float(y))], 2, (255, 0, 0))

        return ma_coords, li_coords


if __name__ == '__main__':
    data_root = "/mnt/h/Dataset/2.Carotid-Artery/DATASET_CUBS_tech"
    reader = DataReader(data_root)
    reader._split_dataset()
    # reader.call_generate_mask()
    # reader.read_single_mask("clin_0001_L.tiff")
