import glob
import random
import os
import numpy as np
import torch
import cv2

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision_x_functional as TF_x


class ImageDataset_paper(Dataset):
    def __init__(self, root, mode="train", use_mask=False):
        self.mode = mode
        self.root = root
        self.use_mask = use_mask
        self.retoucher = 'wanglei'
        print(self.retoucher)
        self.train_input_files = sorted(glob.glob(os.path.join(root, "train_renamed/source_540_aug" + "/*.tif")))
        self.train_target_files = sorted(glob.glob(os.path.join(root, "train_renamed/target_" + self.retoucher + "/*.tif")))
        self.train_mask_files = sorted(glob.glob(os.path.join(root, "train_renamed/masks_540" + "/*.png")))

        self.test_input_files = sorted(glob.glob(os.path.join(root, "val_renamed/source" + "/*.tif")))
        self.test_target_files = sorted(glob.glob(os.path.join(root, "val_renamed/target_" + self.retoucher + "/*.tif")))
        self.test_mask_files = sorted(glob.glob(os.path.join(root, "val_renamed/masks_val_540" + "/*.png")))

    def __getitem__(self, index):

        if self.mode == "train":
            img_name = os.path.split(self.train_input_files[index % len(self.train_input_files)])[-1]
            img_input = cv2.imread(self.train_input_files[index % len(self.train_input_files)],-1)
            if len(self.train_input_files) == len(self.train_target_files):
                img_exptC = Image.open(self.train_target_files[index % len(self.train_target_files)])
                if self.use_mask:
                    img_mask = Image.open(os.path.join(self.root, "train_renamed/masks_540/" + img_name[:-4] + ".png"))
            else:
                split_name = img_name.split('_')
                if len(split_name) == 2:
                    img_exptC = Image.open(os.path.join(self.root, "train_renamed/target_" + self.retoucher + '/' + img_name))
                    if self.use_mask:
                        img_mask = Image.open(os.path.join(self.root, "train_renamed/masks_540/" + img_name[:-4] + ".png"))
                else:
                    img_exptC = Image.open(
                        os.path.join(self.root, "train_renamed/target_" + self.retoucher + '/' + split_name[0] + "_" + split_name[1] + ".tif"))
                    if self.use_mask:
                        img_mask = Image.open(
                            os.path.join(self.root, "train_renamed/masks_540/" + split_name[0] + "_" + split_name[1] + ".png"))

        elif self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)],-1)
            img_exptC = Image.open(self.test_target_files[index % len(self.test_target_files)])
            if self.use_mask:
                img_mask = Image.open(self.test_mask_files[index % len(self.test_mask_files)])

        img_input = np.array(img_input)

        img_input = img_input[:, :, [2, 1, 0]]

        if self.mode == "train":

            ratio_H = np.random.uniform(0.6, 1.0)
            ratio_W = np.random.uniform(0.6, 1.0)
            W,H = img_exptC._size
            crop_h = round(H*ratio_H)
            crop_w = round(W*ratio_W)
            i, j, h, w = transforms.RandomCrop.get_params(img_exptC, output_size=(crop_h, crop_w))
            try:
                img_input = TF_x.resized_crop(img_input, i, j, h, w, (448,448))
            except:
                print(crop_h, crop_w, img_name)
            img_exptC = TF.resized_crop(img_exptC, i, j, h, w, (448,448))
            if self.use_mask:
                img_mask = TF.resized_crop(img_mask, i, j, h, w, (448,448))

            if np.random.random() > 0.5:
                img_input = TF_x.hflip(img_input)
                img_exptC = TF.hflip(img_exptC)
                if self.use_mask:
                    img_mask = TF.hflip(img_mask)

        img_input = TF_x.to_tensor(img_input)
        img_exptC = TF.to_tensor(img_exptC)
        if self.use_mask:
            img_mask = TF.to_tensor(img_mask)
        if self.use_mask:
            return {"A_input": img_input, "A_exptC": img_exptC, "input_name": img_name, "mask": img_mask}
        else:
            return {"A_input": img_input, "A_exptC": img_exptC, "input_name": img_name}

    def __len__(self):
        if self.mode == "train":
            return len(self.train_input_files)
            # return 1
        elif self.mode == "test":
            return len(self.test_input_files)
