import numpy as np
import torch
import torch.utils.data as data
import os
from random import choice
import random
import cv2
import glob

class ImageDataset_paper(data.Dataset):
    def __init__(self, root, mode="train", use_mask=False):
        super(ImageDataset_paper, self).__init__()

        self.mode = mode
        self.root = root
        self.retoucher = 'xiaobai'
        print(self.retoucher)

        self.dataroot_LQ_train = os.path.join(root, "train_renamed/source_540_aug")
        self.dataroot_GT_train = os.path.join(root, "train_renamed/target_" + self.retoucher)
        self.dataroot_LQ_val = os.path.join(root, "val_renamed/source")
        self.dataroot_GT_val = os.path.join(root, "val_renamed/target_" + self.retoucher)
        self.dataroot_mask = os.path.join(root, "train_renamed/masks_540")
        # self.dataroot_mask_val = os.path.join(root, "val_renamed/masks_val_540")
        self.use_mask = use_mask

        self.train_input_files = sorted(glob.glob(os.path.join(self.dataroot_LQ_train + "/*.tif")))
        self.train_target_files = sorted(glob.glob(os.path.join(self.dataroot_GT_train + "/*.tif")))
        self.train_mask_files = sorted(glob.glob(os.path.join(self.dataroot_mask + "/*.png")))

        self.test_input_files = sorted(glob.glob(os.path.join(self.dataroot_LQ_val + "/*.tif")))
        self.test_target_files = sorted(glob.glob(os.path.join(self.dataroot_GT_val + "/*.tif")))
        # self.test_mask_files = sorted(glob.glob(os.path.join(self.dataroot_mask_val + "/*.png")))

        self.path_LQ_dict = {}
        for path in self.train_target_files:
            self.path_LQ_dict[path.split('/')[-1].split('.')[0]] = []
        for paths in self.train_input_files:
            file_index = paths.split('/')[-1].split('_')[0] + '_' + paths.split('/')[-1].split('.')[0].split('_')[1]
            file = paths.split('/')[-1]
            self.path_LQ_dict[file_index].append(file)

        assert self.train_target_files, 'Error: GT path is empty.'

    def __getitem__(self, index):

        if self.mode == 'train':

            img_name = os.path.split(self.train_input_files[index % len(self.train_input_files)])[-1]
            GT_name_norear = img_name.split('_')[0] + '_' + img_name.split('.')[0].split('_')[1]
            cand_list = self.path_LQ_dict[GT_name_norear]
            selected_for_cropping = choice(cand_list)
            img_input_2_path = os.path.join(self.dataroot_LQ_train, selected_for_cropping)

            img_input_1 = read_img(self.train_input_files[index % len(self.train_input_files)])
            img_input_2 = read_img(img_input_2_path)

            if len(self.train_input_files) == len(self.train_target_files):
                img_exptC = read_img(self.train_target_files[index % len(self.train_target_files)])
                if self.use_mask:
                    img_mask = read_img(os.path.join(self.dataroot_mask, img_name[:-4] + ".png"))
            else:
                split_name = img_name.split('_')
                if len(split_name) == 2:
                    img_exptC = read_img(os.path.join(self.dataroot_GT_train, img_name))
                    if self.use_mask:
                        img_mask = read_img(os.path.join(self.dataroot_mask, img_name[:-4] + ".png"))
                else:
                    img_exptC = read_img(
                        os.path.join(self.dataroot_GT_train, split_name[0] + "_" + split_name[1] + ".tif"))
                    if self.use_mask:
                        img_mask = read_img(
                            os.path.join(self.dataroot_mask, split_name[0] + "_" + split_name[1] + ".png"))

        if self.mode == "test":
            img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
            img_input_1 = read_img(self.test_input_files[index % len(self.test_input_files)])
            img_exptC = read_img(self.test_target_files[index % len(self.test_target_files)])

        GT_size = 448
        img_input_1 = cv2.resize(img_input_1, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
        img_exptC = cv2.resize(img_exptC, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
        if self.mode == 'train':
            img_input_2 = cv2.resize(img_input_2, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
            if self.use_mask:
                img_mask = cv2.resize(img_mask, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
                img_mask_c3 = np.zeros_like(img_exptC)
                img_mask_c3[:, :, 0] = img_mask
                img_mask_c3[:, :, 1] = img_mask
                img_mask_c3[:, :, 2] = img_mask

        # # resize, crop and augmentation
        if self.mode == 'train':

            H, W, C = img_input_1.shape
            cropped_size = 336

            # randomly crop
            rnd_h_1 = random.randint(0, max(0, H - cropped_size))
            rnd_w_1 = random.randint(0, max(0, W - cropped_size))
            img_input_1 = img_input_1[rnd_h_1:rnd_h_1 + cropped_size, rnd_w_1:rnd_w_1 + cropped_size, :]
            img_GT_1 = img_exptC[rnd_h_1:rnd_h_1 + cropped_size, rnd_w_1:rnd_w_1 + cropped_size, :]
            if self.use_mask:
                img_mask_1 = img_mask_c3[rnd_h_1:rnd_h_1 + cropped_size, rnd_w_1:rnd_w_1 + cropped_size, :]

            rnd_h_2 = random.randint(0, max(0, H - cropped_size))
            rnd_w_2 = random.randint(0, max(0, W - cropped_size))
            img_input_2 = img_input_2[rnd_h_2:rnd_h_2 + cropped_size, rnd_w_2:rnd_w_2 + cropped_size, :]
            img_GT_2 = img_exptC[rnd_h_2:rnd_h_2 + cropped_size, rnd_w_2:rnd_w_2 + cropped_size, :]
            if self.use_mask:
                img_mask_2 = img_mask_c3[rnd_h_1:rnd_h_1 + cropped_size, rnd_w_1:rnd_w_1 + cropped_size, :]

            # calculate coincident region index
            if rnd_h_1 == rnd_h_2:
                h_coi_left = rnd_h_1
                h_coi_right = rnd_h_1 + cropped_size
            else:
                h_coi_left = max(rnd_h_1, rnd_h_2)
                h_coi_right = min(rnd_h_1, rnd_h_2) + cropped_size
            if rnd_w_1 == rnd_w_2:
                w_coi_left = rnd_w_1
                w_coi_right = rnd_w_1 + cropped_size
            else:
                w_coi_left = max(rnd_w_1, rnd_w_2)
                w_coi_right = min(rnd_w_1, rnd_w_2) + cropped_size

            h_coi_left_LQ = h_coi_left - rnd_h_1
            h_coi_right_LQ = h_coi_right - rnd_h_1
            w_coi_left_LQ = w_coi_left - rnd_w_1
            w_coi_right_LQ = w_coi_right - rnd_w_1

            h_coi_left_cropping = h_coi_left - rnd_h_2
            h_coi_right_cropping = h_coi_right - rnd_h_2
            w_coi_left_cropping = w_coi_left - rnd_w_2
            w_coi_right_cropping = w_coi_right - rnd_w_2

            coi_index_1 = [h_coi_left_LQ, h_coi_right_LQ, w_coi_left_LQ, w_coi_right_LQ]
            coi_index_2 = [h_coi_left_cropping, h_coi_right_cropping, w_coi_left_cropping, w_coi_right_cropping]

        img_input_1 = channel_convert(img_input_1.shape[2], 'RGB', [img_input_1])[0]
        if self.mode == 'train':
            img_GT_1 = channel_convert(img_GT_1.shape[2], 'RGB', [img_GT_1])[0]
        if self.mode == 'test':
            img_GT_1 = channel_convert(img_exptC.shape[2], 'RGB', [img_exptC])[0]
        if self.use_mask and self.mode == 'train':
            img_mask_1 = channel_convert(img_mask_1.shape[2], 'RGB', [img_mask_1])[0]
        if self.mode == 'train':
            img_input_2 = channel_convert(img_input_2.shape[2], 'RGB', [img_input_2])[0]
            img_GT_2 = channel_convert(img_GT_2.shape[2], 'RGB', [img_GT_2])[0]
            if self.use_mask:
                img_mask_2 = channel_convert(img_mask_2.shape[2], 'RGB', [img_mask_2])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_input_1.shape[2] == 3:
            img_input_1 = img_input_1[:, :, [2, 1, 0]]
            img_GT_1 = img_GT_1[:, :, [2, 1, 0]]
            if self.use_mask and self.mode == 'train':
                img_mask_1 = img_mask_1[:, :, [2, 1, 0]]
            if self.mode == 'train':
                img_input_2 = img_input_2[:, :, [2, 1, 0]]
                img_GT_2 = img_GT_2[:, :, [2, 1, 0]]
                if self.use_mask:
                    img_mask_2 = img_mask_2[:, :, [2, 1, 0]]

        img_input_1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_input_1, (2, 0, 1)))).float()
        img_GT_1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT_1, (2, 0, 1)))).float()
        if self.use_mask and self.mode == 'train':
            img_mask_1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_mask_1, (2, 0, 1)))).float()
        if self.mode == 'train':
            img_input_2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_input_2, (2, 0, 1)))).float()
            img_GT_2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT_2, (2, 0, 1)))).float()
            if self.use_mask:
                img_mask_2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_mask_2, (2, 0, 1)))).float()

        if self.mode == "train":
            if self.use_mask:
                return {"A_input_1": img_input_1, "A_exptC_1": img_GT_1, "mask_1": img_mask_1, "input_name": img_name,
                        "A_input_2": img_input_2, "A_exptC_2": img_GT_2, "mask_2": img_mask_2, "coi_index_1": coi_index_1, "coi_index_2": coi_index_2}
            else:
                return {"A_input_1": img_input_1, "A_exptC_1": img_GT_1, "input_name": img_name,
                        "A_input_2": img_input_2, "A_exptC_2": img_GT_2, "coi_index_1": coi_index_1, "coi_index_2": coi_index_2}
        else:
            return {"A_input": img_input_1, "A_exptC": img_GT_1, "input_name": img_name}

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_input_files)
        if self.mode == 'test':
            return len(self.test_input_files)

def read_img(path):
    '''read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]'''
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

def channel_convert(in_c, tar_type, img_list):
    """conversion among BGR, gray and y"""
    if in_c == 3 and tar_type == 'gray':  # BGR to gray
        gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in gray_list]
    elif in_c == 1 and tar_type == 'RGB':  # gray/y to BGR
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
    else:
        return img_list