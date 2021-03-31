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

        self.test_input_files = sorted(glob.glob("/data1/liangjie/AIPS_data_valsource_full" + "/*.tif"))

    def __getitem__(self, index):

        img_name = os.path.split(self.test_input_files[index % len(self.test_input_files)])[-1]
        img_input = cv2.imread(self.test_input_files[index % len(self.test_input_files)], -1)

        img_input = np.array(img_input)

        img_input = img_input[:, :, [2, 1, 0]]

        img_input = TF_x.to_tensor(img_input)

        return {"A_input": img_input, "input_name": img_name}

    def __len__(self):
        return len(self.test_input_files)
