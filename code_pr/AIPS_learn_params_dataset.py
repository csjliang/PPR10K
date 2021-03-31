import os
import random
import pickle
import glob
import numpy as np
import torch
import torch.utils.data as data
import util
from torchvision import transforms


class AIPS_param_Dataset(data.Dataset):

    def __init__(self, opt, phase):
        super(AIPS_param_Dataset, self).__init__()
        self.opt = opt
        self.phase = phase

        if  self.phase == 'train':
            self.img_paths = sorted(glob.glob(os.path.join(self.opt.data_root_train) + "/*.*"))
        if self.phase == 'val':
            self.img_paths = sorted(glob.glob(os.path.join(self.opt.data_root_val) + "/*.*"))
        self.xmp_dict = util.read_xmp(self.opt.xmp_pkl_path)
        ## statistics from imagenet
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __getitem__(self, index):
        img = util.read_img(self.img_paths[index % len(self.img_paths)])
        H, W, C = img.shape

        if self.phase == 'train':
            ## random crop
            rnd_h = random.randint(0, max(0, H - self.opt.img_size))
            rnd_w = random.randint(0, max(0, W - self.opt.img_size))
            img = img[rnd_h:rnd_h + self.opt.img_size, rnd_w:rnd_w + self.opt.img_size, :]
            ## random flip and rotation
            img = util.augment(img, hflip=True, rot=True)

        if self.opt.if_imagenet_normalization:
            for i in range(C):
                img[:, :, i] = (img[:, :, i] - self.mean[i]) / self.std[i]

        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()

        # img = Image.open(self.img_paths[index % len(self.img_paths)])
        # print(img)
        # img = self.transform(img)
        img_name = self.img_paths[index].split('/')[-1].split('.')[0]
        img_class = img_name.split('_')[0]
        # if self.phase == 'train':
        param_GT = self.xmp_dict[img_name]
        maxs = self.xmp_dict['max']
        mins = self.xmp_dict['min']

        return {'img': img, 'param_GT': param_GT, 'img_class': img_class, 'img_name': img_name, 'max': maxs,
                'min': mins}

        # if self.phase == 'train':
        #     return {'img': img, 'param_GT': param_GT, 'img_class': img_class, 'img_name': img_name, 'max': maxs, 'min': mins}
        # if self.phase == 'val':
        #     return {'img': img, 'img_class': img_class, 'img_name': img_name, 'max': maxs, 'min': mins}

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    pass