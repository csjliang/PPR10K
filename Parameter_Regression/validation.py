from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import argparse
from AIPS_learn_params_dataset import AIPS_param_Dataset
from model import set_parameter_requires_grad, initialize_model
import util
import numpy as np
import glob
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default='resnet_bz16_xiaobai_relative', help="experiment name")
parser.add_argument("--gpu_ids", type=str, default='7', help="used gpus")
parser.add_argument("--load_mode", type=str, default='best', help="mode of loading models, [imagenet_pretrained | best | epoch]")
parser.add_argument("--num_classes", type=int, default=11, help="number of learned params")
parser.add_argument("--epoch", type=int, default=30, help="set to non-zero to select saved models")
parser.add_argument("--data_root_val", type=str, default="/data/vdb/liangjie/AIPS_data/PPR10Kdataset_newval_10081/new_selected_540", help="name of the dataset")
parser.add_argument("--save_root", type=str, default="/data/vdb/liangjie/AIPS_data/PPR10Kdataset_newval_10081/experiment_pr", help="save checkpoint root")
parser.add_argument("--xmp_ori_files_path", type=str, default="/data/vdb/liangjie/AIPS_data/PPR10Kdataset_newval_10081/new_selected_540_xmp/xmp_ori", help="xmp files for generating the predicted xmp")
parser.add_argument("--xmp_pkl_path", type=str, default="/data/vdb/liangjie/AIPS_data/PPR10Kdataset_newval_10081/xiaobai_params_norm_tmp.pkl", help="name of the dataset")
parser.add_argument("--img_size", type=int, default=224, help="resolution of the input image")
parser.add_argument("--model_name", type=str, default="resnet", help="selected model [from_raw | vgg16]")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--if_imagenet_normalization", default=False, help="")
parser.add_argument("--get_pred_xmp", default=True, help="")
opt = parser.parse_args()
print(opt)

feature_extract = False

device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
# device = torch.device('cpu')  # get device name: CPU or GPU
# print(device)

# Initialize the model for this run
model = initialize_model(opt.model_name, opt.num_classes, feature_extract, opt)

print(model)

if not os.path.exists(opt.save_root):
    os.mkdir(opt.save_root)

save_path = os.path.join(opt.save_root, opt.experiment_name)
if not os.path.exists(save_path):
    os.mkdir(save_path)
save_xmp_path = os.path.join(opt.save_root, opt.experiment_name + '/xmp_val')
os.makedirs(save_xmp_path, exist_ok=True)

print("Initializing Datasets and Dataloaders...")

image_dataset = AIPS_param_Dataset(opt, 'val')
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=False, num_workers=1)

# Send the model to GPU
model = model.to(device)
model.eval()

img_paths = sorted(glob.glob(opt.data_root_val + "/*.*"))
xmp_dict = util.read_xmp(opt.xmp_pkl_path)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
maxs = xmp_dict['max']
mins = xmp_dict['min']


for i in range(len(img_paths)):
    img = util.read_img(img_paths[i % len(img_paths)])
    H, W, C = img.shape

    if opt.if_imagenet_normalization:
        for c in range(C):
            img[:, :, c] = (img[:, :, c] - mean[c]) / std[c]

    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    img = img.unsqueeze(dim=0)

    img_name = img_paths[i].split('/')[-1].split('.')[0]
    # param_GT = xmp_dict[img_name]

    inputs = img.to(device)
    # labels = param_GT.to(device)
    with torch.set_grad_enabled(False):
        outputs = model(inputs)
    # outputs = model(inputs)

    # print(inputs, outputs)

    # reconstructed_pred = util.recons_params_val(outputs.squeeze())
    reconstructed_pred = util.recons_params_val(outputs.squeeze(), maxs, mins)
    # reconstructed_label = util.recons_params(labels.squeeze())
    # print('***************')
    # print('pred:  ', reconstructed_pred, img_name)
    # print('label: ', reconstructed_label, img_name)
    # print('***************')

    # get pred xmp of val
    if opt.get_pred_xmp:
        util.get_pred_xmp_relative_val(img_name, reconstructed_pred, opt.xmp_ori_files_path, save_xmp_path)
