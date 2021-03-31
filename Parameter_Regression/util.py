import os
import cv2
import numpy as np
import pickle
import random
import math
import torch

def get_mapping_modified_attributes(mapping_txt_path):
    name_value_dict = {}
    with open(mapping_txt_path, 'r') as f:
        for line in f:
            # print(line) 162_1_2+-467.66+5.28935+-22.033+0.11949+-24.506+-32.043+.xmp
            name = line.split('+')[0]
            values_this_file = []
            for i in range(6):
                value = float(line.split('+')[i+1])
                values_this_file.append(value)
            name_value_dict[name] = values_this_file
    return name_value_dict

def get_gt_score_glc(values_1, values_2):

    ranges = [1000, 20, 30, 2, 30, 70] # temp, tint, sat, exp, con, hig
    score = 0
    for i in range(len(ranges)):
        score += np.abs(values_1[i] - values_2[i]) / ranges[i]
    normed_score = 1 - score / len(ranges)
    return normed_score

def get_filename_dict(data_dir):
    paths = os.listdir(data_dir)
    neme_dict = {}
    for i in range(1520):
        neme_dict[str(i)] = []
    for path in paths:
        class_index = path.split('/')[-1].split('_')[0]
        file_index = path.split('/')[-1].split('.')[0]
        neme_dict[class_index].append(file_index)
    return neme_dict


def read_img(path):
    """read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]"""
    img = cv2.imread(path, -1)
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.
    else:
        img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img

def save_img(path, image):
    cv2.imwrite(path, image.astype(np.uint16))

def read_xmp(xmp_path):

    with open(xmp_path, 'rb') as f:
        read_xmp_dict = pickle.load(f)

    return read_xmp_dict


def recons_params(pred):

    # zhangli
    # maxs = [28000.0, 63.0, 38.0, 2.9, 67.0, 100.0, 93.0, 53.0, 100.0, 28.0, 36.0]
    # mins = [2300.0, -96.0, -19.0, -2.9, -52.0, -100.0, -80.0, -51.0, -48.0, -14.0, -8.0]

    # wanglei
    # maxs = [50000.0, 98.0, 15.0, 3.75, 100.0, 30.0, 100.0, 18.0, 52.0, 15.0, 13.0]
    # mins = [2250.0, -49.0, -28.0, -2.2, -18.0, -100.0, -17.0, -87.0, -99.0, 0.0, 0.0]

    # xiaobai
    maxs = [15750.0, 56.0, 15.0, 3.95, 69.0, 14.0, 80.0, 51.0, 38.0, 0.0, 21.0]
    mins = [2000.0, -56.0, -38.0, -2.65, -45.0, -100.0, -56.0, -100.0, -51.0, 0.0, 0.0]

    assert len(maxs) == len(pred)

    reconstructed_pred = []

    for i in range(len(pred)):
        reconstructed_pred.append(pred[i].cpu().detach().numpy() * (maxs[i] - mins[i]) + mins[i])

    return np.around(reconstructed_pred, decimals=2)


def recons_params_val(pred, maxs, mins):

    assert len(maxs) == len(pred)

    reconstructed_pred = []

    for i in range(len(pred)):
        reconstructed_pred.append(pred[i] * (maxs[i] - mins[i]) + mins[i])

    return reconstructed_pred

def augment(img, hflip=True, rot=True):
    """horizontal flip OR rotate (0, 90, 180, 270 degrees)"""
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    if hflip:
        img = img[:, ::-1, :]
    if vflip:
        img = img[::-1, :, :]
    if rot90:
        img = img.transpose(1, 0, 2)

    return img

def get_pred_xmp_val(name, pred_param, xmp_file_root, pred_xmp_root):

    modified_attributes = ['WhiteBalance', 'Temperature', 'Tint', 'Saturation', 'Exposure2012', 'Contrast2012', 'Highlights2012', 'Shadows2012', 'Whites2012', 'Blacks2012', 'Clarity2012', 'Dehaze']

    pred_dict = {}
    for i in range(len(modified_attributes) - 1):
        pred_dict[modified_attributes[i + 1]] = pred_param[i]

    xmp_path = os.path.join(xmp_file_root, name + '.xmp')
    pred_xmp_path = os.path.join(pred_xmp_root, name + '.xmp')
    if not os.path.exists(pred_xmp_root):
        os.mkdir(pred_xmp_root)

    with open(xmp_path, 'r', encoding='UTF-8') as f_ori, open(pred_xmp_path, "w", encoding="utf-8") as f_pred:
        for line in f_ori:
            done = False
            for attribute in modified_attributes:
                if line.split('=')[0].split(':')[-1] == attribute and 'crs' in line.split(':')[0]:
                    if not done:
                        value_ori = line.split('=')[1][1:-2]
                        value_pred = ''
                        if attribute == 'WhiteBalance':
                            value_pred = 'Custom'
                        elif attribute == 'Exposure2012':
                            value_pred = '{:+.2f}'.format(pred_dict[attribute])
                        elif attribute == 'Temperature':
                            value_pred = str(int(pred_dict[attribute]))
                        else:
                            if int(pred_dict[attribute]) > 0:
                                value_pred = '+' + str(int(pred_dict[attribute]))
                            elif int(pred_dict[attribute]) < 0:
                                value_pred = str(int(pred_dict[attribute]))
                            else:
                                value_pred = str(0)
                        line_ = line
                        line_pred = line_.split('=')[0] + '=' + line_.split('=')[-1].replace(value_ori, value_pred)
                        f_pred.write(line_pred)
                        done = True
            if not done:
                f_pred.write(line)

def get_pred_xmp_relative_val(name, pred_param, xmp_file_root, pred_xmp_root):

    modified_attributes = ['WhiteBalance', 'Temperature', 'Tint', 'Saturation', 'Exposure2012', 'Contrast2012', 'Highlights2012', 'Shadows2012', 'Whites2012', 'Blacks2012', 'Clarity2012', 'Dehaze']

    pred_dict = {}
    for i in range(len(modified_attributes)-1):
        pred_dict[modified_attributes[i+1]] = pred_param[i]

    xmp_path = os.path.join(xmp_file_root, name + '.xmp')
    pred_xmp_path = os.path.join(pred_xmp_root, name + '.xmp')
    if not os.path.exists(pred_xmp_root):
        os.mkdir(pred_xmp_root)

    with open(xmp_path, 'r', encoding='UTF-8') as f_ori, open(pred_xmp_path, "w", encoding="utf-8") as f_pred:
        for line in f_ori:
            done = False
            for attribute in modified_attributes:
                if line.split('=')[0].split(':')[-1] == attribute and 'crs' in line.split(':')[0]:
                    if not done:
                        value_ori = line.split('=')[1][1:-2]
                        value_pred = ''
                        if attribute == 'WhiteBalance':
                            value_pred = 'Custom'
                        elif attribute == 'Exposure2012':
                            value_pred = '{:+.2f}'.format(pred_dict[attribute])
                        elif attribute == 'Temperature':
                            # print(int(pred_dict[attribute]), int(value_ori))
                            value_pred = str(int(pred_dict[attribute]) + int(value_ori))
                        elif attribute == 'Tint':
                            if int(pred_dict[attribute]) + int(value_ori) > 0:
                                value_pred = '+' + str(int(pred_dict[attribute]) + int(value_ori))
                            elif int(pred_dict[attribute]) + int(value_ori) < 0:
                                value_pred = str(int(pred_dict[attribute]) + int(value_ori))
                            else:
                                value_pred = str(0)
                        else:
                            if int(pred_dict[attribute]) > 0:
                                value_pred = '+' + str(int(pred_dict[attribute]))
                            elif int(pred_dict[attribute]) < 0:
                                value_pred = str(int(pred_dict[attribute]))
                            else:
                                value_pred = str(0)
                        line_ = line
                        line_pred = line_.split('=')[0] + '=' + line_.split('=')[-1].replace(value_ori, value_pred)
                        f_pred.write(line_pred)
                        done = True
            if not done:
                f_pred.write(line)


if __name__ == '__main__':

    pass