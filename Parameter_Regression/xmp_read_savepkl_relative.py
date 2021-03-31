import os
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt

def read_xmps_get_pkl(xmp_root_target, xmp_root_source, max_temp, min_temp, save_xmp_pkl_path, save_xmp_hist_path):

    # attributes = ['Exposure2012', 'Contrast2012', 'Highlights2012', 'Shadows2012', 'Whites2012', 'Blacks2012', 'Clarity2012', 'Dehaze', 'Tint', 'Saturation', 'Temperature']
    attributes = ['Temperature', 'Tint', 'Saturation', 'Exposure2012', 'Contrast2012', 'Highlights2012', 'Shadows2012', 'Whites2012', 'Blacks2012', 'Clarity2012', 'Dehaze', ]

    num_attributes = len(attributes)
    num_image = len(os.listdir(xmp_root_target))

    xmp_all_dict_target = {}
    for xmp_name in os.listdir(xmp_root_target):
        f = open(os.path.join(xmp_root_target, xmp_name),'r', encoding='UTF-8')
        line = f.readline()
        # xml_info_dict = {}
        values = {}
        values_ordered = []
        while line:
            line = f.readline()
            if 'crs' in line:
                dict_info = line.split(':')[-1]
                if '=' in dict_info:
                    key = dict_info.split('=')[0]
                    value = dict_info.split('=')[1][1:-2]
                    if key in attributes:
                        values[key] = value
                    # xml_info_dict[key] = value
        for key in attributes:
            values_ordered.append(float(values[key]))
        xmp_all_dict_target[xmp_name.split('.')[0]] = values_ordered
        f.close()

    xmp_all_dict_source = {}
    for xmp_name in os.listdir(xmp_root_source):
        f = open(os.path.join(xmp_root_source, xmp_name), 'r', encoding='UTF-8')
        line = f.readline()
        values = {}
        values_ordered = []
        while line:
            line = f.readline()
            if 'crs' in line:
                dict_info = line.split(':')[-1]
                if '=' in dict_info:
                    key = dict_info.split('=')[0]
                    value = dict_info.split('=')[1][1:-2]
                    if key in attributes:
                        values[key] = value
        for key in attributes:
            values_ordered.append(float(values[key]))
        xmp_all_dict_source[xmp_name.split('.')[0]] = values_ordered
        f.close()

    all_values = np.zeros([num_image, num_attributes])
    i = 0
    for key, value_target in xmp_all_dict_target.items():
        value_source = xmp_all_dict_source[key]
        print(value_source, value_target, key)
        for j in range(num_attributes):
            if value_target[j] - value_source[j] < min_temp:
                all_values[i, j] = min_temp
            elif value_target[j] - value_source[j] > max_temp:
                all_values[i, j] = max_temp
            else:
                all_values[i, j] = value_target[j] - value_source[j]
        i += 1
    maxs = np.zeros(num_attributes)
    mins = np.zeros(num_attributes)
    for i in range(num_attributes):
        maxs[i] = all_values[:, i].max()
        mins[i] = all_values[:, i].min()

    print(maxs, mins)

    dict_all = {}
    for key, value_target in xmp_all_dict_target.items():
        value_source = xmp_all_dict_source[key]
        norm_value = torch.zeros(num_attributes)
        for i in range(num_attributes):
            # if value_target[i] - value_source[i] < min_temp:
            #     print(key)
            if maxs[i] - mins[i] == 0:
                norm_value[i] = 0
            else:
                norm_value[i] = (value_target[i] - value_source[i] - mins[i]) / (maxs[i] - mins[i])
                if norm_value[i] > 1:
                    norm_value[i] = 1
                if norm_value[i] < -1:
                    norm_value[i] = -1
        dict_all[key] = norm_value

    dict_all['max'] = maxs
    dict_all['min'] = mins

    print(dict_all)

    # with open(save_xmp_pkl_path, 'wb') as f:
    #     pickle.dump(dict_all, f, pickle.HIGHEST_PROTOCOL)

    for i in range(11):

            plt.hist(all_values[:, i], bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
            plt.title(attributes[i])
            plt.savefig(os.path.join(save_xmp_hist_path, attributes[i] + '.png'))
            plt.show()

if __name__ == '__main__':

    xmp_root_target = r'F:\zenghui\wedding_train_raw_20210109\final_xmp'
    xmp_root_source = r'F:\zenghui\wedding_train_raw_20210109\default_xmp'
    save_xmp_pkl_path = r'F:\zenghui\wedding_train_raw_20210109\save.pkl'
    save_xmp_hist_path = r'F:\zenghui\wedding_train_raw_20210109\hist'
    os.makedirs(save_xmp_hist_path, exist_ok=True)
    max_temp, min_temp = 10000, -10000
    read_xmps_get_pkl(xmp_root_target, xmp_root_source, max_temp, min_temp, save_xmp_pkl_path, save_xmp_hist_path)
