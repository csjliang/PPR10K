import os
import numpy as np
import random

def get_xmp_augment(name_ori, name_aug, augment_param, xmp_file_root, pred_xmp_root):

    modified_attributes = ['Temperature', 'Tint', 'Saturation', 'Vibrance', 'Exposure2012', 'Contrast2012', 'Highlights2012', 'Shadows2012', 'Whites2012', 'Blacks2012', 'Clarity2012', 'Dehaze']

    augment_dict = {}
    for i in range(len(modified_attributes)):
        augment_dict[modified_attributes[i]] = augment_param[i]

    xmp_path = os.path.join(xmp_file_root, name_ori + '.xmp')
    pred_xmp_path = os.path.join(pred_xmp_root, name_aug + '.xmp')
    if not os.path.exists(pred_xmp_root):
        os.mkdir(pred_xmp_root)

    with open(xmp_path, 'r', encoding='UTF-8') as f_ori, open(pred_xmp_path, "w", encoding="utf-8") as f_pred:
        for line in f_ori:
            done = False
            for attribute in modified_attributes:
                if line.split('=')[0].split(':')[-1] == attribute and 'crs' in line.split(':')[0]:
                    if not done:
                        value_ori = line.split('=')[1][1:-2]
                        if attribute == 'Exposure2012':
                            value_pred = '{:+.2f}'.format(float(augment_dict[attribute]))
                        elif attribute == 'Temperature':
                            value_pred = str(int(augment_dict[attribute]))
                        else:
                            if int(augment_dict[attribute]) > 0:
                                value_pred = '+' + str(int(augment_dict[attribute]))
                            elif int(augment_dict[attribute]) < 0:
                                value_pred = str(int(augment_dict[attribute]))
                            else:
                                value_pred = str(0)
                        line_ = line
                        line_pred = line_.split('=')[0] + '=' + line_.split('=')[-1].replace(value_ori, value_pred)
                        f_pred.write(line_pred)
                        done = True
            if not done:
                f_pred.write(line)

def read_xmp_and_statis(xmp_root):

    xmp_all_dict = {}

    for xmp_name in os.listdir(xmp_root):

        name = xmp_name.split('.')[0]

        f = open(os.path.join(xmp_root, xmp_name), 'r', encoding='UTF-8')

        line = f.readline()

        xml_info_dict = {}

        while line:
            line = f.readline()
            if 'crs' in line:
                dict_info = line.split(':')[-1]
                if '=' in dict_info:
                    key = dict_info.split('=')[0]
                    value = dict_info.split('=')[1][1:-2]
                    xml_info_dict[key] = value
        xmp_all_dict[name] = xml_info_dict
        f.close()

    each_key = {}

    for key in ['Exposure2012', 'Contrast2012', 'Highlights2012', 'Shadows2012', 'Whites2012', 'Blacks2012',
               'Clarity2012', 'Dehaze', 'Tint', 'Saturation', 'Temperature', 'Vibrance']:
        value = []

        for name, dict in xmp_all_dict.items():
            value.append(float(dict[key]))
        each_key[key] = value
    return xmp_all_dict, each_key

def get_GT_attribute_statistics(num_steps, each_key):

    statics_all_aug_attributes = {}

    min_max_each_attribute = {}

    for key in ['Exposure2012', 'Contrast2012', 'Highlights2012', 'Shadows2012', 'Whites2012', 'Blacks2012',
                'Clarity2012', 'Dehaze', 'Tint', 'Saturation', 'Temperature', 'Vibrance']:
        min_max_each_attribute[key + '_min'] = min(each_key[key])
        min_max_each_attribute[key + '_max'] = max(each_key[key])

    for key in ['Temperature', 'Tint', 'Exposure2012', 'Highlights2012']:

        step = (max(each_key[key]) - min(each_key[key])) / num_steps

        value = min(each_key[key])
        statics = {}
        for i in range(num_steps):
            ranges = '{:.2f}'.format(value) + '*' + '{:.2f}'.format(value+step)
            count = 0
            for j in range(len(each_key[key])):
                if each_key[key][j] >= value and each_key[key][j] < (value + step):
                    count += 1
            statics[ranges] = count + 1 if i == num_steps - 1 else count
            value += step

        statics_all_aug_attributes[key] = statics

    return statics_all_aug_attributes, min_max_each_attribute

def get_augment_params_with_names(num_files_aug_all, xmp_train_dict, xmp_file_root, aug_xmp_root):

    num_already = len(os.listdir(xmp_file_root))

    num_aug_each_image = int((num_files_aug_all - num_already) / num_already)

    for name, dict in xmp_train_dict.items():

        modified_attributes = ['Temperature', 'Tint', 'Saturation', 'Vibrance', 'Exposure2012', 'Contrast2012', 'Highlights2012',
                               'Shadows2012', 'Whites2012', 'Blacks2012', 'Clarity2012', 'Dehaze']

        GT_values = []
        for attribute in modified_attributes:
            GT_values.append(dict[attribute])

        auging_attributes = ['Temperature', 'Tint', 'Exposure2012', 'Highlights2012', 'Contrast2012', 'Saturation']

        random_values_all_attribute = np.zeros([len(modified_attributes), num_aug_each_image])

        for i in range(len(modified_attributes)):

            attribute = modified_attributes[i]

            if attribute in auging_attributes:

                if attribute == 'Exposure2012':
                    start = -1
                    end = 1
                if attribute == 'Tint':
                    start = -10
                    end = 10
                if attribute == 'Highlights2012':
                    start = -35
                    end = 35
                if attribute == 'Temperature':
                    start = -500
                    end = 500
                if attribute == 'Contrast2012':
                    start = -30
                    end = 0
                if attribute == 'Saturation':
                    start = 30
                    end = 0

                random_values_all_attribute[i, :] = np.random.uniform(start, end, size=num_aug_each_image)

        for i in range(num_aug_each_image):

            auged_values = GT_values.copy()

            name_aug = name + '_' + str(i+1)

            for j in range(len(modified_attributes)):

                if modified_attributes[j] in auging_attributes:

                    auged_values[j] = random_values_all_attribute[j, i] + float(auged_values[j])

            # print(name_aug, auged_values)

            get_xmp_augment(name, name_aug, auged_values, xmp_file_root, aug_xmp_root)

if __name__ == '__main__':

    # path of the xmp_source
    xmp_file_root = ''
    # path to save the augmented xmps
    aug_xmp_root = ''

    aug_times = 5

    num_train = len(os.listdir(xmp_file_root))

    num_all_files_after_augment = num_train * aug_times

    xmp_train_dict, _ = read_xmp_and_statis(xmp_file_root)

    augment_params_all = get_augment_params_with_names(num_all_files_after_augment, xmp_train_dict, xmp_file_root, aug_xmp_root)