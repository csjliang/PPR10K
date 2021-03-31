import os
import exifread
import shutil

root_dir = r'F:\AIPS\paper\RAW_renamed\images_tif_540\target_wanglei'
# target_path = r'F:\AIPS\paper\final_10081\train_val_files_newval\train\target_540_wanglei_'
#
# if not os.path.exists(target_path):
#     os.mkdir(target_path)

def getExif(file_path, file_name):
    FIELD = 'EXIF DateTimeOriginal'
    fd = open(file_path, 'rb')
    tags = exifread.process_file(fd)
    fd.close()
    if FIELD in tags:
        new_name = str(tags[FIELD]).replace(':', '').replace(' ', '_') + os.path.splitext(file_path)[1]
        tot = 1
        while os.path.exists(new_name):
            new_name = str(tags[FIELD]).replace(':', '').replace(' ', '_') + '_' + str(tot) + \
                       os.path.splitext(file_path)[1]
            tot += 1

        new_name2 = new_name.split(".")[0] + '__' + file_name.split('.')[0]
        # print(new_name2)
        # os.rename(filename, new_name2)
        return new_name2
    # else:
    #     print('No {} found'.format(FIELD))

# 7417
time_tags = []

for file_name in os.listdir(root_dir):
    file_path = os.path.join(root_dir, file_name)
    if os.path.isfile(file_path):
        time_tag = getExif(file_path, file_name)
        time_tags.append(time_tag)

print(time_tags)


#
# pairs = []
# i = 0
#
# for time_tag in time_tags:
#
#     for compare in time_tags:
#
#         pair = time_tag.split('__')[-1].split('_')[0] + '_' + compare.split('__')[-1].split('_')[0]
#
#         if time_tag.split('__')[0] == compare.split('__')[0] and \
#                 time_tag.split('__')[-1].split('_')[0] != compare.split('__')[-1].split('_')[0] and \
#                 pair not in pairs and time_tag.split('__')[0].split('_')[0] != '20000101':
#
#             pairs.append(pair)
#
#             print(time_tag, compare)
#             source_path_left = os.path.join(root_dir, time_tag.split('__')[-1]+'.jpg')
#             source_path_right = os.path.join(root_dir, compare.split('__')[-1] + '.jpg')
#             target_path_left = os.path.join(target_path, str(i) + '_' + time_tag.split('__')[-1] + '.jpg')
#             target_path_right = os.path.join(target_path, str(i) + '_' + compare.split('__')[-1] + '.jpg')
#             # shutil.copy(source_path_left, target_path_left)
#             # shutil.copy(source_path_right, target_path_right)
#             i += 1
#
# print(pairs)

# ['0_2', '10_15', '11_18', '12_8', '12_21', '13_8', '13_6', '14_7', '14_18', '14_21', '14_9', '15_10', '16_19', '17_20', '18_11', '18_21', '18_14', '19_16', '1_21', '1_3', '20_17', '21_1', '21_12', '21_18', '21_14', '2_0', '3_1', '4_6', '5_7', '6_4', '6_13', '7_5', '7_14', '8_13', '8_12', '9_14']

