import os
import cv2
import numpy as np

root_dir = r'D:\AIPS\result_newval_stacked'
target_dir_root = r'D:\AIPS\result_newval_stacked_GLC'
if not os.path.exists(target_dir_root):
    os.mkdir(target_dir_root)

# img_dirs_ = ['source', 'target_wanglei', 'target_zhangli', 'target_xiaobai',
#             '3dlut_wanglei_baseline', '3dlut_zhangli_baseline', '3dlut_xiaobai_baseline',
#             'new_lut_wanglei_nomaskcorrected_GLCcorrected_bz16_99', 'new_lut_zhangli_nomask_GLCcorrected_bz16_99', 'new_lut_xiaobai_nomask_GLCcorrected_bz16_99']
img_dirs_ = ['source',
             'new_lut_wanglei_nomask5_correctedmask_newval_rgb_newaug_54',
             'new_lut_wanglei_mask5_correctedmask_newval_rgb_newaug_155',#    wanglei_both_67  wanglei_GLC_150    ',
             'new_lut_wanglei_nomask5_GLCcorrected_bz16_newval_newaug_48',
             'new_lut_wanglei_mask5_GLCcorrected_bz16_newval_newaug_57',
             ]

img_dirs = []
target_dirs = []
for dir in img_dirs_:
    img_dir = os.path.join(root_dir, dir)
    target_dir = os.path.join(target_dir_root, dir)
    img_dirs.append(img_dir)
    target_dirs.append(target_dir)

files = os.listdir(img_dirs[1])

for file in files:
    k = 0
    for j in range(len(img_dirs_)):
        source_path = os.path.join(img_dirs[j], file)
        print(source_path)
        img = cv2.imread(source_path)
        if k == 0:
            img_stacked = img
        else:
            img_stacked = np.vstack((img_stacked, img))
        k += 1
    target_path = os.path.join(target_dir_root, file)
    cv2.imwrite(target_path, img_stacked)

    #
    #
    # source_path = os.path.join(root_dir, source_name)
    # img = cv2.imread(source_path)
    # print(source_path)
    # h, w, _ = img.shape
    # if h < w:
    #     img = cv2.resize(img, (540, 360))
    # else:
    #     img = cv2.resize(img, (360, 540))
    # img_stacked = img
    # for i in range(1, 4):
    #     source_name = file_name.split('_')[0] + '__' + str(i) + '_' + file_name.split('_')[1] + '.jpg'
    #     source_path = os.path.join(root_dir, source_name)
    #     img = cv2.imread(source_path)
    #     h_, w_, _ = img.shape
    #     if h_ != h:
    #         img = np.rot90(img)
    #     h__, w__, _ = img.shape
    #     if h__ < w__:
    #         img = cv2.resize(img, (540, 360))
    #     else:
    #         img = cv2.resize(img, (360, 540))
    #     img_stacked = np.hstack((img_stacked, img))
    # target_path = os.path.join(target_dir, file_name+'.jpg')
    # cv2.imwrite(target_path, img_stacked)
