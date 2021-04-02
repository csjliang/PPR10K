import os
import cv2
import numpy as np

root_dir = ''
target_dir_root = ''

if not os.path.exists(target_dir_root):
    os.mkdir(target_dir_root)

img_dirs_ = ['source',
             'nomask_noglc_a',
             'mask_noglc_a',
             'nomask_glc_a',
             'mask_glc_a',
             'target_a'
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
