import os
import cv2
import numpy as np

# root of different model results
root_dir = ''
# root to save the stacked images
target_dir_root = ''

if not os.path.exists(target_dir_root):
    os.mkdir(target_dir_root)

img_dirs = []
target_dirs = []
for dir in os.listdir(root_dir):
    img_dir = os.path.join(root_dir, dir)
    target_dir = os.path.join(target_dir_root, dir)
    img_dirs.append(img_dir)
    target_dirs.append(target_dir)

for dir in target_dirs:
    if not os.path.exists(dir):
        os.mkdir(dir)

files = os.listdir(img_dirs[1])

name_sets = {}
for i in range(1356, 1681):
    name_sets[str(i)] = []

for name in files:
    index = name.split('_')[0]
    if int(index) in range(1356, 1681):
        name_sets[index].append(name)

for i in range(1356, 1681):
    name_set = name_sets[str(i)]
    if name_set != []:
        for j in range(8):
            print(name_set)
            k = 0
            for name in name_set:
                source_path = os.path.join(img_dirs[j], name)
                # print(source_path)
                try:
                    img = cv2.imread(source_path)
                    h, w, _ = img.shape
                except:
                    img = cv2.imread(source_path.split('.')[0] + '.tif')
                    h, w, _ = img.shape
                if h > w:
                    img = cv2.resize(img, (360, 540))
                else:
                    img = cv2.resize(img, (810, 540))
                if k == 0:
                    img_stacked = img
                else:
                    img_stacked = np.hstack((img_stacked, img))
                k += 1
            target_path = os.path.join(target_dirs[j], str(i) + '.jpg')
            cv2.imwrite(target_path, img_stacked)
