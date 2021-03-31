
import os
import matplotlib.pyplot as plt
import numpy as np

root_dir = r'F:\AIPS\paper\RAW_renamed\xmp_wanglei_train'
hist_dir = r'F:\AIPS\paper\RAW_renamed'

name = 'number_imgs_each_scene'

if not os.path.exists(hist_dir):
    os.mkdir(hist_dir)

file_names = os.listdir(root_dir)

scene_indexes = np.arange(0, 1681)
numbers = []

for index in scene_indexes:
    number = 0
    for file in file_names:
        if file.split('_')[0] == str(index):
            number += 1
    numbers.append(number)

plt.hist(numbers, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
plt.title(name)
plt.savefig(os.path.join(hist_dir, name + '.png'))
plt.show()