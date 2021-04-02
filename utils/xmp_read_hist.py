import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# path of the xmp files
xmp_root = ''
# path to save the hist
hist_root = ''

if not os.path.exists(hist_root):
    os.mkdir(hist_root)

i = 0

xmp_all_dict = {}

for xmp_name in os.listdir(xmp_root):

    f = open(os.path.join(xmp_root, xmp_name),'r', encoding='UTF-8')

    line = f.readline()

    xml_info_dict = {}
    xml_info_dict['xmp_name'] = xmp_name

    while line:
        line = f.readline()
        if 'crs' in line or 'tiff' in line:
            dict_info = line.split(':')[-1]
            if '=' in dict_info:
                key = dict_info.split('=')[0]
                value = dict_info.split('=')[1][1:-2]
                xml_info_dict[key] = value
    print(xmp_name, xml_info_dict)
    xmp_all_dict[i] = xml_info_dict
    f.close()
    i+=1

print(xmp_all_dict[50])

each_key = {}

for key, _ in xmp_all_dict[0].items():

    if key in ['Model', 'Exposure2012', 'Contrast2012', 'Highlights2012', 'Shadows2012', 'Whites2012', 'Blacks2012', 'Clarity2012', 'Dehaze', 'Tint', 'Saturation', 'Temperature']:
        value = []
        for i, dict in xmp_all_dict.items():
            try:
                if key != 'Model':
                    value.append(float(dict[key]))
                else:
                    value.append((dict[key]))
            except:
                print(dict['xmp_name'])
        each_key[key] = value


i = 0
for key, value in each_key.items():

    if key in ['Model', 'Exposure2012', 'Contrast2012', 'Highlights2012', 'Shadows2012', 'Whites2012', 'Blacks2012', 'Clarity2012', 'Dehaze', 'Tint', 'Saturation', 'Temperature']:

        print(key)
        print(value)
        if key == 'Temperature':
            for a in range(len(value)):
                if value[a] >= 10000:
                    value[a] = 10000

        plt.hist(value, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
        plt.title(key)
        plt.savefig(os.path.join(hist_root, key + '.png'))
        plt.show()