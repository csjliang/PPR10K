import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

xmp_root = r'F:\AIPS\paper\RAW_renamed\xmp_zhangli_train'
hist_root = r'F:\AIPS\paper\RAW_renamed\xmp_zhangli_hist'

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
    else:
        pass
        # value = []
        # for i, dict in xmp_all_dict.items():
        #     value.append(dict[key])
        # each_key[key] = value


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
#
# models = each_key['Model']
# print(models)
# dict = {}
# for key in models:
#     dict[key] = dict.get(key, 0) + 1
# print(dict)
#
# for key, value in dict.items():
#     if value >= 100:
#         print(key, value)

# {'Canon EOS 6D': 1246, 'Canon EOS 60D': 382, 'NIKON D810': 469, 'Canon EOS 5D Mark III': 2516, 'Canon EOS 5D Mark II': 938, 'Canon EOS 5D Mark IV': 976, 'Canon EOS-1D X Mark II': 36, 'Canon EOS 20D': 25, 'ILCE-7RM2': 85, 'NIKON D4': 210, 'NIKON D7000': 40, 'NIKON D600': 153, 'ILCE-7R': 70, 'ILCE-7M2': 279, 'Canon EOS 6D Mark II': 253, 'ILCE-7M3': 147, 'NIKON Df': 53, 'Canon EOS 40D': 29, 'Canon EOS 7D': 73, 'Canon EOS 5DS': 11, 'NIKON D4S': 38, 'NIKON D5200': 41, 'GFX 50S': 42, 'Canon EOS-1D X': 320, 'NIKON D90': 73, 'NIKON D7100': 33, 'NIKON D700': 118, 'NIKON D3400': 7, 'NIKON D800': 151, 'ILCE-7RM3': 55, 'NIKON D300': 124, 'NIKON D750': 197, 'X-T3': 32, 'Canon EOS REBEL T5i': 5, 'NIKON D610': 213, 'Canon EOS 550D': 81, 'NIKON D3': 21, 'ILCE-6000': 36, 'Canon EOS REBEL T3': 29, 'NIKON D3100': 16, 'ILCE-6400': 2, 'Canon EOS 70D': 37, 'DC-GH5': 13, 'Canon EOS 650D': 12, 'NIKON D3000': 3, 'NIKON Z 7': 7, 'Canon EOS 200D': 5, 'NIKON D40': 12, 'NIKON D850': 76, 'Canon EOS-1Ds Mark III': 3, 'Canon EOS 5DS R': 25, 'NIKON Z 6': 18, 'NIKON D5100': 4, 'Canon EOS 5D': 44, 'NIKON D800E': 28, 'Canon EOS 100D': 13, 'NIKON D7200': 16, 'Canon EOS 1200D': 6, 'NIKON D300S': 36, 'Canon EOS R': 9, 'Canon EOS 500D': 11, 'NIKON D3200': 8, 'NIKON D3X': 5, 'Canon EOS 350D DIGITAL': 8, 'Canon EOS 600D': 3, 'SLT-A99': 2, 'Canon EOS 700D': 3, 'ILCE-9': 15, 'NIKON D3300': 6, 'Canon EOS REBEL T2i': 3, 'Canon EOS 750D': 25}