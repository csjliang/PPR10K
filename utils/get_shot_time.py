import os
import exifread
import shutil

root_dir = r'F:\AIPS\paper\RAW_renamed\images_tif_540\target_wanglei'

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
        return new_name2


time_tags = []
for file_name in os.listdir(root_dir):
    file_path = os.path.join(root_dir, file_name)
    if os.path.isfile(file_path):
        time_tag = getExif(file_path, file_name)
        time_tags.append(time_tag)

print(time_tags)
