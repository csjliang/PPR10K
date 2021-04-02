import os
import exifread
from matplotlib import pyplot as plt

# path of images
root_dir = ''
# path to save the hist
save_path = ''

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

months = []
hours = []

for time in time_tags:
    months.append(time.split('_')[0][4:6])
    hours.append(time.split('_')[1][:2])

print(months)
print(hours)

months = sorted(months)
hours = sorted(hours)

plt.hist(months, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
plt.title('month')
plt.savefig(os.path.join(save_path, 'month.png'))
plt.show()

plt.hist(hours, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
plt.title('hour')
plt.savefig(os.path.join(save_path, 'hour.png'))
plt.show()
