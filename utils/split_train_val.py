import os

data_root = ''
target_train_root = ''
target_val_root = ''

os.makedirs(target_train_root, exist_ok=True)
os.makedirs(target_val_root, exist_ok=True)

count_train_file, count_val_file = 0, 0
for file in os.listdir(data_root):
    source_path = os.path.join(data_root, file)
    if int(file.split('_')[0]) < 1356:
        target_path = os.path.join(target_train_root, file)
        count_train_file += 1
    else:
        target_path = os.path.join(target_val_root, file)
        count_val_file += 1
    os.rename(source_path, target_path)

print('num of train files: ', count_train_file)
print('num of val files: ', count_val_file)