import os
import shutil

# path of the augmented xmp files
auged_xmp_path = ''
# path of the raw photos
ori_raw_path = ''
# path to save both the copied raws and the auged xmp files
auged_raw_path = ''

if not os.path.exists(auged_raw_path):
    os.mkdir(auged_raw_path)

raw_names = os.listdir(ori_raw_path)
file_names = os.listdir(auged_xmp_path)

i = 0
print(len(file_names))
for file_name in file_names:

    i += 1

    if i >= 0:
        ori_file_name = file_name.split('.')[0].split('_')[0] + '_' + file_name.split('.')[0].split('_')[1] + '_' + file_name.split('.')[0].split('_')[2] + '.xmp'

        ori_name = ori_file_name[:-4]

        try:
            raw_name = ori_name.split('_')[0]+'_'+ori_name.split('_')[1] + '.CR2'
            source_raw_path = os.path.join(ori_raw_path, raw_name)
            target_name = file_name[:-4] + raw_name[-4:]
            target_raw_path = os.path.join(auged_raw_path, target_name)
            source_xmp_path = os.path.join(auged_xmp_path, file_name)
            target_xmp_path = os.path.join(auged_raw_path, file_name)
            shutil.copyfile(source_raw_path, target_raw_path)
            shutil.copyfile(source_xmp_path, target_xmp_path)
        except:
            try:
                raw_name = ori_name.split('_')[0] + '_' + ori_name.split('_')[1] + '.NEF'
                source_raw_path = os.path.join(ori_raw_path, raw_name)
                target_name = file_name[:-4] + raw_name[-4:]
                target_raw_path = os.path.join(auged_raw_path, target_name)
                source_xmp_path = os.path.join(auged_xmp_path, file_name)
                target_xmp_path = os.path.join(auged_raw_path, file_name)
                shutil.copyfile(source_raw_path, target_raw_path)
                shutil.copyfile(source_xmp_path, target_xmp_path)
            except:
                try:
                    raw_name = ori_name.split('_')[0] + '_' + ori_name.split('_')[1] + '.ARW'
                    source_raw_path = os.path.join(ori_raw_path, raw_name)
                    target_name = file_name[:-4] + raw_name[-4:]
                    target_raw_path = os.path.join(auged_raw_path, target_name)
                    source_xmp_path = os.path.join(auged_xmp_path, file_name)
                    target_xmp_path = os.path.join(auged_raw_path, file_name)
                    shutil.copyfile(source_raw_path, target_raw_path)
                    shutil.copyfile(source_xmp_path, target_xmp_path)
                except:
                    try:
                        raw_name = ori_name.split('_')[0] + '_' + ori_name.split('_')[1] + '.RAF'
                        source_raw_path = os.path.join(ori_raw_path, raw_name)
                        target_name = file_name[:-4] + raw_name[-4:]
                        target_raw_path = os.path.join(auged_raw_path, target_name)
                        source_xmp_path = os.path.join(auged_xmp_path, file_name)
                        target_xmp_path = os.path.join(auged_raw_path, file_name)
                        shutil.copyfile(source_raw_path, target_raw_path)
                        shutil.copyfile(source_xmp_path, target_xmp_path)
                    except:
                        raw_name = ori_name.split('_')[0] + '_' + ori_name.split('_')[1] + '.RW2'
                        source_raw_path = os.path.join(ori_raw_path, raw_name)
                        target_name = file_name[:-4] + raw_name[-4:]
                        target_raw_path = os.path.join(auged_raw_path, target_name)
                        source_xmp_path = os.path.join(auged_xmp_path, file_name)
                        target_xmp_path = os.path.join(auged_raw_path, file_name)
                        shutil.copyfile(source_raw_path, target_raw_path)
                        shutil.copyfile(source_xmp_path, target_xmp_path)
