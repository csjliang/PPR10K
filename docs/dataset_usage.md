# Preparing the data using Adobe Lightroom

### Get raws and xmps

- Download the dataset from [OneDrive](https://connectpolyu-my.sharepoint.com/:f:/g/personal/19109963r_connect_polyu_hk/EsDA5M_nN2lIrYTyNwTFZd0BCgyE-r_j2HzNhcMEQPGLlw?e=5NWXux) 
or from [百度网盘](https://pan.baidu.com/s/1hpMO__JIvqWImdL8rznYcw) with the password wu03.
- Split the train and val files: *You have to specify the data paths before runing all below scripts*
```bash
git clone https://github.com/csjliang/PPR10K
cd utils
python split_train_val.py
```
- If you want to augment the training sets regarding tonal attributes, first get the augmented xmp files with random modifications of the six dominate attributes stated in our paper (Temperature, Tint, Exposure, Highlights, Contrast and Saturation) by:
```bash
python data_augment_get_xmps.py
```
- If you have enough disk space, copy the raws and rename them according to the augmented xmp files:
```bash
python data_augment_copy_raws.py
```

1

### Get Images using Adobe Lightroom

- Add the data folder (raw photos and the corresponding xmps with exactly the same name) by clicking 'Add folder'
- Select all the images and click 'Export';
- Set the save root; 
- Set Image Format = TIFF, Compression=None, Color Space=sRGB, Bit Depth=16 bit/component for the Input image or Bit Depth=8 bit/component for the Target;
- Set Image Sizing: Resize to Fit = Long Edge. Fill in 540 pixels;
- Click 'Export'.

Note when generating the 3 versions of target images, you need to replace the xmps in your folder, re-add the folder to Lightroom and conduct the above operations step-by-step.
 