# Portrait Photo Retouching with PPR10K

### [Paper]() |   [Supplementary Material]()

> **PPR10K: A Large-Scale Portrait Photo Retouching Dataset with Human-Region Mask and Group-Level Consistency** <br>
> [Jie Liang](liangjie.xyz)\*, Hui Zeng\*, Miaomiao Cui, Xuansong Xie and [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang/). <br>
> In CVPR 2021.

The proposed **Portrait Photo Retouching dataset (PPR10K)** is a large-scale and diverse dataset that contains: <br>
- 11,161 high-quality raw portrait photos (resolutions from 4K to 8K) in 1,681 groups; <br>
- 3 versions of manual retouched targets of all photos given by 3 expert retouchers; <br> 
- full resolution human-region masks of all photos.



### Samples

![sample_images](imgs/motivation.jpg)

Two example groups of photos from the PPR10K dataset.
**Top**: the raw photos; 
**Bottom**: the retouched results from expert-a and the human-region masks.
The raw photos exhibit poor visual quality and large variance in subject views, background contexts, 
lighting conditions and camera settings. 
In contrast, the retouched results demonstrate both good visual quality (with *human-region priority*) and *group-level consistency*.

This dataset is first of its kind to consider the two special and practical requirements of portrait photo retouching task, i.e., 
Human-Region Priority and Group-Level Consistency. Three main challenges are expected to be tackled in the follow-up researches: <br>
- Flexible and content-adaptive models for such a diverse task regarding both image contents and lighting conditions; <br>
- Highly efficient models to process practical resolution from 4K to 8K; <br>
- Robust and stable models to meet the requirement of group-level consistency. 

### Agreement

- All files in the PPR10K dataset are available for ***non-commercial research purposes*** only.

### Overview

All data is hosted on [OneDrive](https://connectpolyu-my.sharepoint.com/:f:/g/personal/19109963r_connect_polyu_hk/EsDA5M_nN2lIrYTyNwTFZd0BCgyE-r_j2HzNhcMEQPGLlw?e=5NWXux) 
and [百度网盘](https://pan.baidu.com/s/1hpMO__JIvqWImdL8rznYcw) (验证码: wu03):

| Path | Size | Files | Format | Description
| :--- | :---: | ----: | :----: | :----------
| [PPR10K-dataset](https://connectpolyu-my.sharepoint.com/:f:/g/personal/19109963r_connect_polyu_hk/EsDA5M_nN2lIrYTyNwTFZd0BCgyE-r_j2HzNhcMEQPGLlw?e=5NWXux) | 345 GB | 122,810 | | Main folder
| &boxvr;&nbsp; raw | 313 GB | 11,161 | RAW | All photos in raw format (.CR2, .NEF, .ARW, etc)
| &boxvr;&nbsp; xmp_source | 130 MB | 11,161 | XMP | Default meta-file of the raw photos in CameraRaw, used in our [data augmentation](docs/dataset_usage.md)
| &boxvr;&nbsp; xmp_target_a | 130 MB | 11,161 | XMP | CameraRaw meta-file of the raw photos recoding the full adjustments by expert a
| &boxvr;&nbsp; xmp_target_b | 130 MB | 11,161 | XMP | CameraRaw meta-file of the raw photos recoding the full adjustments by expert b
| &boxvr;&nbsp; xmp_target_c | 130 MB | 11,161 | XMP | CameraRaw meta-file of the raw photos recoding the full adjustments by expert c
| &boxvr;&nbsp; masks_full | 697 MB | 11,161 | PNG | Full-resolution human-region masks in binary format
| &boxvr;&nbsp; masks_360p | 56 MB | 11,161 | PNG | 360p human-region masks for fast training and validation
| &boxvr;&nbsp; train_val_images_tif_360p | 32 GB | 44644 | TIF | 360p Source (16 bit tiff) and target (8 bit tiff) images for fast training and validation
| &boxur;&nbsp; hists | 624KB | 39 | PNG | Overall statistics of the dataset

One can directly use the 360p (of 540x360 or 360x540 resolution in sRGB color space) training and validation files (photos and the corresponding human-region masks) we have provided following the settings in our paper. <br>
Also, see the [instructions](docs/dataset_usage.md) to customize your data (e.g., augment the training samples regarding illuminations and colors, get photos with higher or full resolutions).

### Training and Validating the PPR using [3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT)

#### Installation

- Clone this repo.
```bash
git clone https://github.com/csjliang/PPR10K
cd PPR10K/3DLUT_PPR10K/
```

- Install dependencies.
```bash
pip install -r requirements.txt
```

- Build. Modify the CUDA path in ```trilinear_cpp/setup.sh``` adaptively and
```bash
cd trilinear_cpp
sh trilinear_cpp/setup.sh
```

#### Training

- Training without HRP and GLC strategy, save models:
```bash
python train.py --data_path [path_to_dataset] --gpu_id [gpu_id] --use_mask False --output_dir [path_to_save_models]
```

- Training with HRP and without GLC strategy, save models:
```bash
python train.py --data_path [path_to_dataset] --gpu_id [gpu_id] --use_mask True --output_dir [path_to_save_models]
```

- Training without HRP and with GLC strategy, save models:
```bash
python train_GLC.py --data_path [path_to_dataset] --gpu_id [gpu_id] --use_mask False --output_dir [path_to_save_models]
```

- Training with both HRP and GLC strategy, save models:
```bash
python train_GLC.py --data_path [path_to_dataset] --gpu_id [gpu_id] --use_mask True --output_dir [path_to_save_models]
```

#### Evaluation

- Generate the retouched results:
```bash
python validation.py --data_path [path_to_dataset] --gpu_id [gpu_id] --model_dir [path_to_models]
```

- Use matlab to calculate the measures in our paper:
```bash
../MATLAB/calculate_metrics.m
```

### Citation
If you use this dataset or code for your research, please cite our paper.
```
@inproceedings{park2019SPADE,
  title={PPR10K: A Large-Scale Portrait Photo Retouching Dataset with Human-Region Mask and Group-Level Consistency},
  author={Liang, Jie and Zeng, Hui and Cui, Miaomiao and Xie, Xuansong and Zhang, Lei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

### Related Projects

[3D LUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT)