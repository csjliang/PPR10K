# Portrait Photo Retouching with PPR10K

### [Paper]() |   [Supplementary Material]()

> **PPR10K: A Large-Scale Portrait Photo Retouching Dataset with Human-Region Mask and Group-Level Consistency** <br>
> [Jie Liang](liangjie.xyz)\*, Hui Zeng\*, Miaomiao Cui, Xuansong Xie and [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang/). <br>
> In CVPR 2021.

PPR10K contains 1,681 groups and 11,161 high-quality raw portrait photos in total. 
High-resolution segmentation masks of human regions are provided. 
Each raw photo is retouched by three experts, while they elaborately adjust each group of photos to have consistent tones.

### Sample Images

![sample_images](imgs/sample_imgs.jpg)

### Overview

All data is hosted on [Baidu Drive](https://pan.baidu.com/s/1hpMO__JIvqWImdL8rznYcw) (Password: wu03):

| Path | Size | Files | Format | Description
| :--- | :---: | ----: | :----: | :----------
| [PPR10K-dataset](https://pan.baidu.com/s/1hpMO__JIvqWImdL8rznYcw) | 345 GB | 122,810 | | Main folder
| &boxvr;&nbsp; raw | 313 GB | 11,161 | RAW | All photos in raw format (.CR2, .NEF, .ARW, etc)
| &boxvr;&nbsp; xmp_source | 130 MB | 11,161 | XMP | Default meta-file of the raw photos in CameraRaw, used in our [data augmentation]()
| &boxvr;&nbsp; xmp_target_a | 130 MB | 11,161 | XMP | Meta-file of the raw photos retouched by the expert a
| &boxvr;&nbsp; xmp_target_b | 130 MB | 11,161 | XMP | Meta-file of the raw photos retouched by the expert b
| &boxvr;&nbsp; xmp_target_c | 130 MB | 11,161 | XMP | Meta-file of the raw photos retouched by the expert c
| &boxvr;&nbsp; masks_full | 697 MB | 11,161 | PNG | Full-resolution human-region masks in binary format
| &boxvr;&nbsp; masks_360p | 56 MB | 11,161 | PNG | 360p human-region masks for fast training and validation
| &boxvr;&nbsp; train_val_images_tif_360p | 32 GB | 44644 | TIF | 360p Source (16 bit tiff) and target (8 bit tiff) images for fast training and validation
| &boxur;&nbsp; hists | 624KB | 39 | PNG | Overall statistics of the dataset

For more details and the instructions for usage, please refer to the documents [here](dataset/README.md).

### Agreement

- All files in the PPR10K dataset are available for ***non-commercial research purposes*** only.


### Citation
If you use this dataset and code for your research, please cite our paper.
```
@inproceedings{park2019SPADE,
  title={PPR10K: A Large-Scale Portrait Photo Retouching Dataset with Human-Region Mask and Group-Level Consistency},
  author={Liang, Jie and Zeng, Hui and Cui, Miaomiao and Xie, Xuansong and Zhang, Lei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```
