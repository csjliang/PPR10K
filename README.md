# Portrait Photo Retouching with PPR10K

### [Paper]() |   [Supplementary Material]()

> **PPR10K: A Large-Scale Portrait Photo Retouching Dataset with Human-Region Mask and Group-Level Consistency** <br>
> [Jie Liang](liangjie.xyz)\*, Hui Zeng\*, Miaomiao Cui, Xuansong Xie and [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang/). <br>
> In CVPR 2021.

PPR10K contains 1,681 groups and 11,161 high-quality raw portrait photos in total. 
High-resolution segmentation masks of human regions are provided. 
Each raw photo is retouched by three experts, while they elaborately adjust each group of photos to have consistent tones.

## Overview

All data is hosted on [Baidu Drive](https://pan.baidu.com/s/1hpMO__JIvqWImdL8rznYcw) (Password: wu03):

| Path | Size | Files | Format | Description
| :--- | :---: | ----: | :----: | :----------
| [PPR10K-dataset](https://pan.baidu.com/s/1hpMO__JIvqWImdL8rznYcw) | 345 GB | 122,810 | | Main folder
| &boxvr;&nbsp; raw | 313 GB | 11,161 | RAW | 
| &boxvr;&nbsp; xmp_source | 130 MB | 11,161 | XMP | 
| &boxvr;&nbsp; xmp_target_a | 130 MB | 11,161 | XMP | 
| &boxvr;&nbsp; xmp_target_b | 130 MB | 11,161 | XMP | 
| &boxvr;&nbsp; xmp_target_c | 130 MB | 11,161 | XMP | 
| &boxvr;&nbsp; masks_full | 697 MB | 11,161 | PNG | 
| &boxvr;&nbsp; masks_360p | 56 MB | 11,161 | PNG | 
| &boxvr;&nbsp; train_val_images_tif_360p | 32 GB | 44644 | TIF | 
| &boxvr;&nbsp; masks_full | 697 MB | 11,161 | PNG | 
| &boxur;&nbsp; hists | 624KB | 39 | PNG | 



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