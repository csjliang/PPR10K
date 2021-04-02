# Preparing the data using Adobe Lightroom

### Get raws and xmps

- Download the dataset from [OneDrive](https://connectpolyu-my.sharepoint.com/:f:/g/personal/19109963r_connect_polyu_hk/EsDA5M_nN2lIrYTyNwTFZd0BCgyE-r_j2HzNhcMEQPGLlw?e=5NWXux) 
or from [百度网盘](https://pan.baidu.com/s/1hpMO__JIvqWImdL8rznYcw) with the password wu03.
- Split the train and val files:
```bash
git clone https://github.com/csjliang/PPR10K
cd utils
python split_train_val.py
```
- If you want to augment the training sets regarding tonal attributes, first 

