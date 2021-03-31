from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import argparse
from AIPS_learn_params_dataset import AIPS_param_Dataset as AIPS_Dataset
from model import set_parameter_requires_grad, initialize_model
import util
import numpy as np
np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, default='resnet_bz16_xiaobai_relative_valloss', help="experiment name")
parser.add_argument("--gpu_ids", type=str, default='4', help="used gpus")
parser.add_argument("--load_mode", type=str, default='imagenet_pretrained', help="mode of loading models, [imagenet_pretrained | best | epoch]")
parser.add_argument("--num_classes", type=int, default=11, help="number of learned params")
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--num_epochs", type=int, default=10000, help="number of epochs of training")
parser.add_argument("--data_root_train", type=str, default="/data/vdb/liangjie/AIPS_data/PPR10Kdataset_newval_10081/source_540", help="name of the dataset")
parser.add_argument("--data_root_val", type=str, default="/data/vdb/liangjie/AIPS_data/PPR10Kdataset_newval_10081/source_540_val", help="name of the dataset")
parser.add_argument("--save_root", type=str, default="/data/vdb/liangjie/AIPS_data/PPR10Kdataset_newval_10081/experiment_pr", help="save checkpoint root")
parser.add_argument("--xmp_ori_val_path", type=str, default="/data/vdb/liangjie/AIPS_data/PPR10Kdataset_newval_10081/xmp_ori_10081_renamed", help="xmp files for generating the predicted xmp")
parser.add_argument("--xmp_pkl_path", type=str, default="/data/vdb/liangjie/AIPS_data/PPR10Kdataset_newval_10081/xiaobai_params_norm_tmp.pkl", help="name of the dataset")
parser.add_argument("--img_size", type=int, default=224, help="resolution of the input image")
parser.add_argument("--model_name", type=str, default="resnet", help="selected model [from_raw | vgg16]")
parser.add_argument("--optimizer", type=str, default="Adam", help="[Adam | SGD]")
parser.add_argument("--loss_func", type=str, default="l1", help="[l1 | l2]")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=6, help="number of cpu threads to use during batch generation")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model checkpoints")
parser.add_argument("--if_imagenet_normalization", default=False, help="")
parser.add_argument("--get_pred_xmp", default=True, help="")
parser.add_argument("--get_pred_xmp_interval", type=int, default=5, help="")
opt = parser.parse_args()
print(opt)

# os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

def train_model(model, dataloaders, criterion, optimizer, num_epochs, maxs, mins):
    since = time.time()

    val_acc_history = []

    best_loss = 10.0

    if opt.get_pred_xmp:
        pred_xmp_dict = {}

    for epoch in range(num_epochs):
        print('\n{}, Epoch {}/{}'.format(opt.experiment_name, epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for data in dataloaders[phase]:

                inputs = data['img'].to(device)
                # if phase == 'train':
                labels = data['param_GT'].to(device)
                img_name = data['img_name']

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs_param = model(inputs)
                    loss = criterion(labels, outputs_param)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                if phase == 'val' and epoch % opt.get_pred_xmp_interval == 0:
                    reconstructed_pred = util.recons_params_val(outputs_param.squeeze(), maxs, mins)

                    # get pred xmp of val
                    if opt.get_pred_xmp:
                        util.get_pred_xmp_relative_val(img_name[0], reconstructed_pred, opt.xmp_ori_val_path, os.path.join(save_xmp_path, 'epoch_{}'.format(epoch)))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            log = '{} Loss: {:.4f}'.format(phase, epoch_loss)
            print(log)

            # save log
            with open(save_path + '/logs.txt',"a") as file:
                file.write('%s\n' % log)

            # save model
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(save_path, 'net_best_params.pkl'))
            if phase == 'val' and epoch % opt.checkpoint_interval == 0:
                torch.save(model.state_dict(), os.path.join(save_path, 'net_epoch_{}.pkl'.format(epoch)))
            if phase == 'val':
                val_acc_history.append(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    return model, val_acc_history

# Initialize the model for this run
model_ft = initialize_model(opt.model_name, opt.num_classes, feature_extract, opt)

print(model_ft)

if not os.path.exists(opt.save_root):
    os.mkdir(opt.save_root)

save_path = os.path.join(opt.save_root, opt.experiment_name)
if not os.path.exists(save_path):
    os.mkdir(save_path)
save_xmp_path = os.path.join(opt.save_root, opt.experiment_name + '/xmp')
if not os.path.exists(save_xmp_path):
    os.mkdir(save_xmp_path)

print("Initializing Datasets and Dataloaders...")

image_datasets = {x: AIPS_Dataset(opt, x) for x in ['train', 'val']}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batch_size if x=='train' else 1, shuffle=x=='train', num_workers=opt.n_cpu) for x in ['train', 'val']}

# Detect if we have a GPU available
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
# device = torch.device('cpu')  # get device name: CPU or GPU

# Send the model to GPU
model_ft = model_ft.to(device)

params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
if opt.optimizer == 'SGD':
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
elif opt.optimizer == 'Adam':
    optimizer_ft = optim.Adam(params_to_update, lr=opt.lr, betas=(opt.b1, opt.b2))
else:
    print('Invalid optimizer, using Adam')
    optimizer_ft = optim.Adam(params_to_update, lr=opt.lr, betas=(opt.b1, opt.b2))

# Setup the loss fxn
if opt.loss_func == 'l2':
    criterion = nn.MSELoss()
elif opt.loss_func == 'l1':
    criterion = nn.L1Loss()
else:
    print('Invalid loss function, using l1')
    criterion = nn.L1Loss()

xmp_dict = util.read_xmp(opt.xmp_pkl_path)
maxs = xmp_dict['max']
mins = xmp_dict['min']

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=opt.num_epochs, maxs=maxs, mins=mins)