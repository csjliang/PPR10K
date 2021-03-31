import argparse
import time
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import sys
from PIL import Image
import torchvision.transforms.functional as TF

from models_x import *
from datasets_AIPS_evaluation import *

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

parser = argparse.ArgumentParser()
parser.add_argument("--lut_number", type=int, default=5, help="number of lut")
parser.add_argument("--lut_dim", type=int, default=33, help="dimension of lut")
parser.add_argument("--epoch", type=int, default=270, help="epoch to start training from")
parser.add_argument("--aug_test", type=bool, default=True, help="self-ensemble for testing")
parser.add_argument("--dataset_name", type=str, default="consistency_colorfix_rename_1220", help="name of the datasets") #consistency_colorfix_rename_1220  beijing_1225
parser.add_argument("--model_dir", type=str, default="newval_nomask_glc_zhangli", help="path to save model")
parser.add_argument("--use_mask", type=bool, default=False,
                    help="whether to use the human segmentation mask for weighted loss")
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False

criterion_pixelwise = torch.nn.MSELoss()
# Initialize generator and discriminator
LUT1 = Generator3DLUT_identity(dim=opt.lut_dim)
LUT2 = Generator3DLUT_identity(dim=opt.lut_dim)
LUT3 = Generator3DLUT_identity(dim=opt.lut_dim)
LUT4 = Generator3DLUT_identity(dim=opt.lut_dim)
LUT5 = Generator3DLUT_identity(dim=opt.lut_dim)
classifier = resnet18_224(out_dim=5)
trilinear_ = TrilinearInterpolation()

if cuda:
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    LUT3 = LUT3.cuda()
    LUT4 = LUT4.cuda()
    LUT5 = LUT5.cuda()
    classifier = classifier.cuda()
    criterion_pixelwise.cuda()

# Load pretrained models
LUTs = torch.load("saved_models/%s/LUTs_%d.pth" % (opt.model_dir, opt.epoch))
LUT1.load_state_dict(LUTs["1"])
LUT2.load_state_dict(LUTs["2"])
LUT3.load_state_dict(LUTs["3"])
LUT4.load_state_dict(LUTs["4"])
LUT5.load_state_dict(LUTs["5"])

classifier.load_state_dict(torch.load("saved_models/%s/classifier_%d.pth" % (opt.model_dir, opt.epoch)))
classifier.eval()

dataloader = DataLoader(
    ImageDataset_paper("/data1/liangjie/", mode="test", use_mask=opt.use_mask),
    batch_size=1,
    shuffle=False,
    num_workers=1,
)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def generator(img, aug=False):
    pred = classifier(img).squeeze()
    if len(pred.shape) == 1:
        pred = pred.unsqueeze(0)
    gen_A1 = LUT1(img)
    gen_A2 = LUT2(img)
    gen_A3 = LUT3(img)
    gen_A4 = LUT4(img)
    gen_A5 = LUT5(img)

    combine_A = img.new(img.size())
    for i in range(img.size(0)):
        combine_A[i, :, :, :] = (
                    pred[i, 0] * gen_A1[i, :, :, :] + pred[i, 1] * gen_A2[i, :, :, :] + pred[i, 2] * gen_A3[i, :, :,
                                                                                                     :] +
                    pred[i, 3] * gen_A4[i, :, :, :] + pred[i, 4] * gen_A5[i, :, :, :])

    return combine_A


def visualize_result():
    """Saves a generated sample from the validation set"""
    out_dir = "/data1/liangjie/3dlut/images/val_all_full/%s_%d" % (opt.model_dir, opt.epoch)
    os.makedirs(out_dir, exist_ok=True)
    for i, batch in enumerate(dataloader):
        real_A = Variable(batch["A_input"].type(Tensor))
        img_name = batch["input_name"]
        fake_B = generator(real_A, aug=opt.aug_test)
        save_image(fake_B, os.path.join(out_dir, "%s.png" % (img_name[0][:-4])), nrow=1, normalize=False)

visualize_result()
