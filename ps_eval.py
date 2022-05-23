import os
import glob
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.autograd import Variable

from metrics.lpips import LPIPS
from metrics.pps import PPS 

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str)
parser.add_argument('--num_inters', type=int, default=20)
parser.add_argument('--suffix', type=str, default='png', help='png or jpg')
parser.add_argument('--img_size', type=int, default=256)
args = parser.parse_args()

lpips = LPIPS().eval().cuda()

transform = transforms.Compose([
    transforms.Resize([args.img_size, args.img_size]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])])

images_list = glob.glob(os.path.join(args.data_dir, "*.{}".format(args.suffix)))
image_names = []
for im in images_list:
    if "afhq" in args.data_dir:
        im = im.split("/")[-1].split("-")[0]
    else:
        im = im.split("/")[-1].split("_")[0]
    if im not in image_names:
        image_names.append(im)

pps_scores = []
for im in tqdm(image_names):
    # load images in sequence 
    images = []
    for i in range(args.num_inters):
        if "afhq" in args.data_dir:
            fpath = os.path.join(args.data_dir, "{}-{:03d}.{}".format(im, i, args.suffix))
        else:
            fpath = os.path.join(args.data_dir, "{}_{:03d}.{}".format(im, i, args.suffix))
        images.append(Variable(transform(Image.open(fpath).convert('RGB')).unsqueeze(0).cuda()))
    
    # compute the scores 
    scores = []
    for i in range(args.num_inters-1):
        #scores.append(lpips(images[i], images[i+1]).cpu().data.numpy())
        scores.append(lpips(images[i], images[i+1]).cpu().item())
    scores.append(lpips(images[0], images[-1]).cpu().item())
    scores = np.array(scores).reshape(-1, args.num_inters)

    pps_scores.append(PPS(scores[0, -1], scores[0:1, :-1]))
print("PS score:", np.mean(pps_scores))

