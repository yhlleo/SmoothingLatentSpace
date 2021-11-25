"""
  Smoothing the Disentangled Latent Style Space for Unsupervised Image-to-image Translation
  Based on the codes: https://github.com/clovaai/stargan-v2
"""

import os
import glob
import argparse
from munch import Munch
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
from torch.backends import cudnn
from torchvision import transforms
from torch.autograd import Variable
import torchvision.utils as vutils

from core.solver import Solver
from core.data_loader import (
    get_train_loader, 
    get_test_loader, 
    get_eval_loader
)
from core.utils import tensor2ndarray255, save_video

def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]

def slerp(low, high, weight):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-weight)*omega)/so).unsqueeze(1)*low + \
        (torch.sin(weight*omega)/so).unsqueeze(1) * high
    return res

def interp(nets, s1, s2, lerp_step, lerp_fun):
    outputs = []
    with torch.no_grad():
        for alpha in np.arange(0., 1., lerp_step):
            s = lerp_fun(s1, s2, alpha)
            fake = nets.generator(image, s, masks=masks)
            outputs.append(fake)
    return outputs

def interpolations(
    nets, 
    latent_dim, 
    image, 
    masks=None, 
    lerp_step=0.05, 
    y1=None, 
    y2=None, 
    lerp_mode='lerp'
):
    if y1 is None:
        y1 = torch.tensor([0]).long().cuda()
    if y2 is None:
        y2 = torch.tensor([1]).long().cuda()
    s1 = nets.mapping_network(torch.randn(1, latent_dim).cuda(), y1)
    s2 = nets.mapping_network(torch.randn(1, latent_dim).cuda(), y2)
 
    lerp_fun = torch.lerp if lerp_mode == "lerp" else slerp
    outputs = interp(nets, s1, s2, lerp_step, lerp_fun)
    outputs = torch.clamp(torch.cat(outputs, dim=3)*0.5+0.5, 0, 1)
    return outputs

def interpolations_loop(
    nets, 
    latent_dim, 
    image, 
    masks=None, 
    lerp_step=0.05, 
    y1=None, 
    y2=None, 
    lerp_mode='lerp'
):
    if y1 is None:
        y1 = torch.tensor([0]).long().cuda()
    if y2 is None:
        y2 = torch.tensor([1]).long().cuda()
    s1 = nets.mapping_network(torch.randn(1, latent_dim).cuda(), y1)
    s2 = nets.mapping_network(torch.randn(1, latent_dim).cuda(), y2)
    s3 = nets.mapping_network(torch.randn(1, latent_dim).cuda(), y2)
    s4 = nets.mapping_network(torch.randn(1, latent_dim).cuda(), y1)

    outputs = []
    lerp_fun = torch.lerp if lerp_mode == "lerp" else slerp
    outputs += interp(nets, s1, s2, lerp_step, lerp_fun)
    outputs += interp(nets, s2, s3, lerp_step, lerp_fun)
    outputs += interp(nets, s3, s4, lerp_step, lerp_fun)
    outputs += interp(nets, s4, s1, lerp_step, lerp_fun)
    return outputs

def test_single(
    nets, 
    image, 
    masks, 
    latent_dim, 
    ref_image=None, 
    y=0, 
    mode='latent'
):
    z = torch.randn(1, latent_dim).cuda()
    y = torch.tensor([y]).long().cuda()
    if mode == 'latent':
        s = nets.mapping_network(z, y)
    else:
        s = nets.style_encoder(ref_image, y)
    fake = nets.generator(image, s, masks=masks)
    return fake

def main(args):
    print(args)
    cudnn.benchmark = True
    if args.mode == 'train':
        torch.manual_seed(args.seed)

    solver = Solver(args)

    transform = transforms.Compose([
        transforms.Resize([args.img_size, args.img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])])

    if args.mode == 'train':
        assert len(subdirs(args.train_img_dir)) == args.num_domains
        assert len(subdirs(args.val_img_dir)) == args.num_domains
        if args.resume_iter > 0:
            solver._load_checkpoint(args.resume_iter)
        loaders = Munch(src=get_train_loader(root=args.train_img_dir,
                                             which='source',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers),
                        ref=get_train_loader(root=args.train_img_dir,
                                             which='reference',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers),
                        val=get_test_loader(root=args.val_img_dir,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers))
        solver.train(loaders)
    elif args.mode == 'eval':
        solver.evaluate()
    
    elif args.mode == 'align':
        from core.wing import align_faces
        align_faces(args, args.inp_dir, args.out_dir)

    elif args.mode == 'inter': # interpolation
        save_dir = args.save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        solver._load_checkpoint(args.resume_iter)
        nets_ema = solver.nets_ema
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        image_name = os.path.basename(args.input)
        image = Variable(transform(Image.open(args.input).convert('RGB')).unsqueeze(0).to(device))
        masks = nets_ema.fan.get_heatmap(image) if args.w_hpf > 0 else None
        y1 = torch.tensor([args.y1]).long().cuda()
        y2 = torch.tensor([args.y2]).long().cuda()
        outputs = interpolations(
            nets_ema, 
            args.latent_dim, 
            image, 
            masks, 
            lerp_step=0.1, 
            y1=y1, 
            y2=y2, 
            lerp_mode=args.lerp_mode
        )
        outputs = torch.clamp(torch.cat(outputs, dim=3)*0.5+0.5, 0, 1)
        path = os.path.join(save_dir, image_name)
        vutils.save_image(outputs.data, path, padding=0)

    elif args.mode == 'test':
        save_dir = args.save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        solver._load_checkpoint(args.resume_iter)
        nets_ema = solver.nets_ema
        
        image_name = os.path.basename(args.input)
        image = Variable(transform(Image.open(args.input).convert('RGB')).unsqueeze(0)).cuda()
        masks = nets_ema.fan.get_heatmap(image) if args.w_hpf > 0 else None
        
        image_ref = None
        if args.test_mode == 'reference':
            image_ref = Variable(transform(Image.open(args.input_ref).convert("RGB")).unsqueeze(0)).cuda()
        
        fake = test_single(
            nets_ema, 
            image, 
            masks, 
            args.latent_dim, 
            image_ref, 
            args.target_domain, 
            args.single_mode
        )
        fake = torch.clamp(fake*0.5+0.5, 0, 1)
        path = os.path.join(save_dir, image_name)
        vutils.save_image(fake.data, path, padding=0)
    
    elif args.mode == 'video':
        save_dir = args.save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        solver._load_checkpoint(args.resume_iter)
        nets_ema = solver.nets_ema

        image_name = os.path.basename(args.input)
        image = Variable(transform(Image.open(args.input).convert('RGB')).unsqueeze(0)).cuda()
        masks = nets_ema.fan.get_heatmap(image) if args.w_hpf > 0 else None

        y1 = torch.tensor([args.y1]).long().cuda()
        y2 = torch.tensor([args.y2]).long().cuda()
        outputs = interpolations_loop(
            nets_ema, 
            args.latent_dim, 
            image, 
            masks, 
            lerp_step=0.02, 
            y1=y1, 
            y2=y2, 
            lerp_mode=args.lerp_mode
        )
        outputs = torch.cat(outputs)
        outputs = tensor2ndarray255(outputs)
        path = os.path.join(save_dir, '{}-video.mp4'.format(image_name))
        save_video(path, outputs)

    else:
        raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=1,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=1,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--lambda_tri', type=float, default=1,
                        help='Weight for triplet loss')
    parser.add_argument('--init_lambda_kl', type=float, default=0,
                        help='Inital weight for kl loss')
    parser.add_argument('--lambda_kl', type=float, default=1,
                        help='Weight for kl loss')
    parser.add_argument('--ds_iter', type=int, default=100000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--kl_start_iter', type=int, default=40000,
                        help='Number of iterations to use kl loss')
    parser.add_argument('--kl_iter', type=int, default=60000, 
                        help='Number of iterations to increate the kl loss weight')
    parser.add_argument('--w_hpf', type=float, default=1,
                        help='weight for high-pass filtering')
    parser.add_argument('--lambda_lpips', type=float, default=0,
                        help='weight of similarity between original image and generated image')
    parser.add_argument('--triplet_margin', type=float, default=0.1)

    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5,
                        help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=100000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--adam_beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--adam_beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=10,
                        help='Number of generated images per domain during sampling')
    
    # misc
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train',  'inter', 'eval', 'align', 'test', 'video'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')
    parser.add_argument('--dataset', type=str, default='celeba_hq', help='[celeba_hq | afhq | FacePoses]')
    parser.add_argument('--lerp_mode', type=str, default='lerp', help='[lerp | slerp]')
    parser.add_argument('--dist_mode', type=str, default='squared_l2', help='[l2 | squared_l2], the distance type of LPIPS')

    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='data/celeba_hq/train',
                        help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='data/celeba_hq/val',
                        help='Directory containing validation images')
    parser.add_argument('--sample_dir', type=str, default='expr/samples',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints',
                        help='Directory for saving network checkpoints')

    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='expr/eval',
                        help='Directory for saving metrics, i.e., FID and LPIPS')

    # directory for testing
    parser.add_argument('--result_dir', type=str, default='expr/results',
                        help='Directory for saving generated images and videos')
    parser.add_argument('--src_dir', type=str, default='assets/representative/celeba_hq/src',
                        help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref',
                        help='Directory containing input reference images')
    parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
                        help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
                        help='output directory when aligning faces')
    parser.add_argument('--output_name', type=str)

    # face alignment
    parser.add_argument('--wing_path', type=str, default='expr/checkpoints/wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='expr/checkpoints/celeba_lm_mean.npz')

    # step size
    parser.add_argument('--print_every', type=int, default=40)
    parser.add_argument('--sample_every', type=int, default=5000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--eval_every', type=int, default=100000)
    parser.add_argument('--ppl_image_list', type=str, help='eval image list for ppl metric')
    parser.add_argument('--ppl_mode', type=str, default='latent', help='[latent | reference]')
    parser.add_argument('--test_mode', type=str, default='latent', help='[latent | reference]')
    parser.add_argument('--input', type=str, help='input image name')
    parser.add_argument('--input_ref1', type=str, help='input reference image name')
    parser.add_argument('--input_ref2', type=str, help='input reference image name')
    parser.add_argument('--target_domain', type=int, default=0)
    parser.add_argument('--save_dir', type=str)

    args = parser.parse_args()

    main(args)

