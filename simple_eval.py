import os
import sys
import json
import hashlib
import argparse
import datasets.args as datasets
import models.args as models
from datasets.ray_utils import get_rays
from models.rendering import render
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from einops import rearrange
import numpy as np
import cv2
import imageio


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


parser = argparse.ArgumentParser()
parser = models.add_arguements(parser)
parser = datasets.add_arguements(parser)
parser.add_argument('--exp_name', type=str, default='exp',
                    help='experiment name')
parser.add_argument('--ckpt_path', type=str, default=None,
                    help='pretrained checkpoint to load')
args = parser.parse_args()
category_args = dict(**models.dict_args(parser), **datasets.dict_args(parser), exp_name=args.exp_name)
category_sha1 = hashlib.sha1(json.dumps(category_args).encode("utf8")).hexdigest()
category_dir = os.path.join("results", args.exp_name + "_" + category_sha1[0:10])
os.makedirs(category_dir, exist_ok=True)
with open(os.path.join(category_dir, "category-eval.txt"), "w") as f:
    json.dump(category_args, f)
args_sha1 = hashlib.sha1(str(args).encode("utf8")).hexdigest()
output_dir = os.path.join(category_dir, args_sha1[0:10] + "_outputs")
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "command-eval.txt"), "w") as f:
    json.dump(sys.argv, f)

device = torch.device('cuda')
model = models.parse_args(parser).to(device).eval()
model.load_state_dict(torch.load(args.ckpt_path))

test_set = datasets.parse_args(parser, split='test')
test_loader = DataLoader(
    test_set,
    num_workers=0,
    batch_size=None,
    pin_memory=True)

directions = test_set.directions.to(device)
val_psnr = PeakSignalNoiseRatio(data_range=1).to(device)
val_ssim = StructuralSimilarityIndexMeasure(data_range=1).to(device)
all_psnr, all_ssim = [], []
progress_bar = tqdm(test_loader)
for batch in progress_bar:
    rgb_gt = batch['rgb'].to(device)
    with torch.no_grad():
        rays_o, rays_d = get_rays(directions, batch['pose'].to(device))
        kwargs = {}
        if args.scale > 0.5:
            kwargs['exp_step_factor'] = 1/256
        if args.use_exposure:
            kwargs['exposure'] = batch['exposure']

        results = render(
            model, rays_o, rays_d,
            test_time=True, random_bg=False,
            **kwargs)

        val_psnr(results['rgb'], rgb_gt)
        psnr = val_psnr.compute()
        val_psnr.reset()
        all_psnr.append(float(psnr))

        w, h = test_set.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
        val_ssim(rgb_pred, rgb_gt)
        ssim = val_ssim.compute()
        val_ssim.reset()
        all_ssim.append(float(ssim))

        idx = batch['img_idxs']
        rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
        rgb_pred = (rgb_pred*255).astype(np.uint8)
        depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
        imageio.imsave(os.path.join(output_dir, f'{idx:03d}-psnr-{psnr:.2f}-ssim-{ssim:.4f}.png'), rgb_pred)
        imageio.imsave(os.path.join(output_dir, f'{idx:03d}_depth.png'), depth)

psnr = np.mean(all_psnr)
ssim = np.mean(all_ssim)
with open(os.path.join(output_dir, "results.txt"), "w") as f:
    json.dump({"psnr": psnr, "ssim": ssim}, f)
