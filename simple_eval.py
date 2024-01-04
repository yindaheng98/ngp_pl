import os
import sys
import json
import hashlib
import argparse
from datasets import ColmapDataset
import datasets.args as datasets
import models.args as models
from datasets.ray_utils import get_rays
from models.rendering import render, MAX_SAMPLES
from torch.utils.data import DataLoader
from tqdm import tqdm
from losses import NeRFLoss
from apex.optimizers import FusedAdam
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import PeakSignalNoiseRatio
from kornia.utils.grid import create_meshgrid3d

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
args_dir = os.path.join(category_dir, args_sha1[0:10])
os.makedirs(args_dir, exist_ok=True)
with open(os.path.join(args_dir, "command-eval.txt"), "w") as f:
    json.dump(sys.argv, f)
output_dir = os.path.join(args_dir, "output")
os.makedirs(output_dir, exist_ok=True)

device = torch.device('cuda')
model = models.parse_args(parser).to(device).eval()

test_set = datasets.parse_args(parser, split='test')
test_loader = DataLoader(
    test_set,
    num_workers=0,
    batch_size=None,
    pin_memory=True)

directions = test_set.directions.to(device)
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
        print(results)
