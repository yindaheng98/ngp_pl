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

parser = argparse.ArgumentParser()
parser = models.add_arguements(parser)
parser = datasets.add_arguements(parser)
parser.add_argument('--exp_name', type=str, default='exp',
                    help='experiment name')
parser.add_argument('--num_epochs', type=int, default=30,
                    help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='learning rate')
parser.add_argument('--random_bg', action='store_true', default=False,
                    help='''whether to train with random bg color (real scene only)
                        to avoid objects with black color to be predicted as transparent
                        ''')
parser.add_argument('--distortion_loss_w', type=float, default=0,
                    help='''weight of distortion loss (see losses.py),
                        0 to disable (default), to enable,
                        a good value is 1e-3 for real scene and 1e-2 for synthetic scene
                        ''')
args = parser.parse_args()
category_args = dict(**models.dict_args(parser), **datasets.dict_args(parser), exp_name=args.exp_name)
category_sha1 = hashlib.sha1(json.dumps(category_args).encode("utf8")).hexdigest()
category_dir = os.path.join("results", args.exp_name + "_" + category_sha1[0:10])
os.makedirs(category_dir, exist_ok=True)
with open(os.path.join(category_dir, "category-train.txt"), "w") as f:
    json.dump(category_args, f)
args_sha1 = hashlib.sha1(str(args).encode("utf8")).hexdigest()
args_dir = os.path.join(category_dir, args_sha1[0:10])
os.makedirs(args_dir, exist_ok=True)
with open(os.path.join(args_dir, "command-train.txt"), "w") as f:
    json.dump(sys.argv, f)
ckpt_dir = os.path.join(args_dir, "ckpts")
os.makedirs(ckpt_dir, exist_ok=True)

device = torch.device('cuda')
model = models.parse_args(parser).to(device).train()

train_set = datasets.parse_args(parser, split='train')
train_loader = DataLoader(
    train_set,
    num_workers=8,
    persistent_workers=True,
    batch_size=None,
    pin_memory=True)

net_params = [p for _, p in model.named_parameters()]
net_loss = NeRFLoss(lambda_distortion=args.distortion_loss_w)
net_opt = FusedAdam(net_params, args.lr, eps=1e-15)
net_sch = CosineAnnealingLR(net_opt, args.num_epochs, args.lr/30)
train_psnr = PeakSignalNoiseRatio(data_range=1).to(device)

warmup_steps, update_interval, global_step = 256, 16, 0
poses = train_set.poses.to(device)
directions = train_set.directions.to(device)
model.mark_invisible_cells(train_set.K.to(device), poses, train_set.img_wh)
for epoch in range(args.num_epochs):
    progress_bar = tqdm(train_loader)
    for batch in progress_bar:
        batch['rgb'] = batch['rgb'].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            if global_step % update_interval == 0:
                model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
                                          warmup=global_step < warmup_steps,
                                          erode=isinstance(train_set, ColmapDataset))

            rays_o, rays_d = get_rays(directions[batch['pix_idxs']], poses[batch['img_idxs']])
            kwargs = {}
            if args.scale > 0.5:
                kwargs['exp_step_factor'] = 1/256
            if args.use_exposure:
                kwargs['exposure'] = batch['exposure']

            results = render(
                model, rays_o, rays_d,
                test_time=False, random_bg=args.random_bg,
                **kwargs)

            loss_d = net_loss(results, batch)
            if args.use_exposure:
                zero_radiance = torch.zeros(1, 3, device=device)
                unit_exposure_rgb = model.log_radiance_to_rgb(zero_radiance, exposure=torch.ones(1, 1, device=device))
                loss_d['unit_exposure'] = 0.5*(unit_exposure_rgb-train_set.unit_exposure_rgb)**2
            loss = sum(lo.mean() for lo in loss_d.values())
            with torch.no_grad():
                psnr = train_psnr(results['rgb'], batch['rgb'])
            progress_bar.set_description(f"{epoch+1}/{args.num_epochs} loss %.4f psnr %.4f" % (loss, psnr))
            net_opt.zero_grad()
            loss.backward()
            net_opt.step()
        global_step += 1
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "epoch-%04d-psnr-%.2f.ckpt" % (epoch + 1, psnr)))
    net_sch.step()
torch.save(model.state_dict(), os.path.join(ckpt_dir, "final-psnr-%.2f.ckpt" % psnr))
