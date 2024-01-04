import argparse
import datasets.args as datasets
import models.args as models
from datasets.ray_utils import get_rays
from models.rendering import render
from torch.utils.data import DataLoader
from tqdm import tqdm
from losses import NeRFLoss
from apex.optimizers import FusedAdam
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure
)

parser = argparse.ArgumentParser()
parser = models.add_arguements(parser)
parser = datasets.add_arguements(parser)
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

device = torch.device('cuda')
model = models.parse_args(parser)
train_set = datasets.parse_args(parser, split='train')
train_loader = DataLoader(
    train_set,
    num_workers=16,
    persistent_workers=True,
    batch_size=None,
    pin_memory=True)

model = model.to(device)
net_params = [p for _, p in model.named_parameters()]
net_loss = NeRFLoss(lambda_distortion=args.distortion_loss_w)
net_opt = FusedAdam(net_params, args.lr, eps=1e-15)
net_sch = CosineAnnealingLR(net_opt, args.num_epochs, args.lr/30)
train_psnr = PeakSignalNoiseRatio(data_range=1).to(device)

poses = train_set.poses.to(device)
directions = train_set.directions.to(device)
for epoch in range(args.num_epochs):
    for batch in tqdm(train_loader):
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
        batch['rgb'] = batch['rgb'].to(device)
        loss_d = net_loss(results, batch)
        if args.use_exposure:
            zero_radiance = torch.zeros(1, 3, device=device)
            unit_exposure_rgb = model.log_radiance_to_rgb(zero_radiance,
                                                          **{'exposure': torch.ones(1, 1, device=device)})
            loss_d['unit_exposure'] = \
                0.5*(unit_exposure_rgb-train_set.unit_exposure_rgb)**2
        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            psnr = train_psnr(results['rgb'], batch['rgb'])
        print(psnr, loss)
print(model)
print(len(train_loader))
