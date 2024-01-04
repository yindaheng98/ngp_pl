import argparse
import datasets.args as datasets
import models.args as models
from datasets.ray_utils import get_rays
from models.rendering import render
from torch.utils.data import DataLoader
from tqdm import tqdm

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
args = parser.parse_args()

model = models.parse_args(parser)
train_set = datasets.parse_args(parser, split='train')
train_loader = DataLoader(
    train_set,
    num_workers=16,
    persistent_workers=True,
    batch_size=None,
    pin_memory=True)

model = model.to('cuda')
poses = train_set.poses.to('cuda')
directions = train_set.directions.to('cuda')
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
        print(results)
print(model)
print(len(train_loader))
