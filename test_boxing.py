import torch
import time
import os
import numpy as np
from models.networks import NGP
from models.rendering import render
from metrics import psnr
from datasets import dataset_dict
from datasets.ray_utils import get_rays
from utils import load_ckpt
from train import depth2img
import imageio
path_prefix = '/volume/'
path_name = path_prefix + 'data/boxing/'
dataset_name = 'colmap'
scene = 'boxing'
exp_names = 'boxing{}'
frame_names = 'frame{}'
frame_indexes = [1, 2, 3, 4] # range(1, 3)
frame_images = []
frame_depths = []
datasets = []
for frame_index in frame_indexes:
    exp_name = exp_names.format(frame_index)
    frame_name = frame_names.format(frame_index)
    dataset = dataset_dict[dataset_name](
        f'{path_name}{frame_name}',
        split='all', downsample=1.0/4
    )
    model = NGP(scale=0.5).cuda()
    load_ckpt(model, f'{path_prefix}ckpts/{dataset_name}/{exp_name}/epoch=19_slim.ckpt')
    psnrs = []; ts = []; imgs = []; depths = []
    os.makedirs(f'{path_prefix}results/{dataset_name}/{exp_name}_traj/', exist_ok=True)
    # print(len(dataset))
    directions = []
    poses = []
    for img_idx in range(len(dataset)):
        print(f'index at {img_idx}')
        t = time.time()
        rays_o, rays_d = get_rays(dataset.directions.cuda(), dataset[img_idx]['pose'].cuda())
        results = render(model, rays_o, rays_d,
                        **{'test_time': True,
                            'T_threshold': 1e-2,
                            'exp_step_factor': 1/256})
        torch.cuda.synchronize()
        ts += [time.time()-t]

        pred = results['rgb'].reshape(dataset.img_wh[1], dataset.img_wh[0], 3).cpu().numpy()
        pred = (pred*255).astype(np.uint8)
        depth = results['depth'].reshape(dataset.img_wh[1], dataset.img_wh[0]).cpu().numpy()
        depth_ = depth2img(depth)
        imgs += [pred]
        depths += [depth_]
        # rgb_write_path = f'{path_prefix}results/{dataset_name}/{exp_name}_traj/{img_idx:03d}.png'
        # print(f'rgb write path is {rgb_write_path}')
        imageio.imwrite(f'{path_prefix}results/{dataset_name}/{exp_name}_traj/{img_idx:03d}.png', pred)
        imageio.imwrite(f'{path_prefix}results/{dataset_name}/{exp_name}_traj/{img_idx:03d}_d.png', depth_)

        if dataset.split != 'test_traj':
            rgb_gt = dataset[img_idx]['rgb'].cuda()
            psnrs += [psnr(results['rgb'], rgb_gt).item()]
    frame_images += [imgs]
    frame_depths += [depths]
    if psnrs: print(f'mean PSNR: {np.mean(psnrs):.2f}, min: {np.min(psnrs):.2f}, max: {np.max(psnrs):.2f}')
    print(f'mean time: {np.mean(ts):.4f} s, FPS: {1/np.mean(ts):.2f}')
    print(f'mean samples per ray: {results["total_samples"]/len(rays_d):.2f}')

out_imgs = []
out_depths = []
for img_idx in range(len(imgs) * len(frame_images)):
        out_imgs += [frame_images[img_idx % len(frame_images)][img_idx % len(imgs)]]
        out_depths += [frame_depths[img_idx % len(frame_depths)][img_idx % len(depths)]]

if len(out_imgs) > 1:
    print('in video saving branch')
    print(f'concatenating {len(out_imgs)} images')
    os.makedirs(f'{path_prefix}results/{dataset_name}/boxing_traj/', exist_ok=True)
    imageio.mimsave(f'{path_prefix}results/{dataset_name}/boxing_traj/rgb.mp4', out_imgs, fps=24)
    imageio.mimsave(f'{path_prefix}results/{dataset_name}/boxing_traj/depth.mp4', out_depths, fps=24)
    print('branch complete')
