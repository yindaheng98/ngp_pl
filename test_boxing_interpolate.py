import torch
import time
import os
import numpy as np
from models.networks import NGP
from models.rendering import render
from datasets import dataset_dict
from datasets.ray_utils import get_rays
from utils import load_ckpt
from train import depth2img
import imageio
import tqdm

def export_video(path_prefix, dataset_name, imgs, depths):
    print(f'concatenating {len(imgs)} images')
    os.makedirs(f'{path_prefix}results/{dataset_name}/boxing_traj/', exist_ok=True)
    imageio.mimsave(f'{path_prefix}results/{dataset_name}/boxing_traj/rgb.mp4', imgs, fps=24)
    imageio.mimsave(f'{path_prefix}results/{dataset_name}/boxing_traj/depth.mp4', depths, fps=24)
    print('video saved')

def remove_imgs(path_prefix, dataset_name, exp_names, frame_indexes):
    for frame_index in frame_indexes:
        exp_name = exp_names.format(frame_index)
        traj_path = f'{path_prefix}results/{dataset_name}/{exp_name}_traj/'
        for file in os.listdir(traj_path):
            # print(f'removing file {traj_path + file}')
            os.remove(traj_path + file)
        os.rmdir(traj_path)
    print('intermediate image files removed')


def interpolate_pose(dataset, interpolate_ratio):
    dataset_len = len(dataset)
    prev_index_candidate = int(np.ceil(interpolate_ratio * dataset_len )) - 2
    # print(f'candidate is {prev_index_candidate}')
    prev_idx = prev_index_candidate if (prev_index_candidate + 1) / dataset_len <= interpolate_ratio else prev_index_candidate - 1
    prev = dataset[prev_idx]['pose']
    next = dataset[prev_idx + 1]['pose']
    next_ratio =  (prev_idx + 2) / dataset_len
    prev_ratio = (prev_idx + 1) / dataset_len
    prev_next_ratio = 1 - ((interpolate_ratio - prev_ratio) / (next_ratio - prev_ratio))
    # print(f"a = {prev_ratio}, b = {next_ratio}, x = {interpolate_ratio}")
    print(f'interpolating between prev index {prev_idx}, next index {prev_idx + 1} at ratio {prev_next_ratio}')
    return prev * prev_next_ratio + next * (1 - prev_next_ratio)

def interpolate_pose2(dataset, num_interpolations, interpolate_idx):

    def zero_div_mod(numerator, denominator):
        quotient = 0
        while numerator > 0 and numerator > denominator:
            numerator -= denominator
            quotient += 1
        return quotient, numerator

    prev_index, remainder = zero_div_mod(interpolate_idx + 1, num_interpolations)
    print(f'dividing {len(dataset)} by {prev_index}')
    prev_index, _ = zero_div_mod(prev_index, len(dataset))
    interpolate_ratio = 1 - (remainder / num_interpolations)
    print(f'interpolating between prev index {prev_index}, next index {prev_index + 1} at ratio {interpolate_ratio}')
    prev = dataset[prev_index]['pose']
    next = dataset[prev_index + 1]['pose']
    return prev * interpolate_ratio + next * (1 - interpolate_ratio)

def main():
    path_prefix = '/volume/'
    path_name = path_prefix + 'data/boxing/'
    dataset_name = 'colmap'
    exp_names = 'boxing{}'
    frame_names = 'frame{}'
    frame_indexes = [1, 2, 3, 4] # range(1, 3)
    interpolations_per_img = 35
    frame_count = 0
    total_interpolations = 0
    frame_poses = []
    ts = []; imgs = []; depths = []
    for frame_index in frame_indexes:
        interpolated_poses =[]
        exp_name = exp_names.format(frame_index)
        frame_name = frame_names.format(frame_index)

        dataset = dataset_dict[dataset_name](
            f'{path_name}{frame_name}',
            split='all', downsample=1.0 / 4
        )
        model = NGP(scale=0.5).cuda()
        load_ckpt(model, f'{path_prefix}ckpts/{dataset_name}/{exp_name}/epoch=19_slim.ckpt')

        os.makedirs(f'{path_prefix}results/{dataset_name}/{exp_name}_traj/', exist_ok=True)
        # print(len(dataset))

        directions = dataset.directions
        for interpolate_idx in range(interpolations_per_img):
            ratio = (total_interpolations + 1) / (interpolations_per_img * len(frame_indexes))
            print(f'ratio : {ratio}')
            interpolated_poses += [interpolate_pose(dataset, ratio)]
            print(f'len of pose is now {len(interpolated_poses)}')
            total_interpolations += 1
        # for interpolate_idx in range((len(dataset) - 1) * interpolations_per_img ):
        #     interpolated_poses += [interpolate_pose(dataset, interpolations_per_img, interpolate_idx)]

        print('interpolating between camera positions')
        for out_idx in tqdm.tqdm(range(len(interpolated_poses))):
            # print(f'interpolating pic {interpolate_idx}')
            t = time.time()
            rays_o, rays_d = get_rays(directions.cuda(),
                                      interpolated_poses[out_idx].cuda())
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
            imageio.imwrite(f'{path_prefix}results/{dataset_name}/{exp_name}_traj/{out_idx:03d}.png', pred)
            imageio.imwrite(f'{path_prefix}results/{dataset_name}/{exp_name}_traj/{out_idx:03d}_d.png', depth_)
        frame_count += 1
        frame_poses += interpolated_poses

    # out_imgs = []
    # out_depths = []
    # for img_idx in range(len(imgs) * len(frame_images)):
    #         out_imgs += [frame_images[img_idx % len(frame_images)][img_idx % len(imgs)]]
    #         out_depths += [frame_depths[img_idx % len(frame_depths)][img_idx % len(depths)]]

    export_video(path_prefix, dataset_name, imgs, depths)
    remove_imgs(path_prefix, dataset_name, exp_names, frame_indexes)

if __name__ == '__main__':
    main()