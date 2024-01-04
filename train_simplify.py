import torch
from opt import get_opts
import os
import glob
import imageio
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils import slim_ckpt
from train import NeRFSystem
import time

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    from tqdm import tqdm
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    device = torch.device("cuda")
    system.setup(None)
    (opts, ), (net_sch, ) = system.configure_optimizers()
    system.to(device)
    system.on_train_start()
    system.train()
    for epoch in range(hparams.num_epochs):
        progress_bar = tqdm(system.train_dataloader())
        for batch in progress_bar:
            batch['rgb'] = batch['rgb'].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                stime = time.time()
                loss = system.training_step(batch, None)
                etime = time.time()
                opts.zero_grad()
                loss.backward()
            progress_bar.set_description(f"loss {loss} time {etime - stime}")
        net_sch.step()

    if not hparams.val_only:  # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
                      save_poses=hparams.optimize_ext)
        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')

    if (not hparams.no_save_test) and \
       hparams.dataset_name == 'nsvf' and \
       'Synthetic' in hparams.root_dir:  # save video
        imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
                        [imageio.imread(img) for img in imgs[::2]],
                        fps=30, macro_block_size=1)
        imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
                        [imageio.imread(img) for img in imgs[1::2]],
                        fps=30, macro_block_size=1)
