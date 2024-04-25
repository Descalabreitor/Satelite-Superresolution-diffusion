import glob
import os
import re
import subprocess

import torch


def get_last_checkpoint(checkpoints_dir, steps=None):
    checkpoint = None
    last_ckpt_path = None
    ckpt_paths = get_all_ckpts(checkpoints_dir, steps)
    if len(ckpt_paths) > 0:
        last_ckpt_path = ckpt_paths[0]
        checkpoint = torch.load(last_ckpt_path, map_location='cpu')
    return checkpoint, last_ckpt_path


def get_all_ckpts(checkpoints_dir, steps=None):
    if steps is None:
        ckpt_path_pattern = f'{checkpoints_dir}/model_ckpt_steps_*.ckpt'
    else:
        ckpt_path_pattern = f'{checkpoints_dir}/model_ckpt_steps_{steps}.ckpt'
    return sorted(glob.glob(ckpt_path_pattern),
                  key=lambda x: -int(re.findall('.*steps_(\d+)\.ckpt', x)[0]))


def load_checkpoint(model, optimizer, checkpoints_dir):
    checkpoint, _ = get_last_checkpoint(checkpoints_dir)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict']['model'])
        model.cuda()
        optimizer.load_state_dict(checkpoint['optimizer_states'][0])
        training_step = checkpoint['global_step']
        del checkpoint
        torch.cuda.empty_cache()
    else:
        training_step = 0
        model.cuda()
    return training_step

def save_checkpoint(model, optimizer, work_dir, global_step):
    ckpt_path = f'{work_dir}\\model_ckpt_steps_{global_step}.ckpt'
    print(f'Step@{global_step}: saving model to {ckpt_path}')
    checkpoint = {'global_step': global_step}
    optimizer_states = []
    optimizer_states.append(optimizer.state_dict())
    checkpoint['optimizer_states'] = optimizer_states
    checkpoint['state_dict'] = {'model': model.state_dict()}
    torch.save(checkpoint, ckpt_path, _use_new_zipfile_serialization=False)


