import os

import torch
from torch.distributed import init_process_group
from torch.utils.data import DataLoader, DistributedSampler


def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    init_process_group('gloo', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def prepare_data(dataset, batch_size, n_workers=2):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=n_workers,
        sampler=DistributedSampler(dataset)
    )
