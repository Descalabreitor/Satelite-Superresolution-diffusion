import os

import torch
from torch.distributed import init_process_group


def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)