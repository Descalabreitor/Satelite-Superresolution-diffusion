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

def set_up_data(hyperparams, dataset_dir, sat_dataset_dir, world_size):
    lr_size = 64
    hr_size = 256
    dataset_dir = dataset_dir

    transforms = Compose(
        [RandomApply(transforms=[GaussianBlur(7)], p=0.5),
         RandomEqualize()]
    )

    dataset = AerialDataset(dataset_dir, lr_size, hr_size, data_augmentation=None, aux_sat_prob=0.4,
                            sat_dataset_path=sat_dataset_dir)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.6, 0.2, 0.2],
                                                            generator=torch.Generator().manual_seed(420))

    train_dataloader = ddp_utils.prepare_data(train_dataset, hyperparams['batch_size'], world_size)
    val_dataloader = ddp_utils.prepare_data(val_dataset, hyperparams['batch_size'], world_size)
    test_dataloader = ddp_utils.prepare_data(test_dataset, hyperparams['batch_size'], world_size)

    return train_dataloader, val_dataloader, test_dataloader

