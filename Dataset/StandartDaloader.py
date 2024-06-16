import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, RandomVerticalFlip

from Dataset.AerialDataset import AerialDataset
from utils.DDP_utils import prepare_data


def setUpDataloaders(config, dataset_root):
    lr_size = config['lr_size']
    hr_size = config['hr_size']

    transforms = Compose([
        RandomHorizontalFlip(0.2),
        RandomVerticalFlip(0.2)]
    )

    dataset = AerialDataset(dataset_root, lr_size, hr_size, data_augmentation=transforms, aux_sat_prob=0,
                            sat_dataset_path=dataset_root + '\\satelite_dataset', ) #Desactivamos fotos de satelite de momento
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.6, 0.2, 0.2],
                                                            generator=torch.Generator().manual_seed(420))
    if config['num_workers'] > 1:
        train_dataloader = prepare_data(train_dataset, config['batch_size'], torch.cuda.device_count())
        val_dataloader = prepare_data(val_dataset, config['batch_size'], torch.cuda.device_count())
        test_dataloader = prepare_data(test_dataset, config['batch_size'], torch.cuda.device_count())
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader
