import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import PIL.Image


class AerialDataset(Dataset):
    def __init__(self, dataset_path, lr_size, hr_size):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(dataset_path)

        self.dataset_path = dataset_path
        lr_dir = os.path.join(dataset_path, str(lr_size))
        hr_dir = os.path.join(dataset_path, str(hr_size))

        self.low_res_images = [os.path.join(lr_dir, image) for image in os.listdir(lr_dir)]
        self.high_res_images = [os.path.join(hr_dir, image) for image in os.listdir(hr_dir)]

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):
        lr_image = PIL.Image.open(self.low_res_images[idx])
        hr_image = PIL.Image.open(self.high_res_images[idx])

        lr_image = transforms.ToTensor()(lr_image)
        hr_image = transforms.ToTensor()(hr_image)

        return [lr_image, hr_image, sr_lr_image]
