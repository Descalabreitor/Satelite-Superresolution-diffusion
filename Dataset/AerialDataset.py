import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import PIL.Image


class AerialDataset(Dataset):
    def __init__(self, dataset_path, lr_size, hr_size, data_augmentation = None):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(dataset_path)

        self.data_augmentation = data_augmentation

        self.dataset_path = dataset_path
        lr_dir = os.path.join(dataset_path, str(lr_size))
        hr_dir = os.path.join(dataset_path, str(hr_size))
        bicubic_dir = os.path.join(dataset_path, os.path.join("sr", str(lr_size) + "_" + str(hr_size)))

        self.low_res_images = [os.path.join(lr_dir, image) for image in os.listdir(lr_dir)]
        self.high_res_images = [os.path.join(hr_dir, image) for image in os.listdir(hr_dir)]
        self.sr_res_images = [os.path.join(bicubic_dir, image) for image in os.listdir(bicubic_dir)]

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):
        lr_image = PIL.Image.open(self.low_res_images[idx])
        hr_image = PIL.Image.open(self.high_res_images[idx])
        bicubic_image = PIL.Image.open(self.sr_res_images[idx])

        lr_image = transforms.ToTensor()(lr_image)
        hr_image = transforms.ToTensor()(hr_image)
        if self.data_augmentation:
            bicubic_image = self.data_augmentation(transforms.ToTensor()(bicubic_image))
        else:
            bicubic_image = transforms.ToTensor()(bicubic_image)

        return {'bicubic': bicubic_image, 'hr': hr_image, 'lr': lr_image}

    def get_image_from_name(self, name):
        idx = self.get_idx_from_name(name)
        return self.__getitem__(idx)

    def get_idx_from_name(self, name):
        for idx, image in enumerate(self.low_res_images):
            if name in image:
                return idx