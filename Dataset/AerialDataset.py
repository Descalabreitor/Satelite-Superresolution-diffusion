import random
from random import sample
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import PIL.Image


class AerialDataset(Dataset):
    def __init__(self, dataset_path, lr_size, hr_size, data_augmentation=None, aux_sat_prob=0,
                 sat_dataset_path=None):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(dataset_path)

        self.data_augmentation = data_augmentation
        self.dataset_path = dataset_path
        if aux_sat_prob > 0:
            self.aux_sat_images = [os.path.join(sat_dataset_path, image) for image in os.listdir(sat_dataset_path)]
        self.aux_sat_prob = aux_sat_prob
        lr_dir = os.path.join(dataset_path, str(lr_size))
        hr_dir = os.path.join(dataset_path, str(hr_size))
        bicubic_dir = os.path.join(dataset_path, str(lr_size) + "_" + str(hr_size))

        self.low_res_images = [os.path.join(lr_dir, image) for image in os.listdir(lr_dir)]
        self.high_res_images = [os.path.join(hr_dir, image) for image in os.listdir(hr_dir)]
        self.bicubic = [os.path.join(bicubic_dir, image) for image in os.listdir(bicubic_dir)]

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):

        lr_image = PIL.Image.open(self.low_res_images[idx])
        hr_image = PIL.Image.open(self.high_res_images[idx])
        if np.random.random() >= self.aux_sat_prob:
            bicubic_image = PIL.Image.open(self.bicubic[idx])
        else:
            bicubic_image = PIL.Image.open(self.aux_sat_images[(idx * 8) + np.random.randint(0, 7)])
            #Hay 8 imagenes satelitales por cada imagen normal. Multiplicamos por 8 para ir a la primera equivalente
            # y escogemos una aleatoria entre esas 8 sumando un número entre 0 y 7

        if self.data_augmentation:
            seed = np.random.randint(0, 2**10)
            random.seed(seed)
            lr_image = self.data_augmentation(lr_image)
            random.seed(seed)
            hr_image = self.data_augmentation(hr_image)
            random.seed(seed)
            bicubic_image = self.data_augmentation(bicubic_image)
        bicubic_image, hr_image, lr_image = self.__toTensors(bicubic_image, hr_image, lr_image)

        return {'bicubic': bicubic_image, 'hr': hr_image, 'lr': lr_image}

    @staticmethod
    def __toTensors(bicubic_image, hr_image, lr_image):
        hr_image = transforms.ToTensor()(hr_image)
        bicubic_image = transforms.ToTensor()(bicubic_image)
        lr_image = transforms.ToTensor()(lr_image)
        return bicubic_image, hr_image, lr_image

    def get_random_images(self, n_images):
        idxs = sample(range(len(self.low_res_images)), n_images)
        return [self.__getitem__(idx) for idx in idxs]

    def get_image_from_name(self, name):
        idx = self.get_idx_from_name(name)
        return self.__getitem__(idx)

    def get_idx_from_name(self, name):
        for idx, image in enumerate(self.low_res_images):
            if name in image:
                return idx
