import numpy as np
import torch
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import utils.model_utils
from tasks.trainers.Trainer import Trainer
from utils.model_utils import *
from utils.tensor_utils import *
from utils.logger_utils import *
#from utils.metrics_utils import *


class SR3Trainer(Trainer):
    """"
    Trainer for SR3 models
    hyperparams needed:
    - metrics_used
    - model_name
    - grad_acum if none put 0
    - device
    - save dir
    """
    def __init__(self, hyperparams):
        super().__init__(hyperparams)

    def training_step(self, batch):
        img_hr = batch['hr']
        img_lr = batch['lr']
        img_bicubic = batch['bicubic']
        loss = self.model(img_hr, img_bicubic)
        return {"standart loss": loss}

    @torch.no_grad()
    def sample_test(self, batch, get_metrics=True):

        img_hr = batch['hr']
        img_bicubic = batch['bicubic']
        img_sr = self.model.sample(img_bicubic)
        if get_metrics:
            return self.get_metrics(img_sr, img_hr)
        else:
            return img_sr
