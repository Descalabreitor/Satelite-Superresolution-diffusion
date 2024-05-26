import numpy as np
import torch
from tqdm import tqdm

import utils.model_utils
from utils.model_utils import *
from utils.tensor_utils import *
from utils.logger_utils import *
from utils.metrics_utils import *


class SR3Trainer:
    def __init__(self, metrics_used: tuple, model_name: str):
        self.scheduler = None
        self.optimizer = None
        self.model = None
        self.metrics_used = metrics_used
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def set_model(self, model: torch.nn.Module):
        self.model = model

    def set_scheduler(self, scheduler: torch.optim.lr_scheduler):
        self.scheduler = scheduler

    def train(self, train_dataloader):
        final_loss = 0.0
        train_pbar = tqdm(train_dataloader, initial=0, total=len(train_dataloader), dynamic_ncols=True, unit='batch')
        for batch in train_pbar:
            self.model.train()
            move_to_cuda(batch)
            loss = self.training_step(batch)
            self.optimizer.zero_grad()

            loss.backward()
            final_loss += loss
            self.optimizer.step()
            self.scheduler.step()
        return final_loss / len(train_dataloader)

    def save_model(self, save_dir: str):
        utils.model_utils.save_model(self.model, f"{self.model_name}.pt", save_dir)

    def validate(self, val_dataloader):
        self.model.eval()
        final_loss = 0.0
        val_pbar = tqdm(val_dataloader, initial=0, total=len(val_dataloader), dynamic_ncols=True, unit='batch')

        for batch in val_pbar:
            move_to_cuda(batch)
            losses = self.training_step(batch)
            final_loss += losses

        return final_loss / len(val_pbar)

    def training_step(self, batch: dict):
        img_hr = batch['hr']
        img_lr = batch['lr']
        img_bicubic = batch['bicubic']
        loss = self.model(img_hr, img_bicubic)
        return loss

    @torch.no_grad()
    def test(self, test_dataloader) -> dict:
        self.model.eval()
        all_metrics = {metric: 0 for metric in self.metrics_used}

        test_pbar = tqdm(test_dataloader, initial=0, dynamic_ncols=True, unit='batch')
        for batch in test_pbar:
            move_to_cuda(batch)
            _, metrics = self.sample_test(batch)
            for metric in self.metrics_used:
                all_metrics[metric] += metrics[metric] / metrics["n_samples"]
        return {metric: value / len(test_dataloader) for metric, value in all_metrics.items()}

    @torch.no_grad()
    def sample_test(self, batch: dict) -> (torch.Tensor, dict):
        metrics = {k: 0 for k in self.metrics_used}
        metrics['n_samples'] = 0
        img_hr = batch['hr']
        img_bicubic = batch['bicubic']
        img_sr = self.model.sample(img_bicubic)
        for b in range(img_sr.shape[0]):
            metrics['n_samples'] += 1
            ssim = calculate_ssim(tensor2img(img_sr[b]), tensor2img(img_hr[b]))
            psnr = calculate_psnr(tensor2img(img_sr[b]), tensor2img(img_hr[b]))
            metrics['ssim'] += ssim
            metrics['psnr'] += psnr

        return img_sr, metrics

    @torch.no_grad()
    def upscale_img(self, bicubic_img: torch.Tensor) -> torch.Tensor:
        img_sr = self.model.sample(bicubic_img)
        return img_sr
