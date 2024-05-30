import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

import utils.model_utils
from utils.model_utils import *
from utils.tensor_utils import *
from utils.logger_utils import *


class SRDiffTrainer:
    def __init__(self, metrics_used, model_name, device, use_rrdb=True, fix_rrdb=False,
                 aux_l1_loss=True, aux_ssim_loss=False, aux_perceptual_loss=False):
        self.scheduler = None
        self.optimizer = None
        self.model = None
        self.metrics_used = metrics_used
        self.device = device
        self.model_name = model_name
        self.example_image = None
        self.aux_l1_loss = aux_l1_loss
        self.aux_ssim_loss = aux_ssim_loss
        self.aux_perceptual_loss = aux_perceptual_loss
        self.use_rrdb = use_rrdb
        self.fix_rrdb = fix_rrdb

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_model(self, model):
        self.model = model

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def train(self, train_dataloader):
        final_loss = 0.0
        train_pbar = tqdm(train_dataloader, initial=0, total=len(train_dataloader), dynamic_ncols=True, unit='batch')
        for batch in train_pbar:
            self.model.train()
            move_to_cuda(batch, self.device)
            losses, total_loss = self.training_step(batch, self.aux_ssim_loss, self.aux_perceptual_loss, self.use_rrdb,
                                                    self.fix_rrdb)
            self.optimizer.zero_grad()

            total_loss.backward()
            final_loss += total_loss
            self.optimizer.step()
            self.scheduler.step()
            train_pbar.set_postfix(**tensors_to_scalars(losses))
        return final_loss / len(train_dataloader)

    def save_model(self, save_dir):
        utils.model_utils.save_model(self.model, f"{self.model_name}.pt", save_dir)

    def validate(self, val_loader):
        self.model.eval()
        final_loss = 0.0
        val_pbar = tqdm(val_loader, initial=0, total=len(val_loader), dynamic_ncols=True, unit='batch')

        for batch in val_pbar:
            move_to_cuda(batch, self.device)
            losses, total_loss = self.training_step(batch, self.aux_ssim_loss, self.aux_perceptual_loss, self.use_rrdb,
                                                    self.fix_rrdb)
            val_pbar.set_postfix(**tensors_to_scalars(losses))
            final_loss += total_loss

        return final_loss / len(val_pbar)

    def training_step(self, batch, aux_ssim_loss, aux_perceptual_loss, use_rrdb, fix_rrdb):
        img_hr = batch['hr']
        img_lr = batch['lr']
        img_bicubic = batch['bicubic']
        losses, _, _ = self.model(img_hr, img_lr, img_bicubic, use_rrdb=use_rrdb, fix_rrdb=fix_rrdb,
                                  aux_ssim_loss=aux_ssim_loss, aux_l1_loss=self.aux_l1_loss, aux_percep_loss=aux_perceptual_loss)
        total_loss = list(np.sum(losses.values()))[0]
        return losses, total_loss

    @torch.no_grad()
    def test(self, test_dataloader):
        self.model.eval()
        all_metrics = {metric: 0 for metric in self.metrics_used}
        sr_images = []
        test_pbar = tqdm(test_dataloader, initial=0, dynamic_ncols=True, unit='batch')
        for batch in test_pbar:
            move_to_cuda(batch, self.device)
            _, _, metrics = self.sample_test(batch)
            for metric in self.metrics_used:
                all_metrics[metric] += metrics[metric]
            test_pbar.set_postfix(**tensors_to_scalars(metrics))
        return {metric: value / len(test_dataloader) for metric, value in all_metrics.items()}

    @torch.no_grad()
    def sample_test(self, batch):
        metrics = {k: 0 for k in self.metrics_used}
        img_hr = batch['hr']
        img_lr = batch['lr']
        img_bicubic = batch['bicubic']
        img_sr, rrdb_out = self.model.sample(img_lr, img_bicubic, img_hr.shape, use_rrdb=self.use_rrdb)
        ssim = StructuralSimilarityIndexMeasure().to(device=self.device)
        psnr = PeakSignalNoiseRatio().to(device=self.device)
        metrics['psnr'] = psnr(img_sr, img_hr)
        metrics['ssim'] = ssim(img_sr, img_hr)
        return img_sr, rrdb_out, metrics
