from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

import utils.model_utils
from tasks.trainers.Trainer import Trainer
from utils.tensor_utils import *
from utils.logger_utils import *


class SRDiffTrainer(Trainer):
    def __init__(self, metrics_used, model_name, device, use_rrdb=True, fix_rrdb=False, aux_l1_loss=True,
                 aux_ssim_loss=False, aux_perceptual_loss=False, grad_acum=0, save_dir="."):
        super().__init__(device=device, model_name=model_name, metrics_used=metrics_used, save_dir=save_dir)
        self.grad_acum = grad_acum
        self.example_image = None
        self.aux_l1_loss = aux_l1_loss
        self.aux_ssim_loss = aux_ssim_loss
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

    def training_step(self, batch):
        img_hr = batch['hr']
        img_lr = batch['lr']
        img_bicubic = batch['bicubic']
        losses, _, _ = self.model(img_hr, img_lr, img_bicubic, use_rrdb=self.use_rrdb, fix_rrdb=self.fix_rrdb,
                                  aux_ssim_loss=self.aux_ssim_loss, aux_l1_loss=self.aux_l1_loss,
                                  aux_percep_loss=self.aux_perceptual_loss)
        total_loss = sum(losses.values())
        return losses, total_loss

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
