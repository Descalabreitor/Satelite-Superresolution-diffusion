import numpy as np
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

import utils.model_utils
from tasks.trainers.SRDiffTrainer import SRDiffTrainer
from utils.tensor_utils import *
from utils.logger_utils import *


class SRDiffTrainerDDP(SRDiffTrainer):
    def __init__(self, metrics_used, model_name, gpu_id, device, use_rrdb=True, fix_rrdb=False, aux_l1_loss=True,
                 aux_ssim_loss=False, aux_perceptual_loss=False):
        super().__init__(metrics_used, model_name, device, use_rrdb, fix_rrdb, aux_l1_loss, aux_ssim_loss,
                         aux_perceptual_loss)
        self.scheduler = None
        self.optimizer = None
        self.model = None
        self.metrics_used = metrics_used
        self.device = gpu_id
        self.model_name = model_name
        self.example_image = None
        self.aux_l1_loss = aux_l1_loss
        self.aux_ssim_loss = aux_ssim_loss
        self.aux_ssim_loss = aux_ssim_loss
        self.aux_perceptual_loss = aux_perceptual_loss
        self.use_rrdb = use_rrdb
        self.fix_rrdb = fix_rrdb

    def set_model(self, model):
        self.model = DDP(model, device_ids=[self.device])

    def train(self, train_dataloader, epoch):
        final_loss = 0.0
        train_dataloader.sampler.set_epoch(epoch)
        print(f"[GPU{self.device}] Epoch {epoch} Steps: {len(self.train_dataloader)}")
        for batch in train_dataloader:
            self.model.train()
            move_to_cuda(batch, self.device)
            losses, total_loss = self.training_step(batch, self.aux_ssim_loss, self.aux_perceptual_loss, self.use_rrdb,
                                                    self.fix_rrdb)
            self.optimizer.zero_grad()

            total_loss.backward()
            final_loss += total_loss
            self.optimizer.step()
            self.scheduler.step()

        return final_loss / len(train_dataloader)

    def save_model(self, save_dir):
        utils.model_utils.save_model(self.model.module, f"{self.model_name}.pt", save_dir)

    def validate(self, val_loader, epoch):
        self.model.eval()
        final_loss = 0.0
        val_loader.sampler.set_epoch(epoch)
        for batch in val_loader:
            move_to_cuda(batch, self.device)
            losses, total_loss = self.training_step(batch, self.aux_ssim_loss, self.aux_perceptual_loss, self.use_rrdb,
                                                    self.fix_rrdb)
            final_loss += total_loss

        return final_loss / len(val_loader)

    @torch.no_grad()
    def test(self, test_dataloader, epoch):
        self.model.eval()
        all_metrics = {metric: 0 for metric in self.metrics_used}
        sr_images = []
        test_dataloader.sampler.set_epoch(epoch)
        for batch in test_dataloader:
            move_to_cuda(batch, self.device)
            _, _, metrics = self.sample_test(batch)
            for metric in self.metrics_used:
                all_metrics[metric] += metrics[metric]
        return {metric: value / len(test_dataloader) for metric, value in all_metrics.items()}

    def get_model(self):
        return self.model
