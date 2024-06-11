from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

import utils.model_utils
from tasks.trainers.Trainer import Trainer
from utils.tensor_utils import *
from utils.logger_utils import *


class SRDiffTrainer(Trainer):
    """"
    Trainer for SRDiff Models:
    Hyperparams needed:
    - metrics_used
    - model_named
    - device
    - use_rrdb
    - fix_rrfb
    - aux_l1_loss
    - aux_ssim_loss
    - aux_perceptual_loss
    - grad_acum if none must be 0
    - save_dir
    """
    def __init__(self, hyperparams):
        super().__init__(hyperparams)

    def training_step(self, batch):
        img_hr = batch['hr']
        img_lr = batch['lr']
        img_bicubic = batch['bicubic']
        losses, _, _ = self.model(img_hr, img_lr, img_bicubic, use_rrdb=self.hyperparams["use_rrdb"], fix_rrdb=self.hyperparams["fix_rrdb"],
                                  aux_ssim_loss=self.hyperparams["aux_ssim_loss"], aux_l1_loss=self.hyperparams["aux_l1_loss"],
                                  aux_percep_loss=self.hyperparams["aux_perceptual_loss"])
        total_loss = sum(losses.values())
        return losses, total_loss

    @torch.no_grad()
    def sample_test(self, batch, get_metrics=True):
        metrics = {k: 0 for k in self.hyperparams["metrics_used"]}
        img_hr = batch['hr']
        img_lr = batch['lr']
        img_bicubic = batch['bicubic']
        img_sr, rrdb_out = self.model.sample(img_lr, img_bicubic, img_hr.shape, use_rrdb=self.hyperparams["use_rrdb"])
        if get_metrics:
            ssim = StructuralSimilarityIndexMeasure().to(device=self.hyperparams["device"])
            psnr = PeakSignalNoiseRatio().to(device=self.hyperparams["device"])
            metrics['psnr'] = psnr(img_sr, img_hr)
            metrics['ssim'] = ssim(img_sr, img_hr)
            return img_sr, metrics
        else:
            return img_sr
