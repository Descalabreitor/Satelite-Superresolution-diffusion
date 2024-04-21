import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.checkpoint_utils import *
from utils.tensor_utils import *
from utils.logger_utils import *
from utils.metrics_utils import *


class Trainer:
    def __init__(self, logs_dir, metrics_used):
        self.scheduler = None
        self.optimizer = None
        self.model = None
        self.metrics_used = metrics_used
        self.logger = SummaryWriter(logs_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_model(self, model):
        self.model = model

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def train(self, train_dataloader, val_dataloader, save_interval, max_steps, checkpoints_dir=None):
        global_step = load_checkpoint(self.model, self.optimizer, checkpoints_dir)
        train_pbar = tqdm(train_dataloader, initial=global_step, total=float('inf'), dynamic_ncols=True, unit='step')
        while global_step < max_steps:
            for batch in train_pbar:
                if global_step % save_interval == 0 and global_step != 0:
                    with torch.no_grad():
                        self.model.eval()
                        self.validate(val_dataloader, global_step)
                    save_checkpoint(self.model, self.optimizer, checkpoints_dir, global_step, 10)
                self.model.train()
                move_to_cuda(batch)
                losses, total_loss = self.training_step(batch)
                self.optimizer.zero_grad()

                total_loss.backward()
                self.optimizer.step()
                global_step += 1
                self.scheduler.step()
                if global_step % save_interval == 0:
                    log_metrics(self.logger, {f'tr/{k}': v for k, v in losses.items()}, global_step)
                train_pbar.set_postfix(**tensors_to_scalars(losses))

    def validate(self, val_loader, global_step):
        for batch in val_loader:
            move_to_cuda(batch)
            img, rrdb_out, ret = self.sample_test(batch)
            metrics = {}
            metrics.update({k: np.mean(ret[k]) for k in self.metrics_used})
            pbar.set_postfix(**tensors_to_scalars(metrics))
            print('Val results:', metrics)
            log_metrics(self.logger, {f'val/{k}': v for k, v in metrics.items()}, global_step)

    #    def test(self, test_dataloader, global_step, checkpoints_dir=None):
    #        load_checkpoint(self.model, self.optimizer, checkpoints_dir=checkpoints_dir)
    #        with torch.no_grad():
    #self.model.eval()
    #pbar = tqdm(enumerate(test_dataloader),total = len(test_dataloader))
    #for batch_idx, batch in pbar:
    #    batch.to(self.device)
    #    img, rrdb_out, ret = self.sample_test(batch)
    #    metrics = {}
    #    metrics.update({k: np.mean(ret[k]) for k in self.metric_keys})
    #    pbar.set_postfix(**tensors_to_scalars(metrics))
    #    print('Test results:', metrics)
    #    log_metrics({f'test/{k}': v for k, v in metrics.items()}, global_step)
    #test_ func deactivated at the moment. Validation func with test dataloader will be used

    def training_step(self, batch):
        img_hr = batch['hr']
        img_lr = batch['lr']
        img_bicubic = batch['bicubic']
        losses, _, _ = self.model(img_hr, img_lr, img_bicubic, use_rrdb=True, fix_rrdb=True,
                                  aux_ssim_loss=False, aux_l1_loss=True, aux_percep_loss=False)
        total_loss = list(np.sum(losses.values()))[0]
        return losses, total_loss

    def sample_test(self, batch):
        results = {k: 0 for k in self.metrics_used}
        results['n_samples'] = 0
        img_hr = batch['hr']
        img_lr = batch['lr']
        img_bicubic = batch['bicubic']
        img_sr, rrdb_out = self.model.sample(img_lr, img_bicubic, img_hr.shape, True)
        for b in range(img_sr.shape[0]):
            results['n_samples'] += 1
            ssim = calculate_ssim(tensor2img(img_sr[b]), tensor2img(img_hr[b]))
            psnr = calculate_psnr(tensor2img(img_sr[b]), tensor2img(img_hr[b]))
            results['ssim'] += ssim
            results['psnr'] += psnr

        return img_sr, rrdb_out, results
