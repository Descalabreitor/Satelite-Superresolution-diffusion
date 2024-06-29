import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from tqdm import tqdm

from utils.model_utils import save_model, load_model
from utils.tensor_utils import move_to_cuda, tensors_to_scalars


class Trainer:

    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def save_model(self, epoch):
        save_model(self.model, f"{self.hyperparams["model_name"]} Epoch{epoch}.pt", self.hyperparams["save_dir"])

    def set_model(self, model):
        self.model = model

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_dataloaders(self, train_dataloader, val_dataloader, test_dataloader):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def train_epoch(self, epoch):
        final_loss = 0.0
        train_pbar = tqdm(self.train_dataloader, initial=0, total=len(self.train_dataloader), dynamic_ncols=True,
                          unit='batch')
        for batch in train_pbar:
            self.model.train()
            move_to_cuda(batch, self.hyperparams["device"])
            losses = self.training_step(batch)
            total_loss = sum(losses.values())

            if self.hyperparams["grad_acum"] > 0:
                if epoch % self.hyperparams["grad_acum"] == 0:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                else:
                    total_loss.backward()
            else:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            final_loss += total_loss
            train_pbar.set_postfix(**tensors_to_scalars(losses))

        return final_loss / len(self.train_dataloader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        final_loss = 0.0
        val_pbar = tqdm(self.val_dataloader, initial=0, total=len(self.val_dataloader), dynamic_ncols=True,
                        unit='batch')

        for batch in val_pbar:
            move_to_cuda(batch, self.hyperparams["device"])
            losses = self.training_step(batch)
            total_loss = sum(losses.values())
            val_pbar.set_postfix(**tensors_to_scalars(losses))
            final_loss += total_loss

        return final_loss / len(self.val_dataloader)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        all_metrics = {metric: 0 for metric in self.hyperparams["metrics_used"]}
        test_pbar = tqdm(self.test_dataloader, initial=0, dynamic_ncols=True, unit='batch')
        for batch in test_pbar:
            move_to_cuda(batch, self.hyperparams["device"])
            _, metrics = self.sample_test(batch)
            for metric in self.hyperparams["metrics_used"]:
                all_metrics[metric] += metrics[metric]
            test_pbar.set_postfix(**tensors_to_scalars(metrics))
        return {metric: value / len(self.test_dataloader) for metric, value in all_metrics.items()}

    @torch.no_grad()
    def sample_visualization(self, n_images):
        self.model.eval()
        images = self.test_dataloader[0][:n_images - 1]
        sr = self.sample_test(images, get_metrics=False)
        return sr

    def get_metrics(self, img_sr=None, img_hr=None):
        metrics = {k: 0 for k in ("psnr", "ssim")}
        ssim = StructuralSimilarityIndexMeasure().to(device=self.hyperparams["device"])
        psnr = PeakSignalNoiseRatio().to(device=self.hyperparams["device"])
        metrics['psnr'] = psnr(img_sr, img_hr)
        metrics['ssim'] = ssim(img_sr, img_hr)
        return img_sr, metrics

    @torch.no_grad()
    def sample_test(self, batch, get_metrics=True):
        raise NotImplementedError

    def training_step(self, batch):
        raise NotImplementedError
