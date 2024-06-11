import torch
from tqdm import tqdm

from utils.model_utils import save_model, load_model
from utils.tensor_utils import move_to_cuda, tensors_to_scalars


class Trainer:

    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        self.device = self.hyperparams["device"]
        self.metrics_used = self.hyperparams["metrics_used"]
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.model_name = self.hyperparams["model_name"]
        self.save_dir = self.hyperparams["save_dir"]

    def save_model(self):
        save_model(self.model, f"{self.model_name}.pt", self.save_dir)

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

    def train_epoch(self):
        final_loss = 0.0
        train_pbar = tqdm(self.train_dataloader, initial=0, total=len(self.train_dataloader), dynamic_ncols=True,
                          unit='batch')
        for batch in train_pbar:
            self.model.train()
            move_to_cuda(batch, self.device)
            losses = self.training_step(batch)
            total_loss = sum(losses.values())
            self.optimizer.zero_grad()
            total_loss.backward()
            final_loss += total_loss
            self.optimizer.step()
            self.scheduler.step()
            train_pbar.set_postfix(**tensors_to_scalars(losses))

        return final_loss / len(self.train_dataloader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        final_loss = 0.0
        val_pbar = tqdm(self.val_dataloader, initial=0, total=len(self.val_dataloader), dynamic_ncols=True,
                        unit='batch')

        for batch in val_pbar:
            move_to_cuda(batch, self.device)
            losses, total_loss = self.training_step(batch)
            val_pbar.set_postfix(**tensors_to_scalars(losses))
            final_loss += total_loss

        return final_loss / len(val_pbar)

    @torch.no_grad()
    def test(self, test_dataloader):
        self.model.eval()
        all_metrics = {metric: 0 for metric in self.metrics_used}
        test_pbar = tqdm(test_dataloader, initial=0, dynamic_ncols=True, unit='batch')
        for batch in test_pbar:
            move_to_cuda(batch, self.device)
            _, metrics = self.sample_test(batch)
            for metric in self.metrics_used:
                all_metrics[metric] += metrics[metric]
            test_pbar.set_postfix(**tensors_to_scalars(metrics))
        return {metric: value / len(test_dataloader) for metric, value in all_metrics.items()}

    @torch.no_grad()
    def sample_visualization(self, n_images):
        self.model.eval()
        images = self.test_dataloader[0][:n_images - 1]
        sr = self.sample_test(images, get_metrics=False)
        return sr

    @torch.no_grad()
    def sample_test(self, batch, get_metrics=True):
        raise NotImplementedError

    def training_step(self, batch):
        raise NotImplementedError
