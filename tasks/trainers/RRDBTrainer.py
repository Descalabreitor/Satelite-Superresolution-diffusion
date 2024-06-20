import torch

from tasks.trainers.Trainer import Trainer


class RRDBTrainer(Trainer):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)

    def training_step(self, batch):
        losses = {}
        lr = batch['lr']
        hr = batch['hr']

        p = self.model(lr)
        losses["l1"] = torch.nn.functional.l1_loss(p, hr, reduction='mean')

        return losses

    def sample_test(self, batch, get_metrics=True):
        hr = batch['hr']
        lr = batch['lr']

        sr = self.model(lr)
        sr = sr.clamp(-1, 1)

        if get_metrics:
            _, metrics = self.get_metrics(sr, hr)
            return sr, metrics
        else:
            return sr
