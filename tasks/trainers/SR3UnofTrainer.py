import torch

from tasks.trainers.Trainer import Trainer


class SR3UnofTrainer(Trainer):
    def __init__(self, hyperparams):
        super().__init__(hyperparams)

    @torch.no_grad()
    def sample_test(self, batch, get_metrics=True):
        with torch.no_grad():
            output = self.model.super_resolution(batch['bicubic'])

            if get_metrics:
                output, metrics = self.get_metrics(output, batch['hr'])
                return output, metrics
            else:
                return output

    def training_step(self, batch):
        return {"p_loss": self.model(batch)}
