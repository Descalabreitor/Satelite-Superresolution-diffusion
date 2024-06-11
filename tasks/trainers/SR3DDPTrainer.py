from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import utils.model_utils
from tasks.trainers.SR3Trainer import SR3Trainer
from utils.tensor_utils import *


class SR3DDPTrainer(SR3Trainer):
    """"
    Trainer for SR3 with DDP support
    """

    def __init__(self, hyperparams):
        super().__init__(hyperparams)

    def set_model(self, model):
        model = model.to(self.hyperparams["device"])
        self.model = DDP(model, device_ids=[self.hyperparams["device"]])

    def train_epoch(self, epoch):
        final_loss = 0.0
        self.train_dataloader.sampler.set_epoch(epoch)
        print(f"[GPU{self.hyperparams["device"]}] Epoch {epoch}]")
        train_pbar = tqdm(self.train_dataloader, initial=0, total=len(self.train_dataloader), dynamic_ncols=True,
                          unit='batch')
        for batch in train_pbar:
            self.model.train()
            move_to_cuda(batch, self.hyperparams["device"])
            losses = self.training_step(batch)
            total_loss = sum(losses.values)
            self.optimizer.zero_grad()

            total_loss.backward()
            final_loss += total_loss
            self.optimizer.step()
            train_pbar.set_postfix(**tensors_to_scalars(losses))
        return final_loss / len(self.train_dataloader)

    def save_model(self, epoch):
        utils.model_utils.save_model(self.model.module, f"{self.hyperparams["model_name"]}{epoch}.pt", self.hyperparams["save_dir"])

    def validate(self, epoch):
        self.model.eval()
        final_loss = 0.0
        self.val_dataloader.sampler.set_epoch(epoch)

        val_pbar = tqdm(self.val_dataloader, initial=0, total=len(self.val_dataloader), dynamic_ncols=True,
                        unit='batch')
        for batch in val_pbar:
            move_to_cuda(batch, self.hyperparams["device"])
            losses = self.training_step(batch)
            total_loss = sum(losses.values)
            final_loss += total_loss
            val_pbar.set_postfix(**tensors_to_scalars(losses))
        self.scheduler.step(final_loss / len(self.val_dataloader))

        return final_loss / len(self.val_dataloader)

    def get_model(self):
        return self.model
