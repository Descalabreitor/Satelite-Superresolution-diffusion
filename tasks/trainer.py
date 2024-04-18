import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.checkpoint_utils import *
from utils.tensor_utils import tensors_to_scalars
from utils.logger_utils import *

class Trainer:
    def __init__(self, logs_dir):
        self.scheduler = None
        self.optimizer = None
        self.model = None
        self.logger = SummaryWriter(logs_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_optimizer(self, optimizer, params, lr=1e-3, momentum=0.9):
        raise NotImplementedError
    def build_model(self, model):
        raise NotImplementedError

    def build_scheduler(self, scheduler_type, lr_decay=0.1, step_size=10):
        raise NotImplementedError

    def train(self, train_dataloader, val_dataloader, save_interval, max_steps, checkpoints_dir=None):
        global_step = load_checkpoint(self.model, self.optimizer, checkpoints_dir)
        train_pbar = tqdm(train_dataloader, initial=global_step, total=float('inf'), dynamic_ncols=True, unit='step')
        while global_step < max_steps:
            for batch in train_pbar:
                if global_step % save_interval == 0:
                    with torch.no_grad():
                        self.model.eval()
                        self.validate(val_dataloader, global_step)
                    save_checkpoint(self.model, self.optimizer, checkpoints_dir, global_step, 10)
                self.model.train()
                batch.to(self.device)
                losses, total_loss = self.training_step(batch)
                self.optimizer.zero_grad()

                total_loss.backward()
                self.optimizer.step()
                global_step += 1
                self.scheduler.step(global_step)
                if global_step % save_interval == 0:
                    log_metrics(self.logger, {f'tr/{k}': v for k, v in losses.items()}, global_step)
                train_pbar.set_postfix(**tensors_to_scalars(losses))



    def validate(self, val_loader, global_step):
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), dynamic_ncols=True, unit='step')
        for batch_idx, batch in pbar:
            batch.to(self.device)
            img, rrdb_out, ret = self.sample_test(batch)
            metrics = {}
            metrics.update({k: np.mean(ret[k]) for k in self.metric_keys})
            pbar.set_postfix(**tensors_to_scalars(metrics))
            print('Val results:', metrics)
            log_metrics({f'val/{k}': v for k, v in metrics.items()}, global_step)

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
        raise NotImplementedError

    def sample_test(self, batch):
        raise NotImplementedError
