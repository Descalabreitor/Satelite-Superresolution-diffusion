import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.checkpoint_utils import *

class Trainer:
    def __init__(self, logs_dir):
        self.scheduler = None
        self.optimizer = None
        self.model = None
        self.logger = SummaryWriter(logs_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_optimizer(self, optimizer, params, lr=1e-3, momentum=0.9):
        optimizers = {
            'adam': torch.optim.Adam(params, lr=lr),
            'sgd': torch.optim.SGD(params, lr=lr, momentum=momentum)
        }

        if optimizer not in optimizers.keys():
            raise ValueError(
                f"Optimizador '{optimizer}' not suported. Valid Options: {', '.join(optimizers.keys())}")

        self.optimizer = optimizers[optimizer]
        return self

    def set_model(self, model):
        self.model = model
        return self

    def set_scheduler(self, scheduler_type, lr_decay=0.1, step_size=10):
        schedulers = {
            'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=step_size,
                                                                            factor=lr_decay, verbose=True),
            'step_lr': torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=lr_decay)
        }

        # Elegir el scheduler basado en el argumento 'scheduler_type'
        if scheduler_type not in schedulers.keys():
            raise ValueError(
                f"Scheduler '{scheduler_type}' no soportado. Opciones válidas: {', '.join(schedulers.keys())}")

        self.scheduler = schedulers[scheduler_type]
        return self

    def train(self, train_dataloader, val_dataloader, save_interval, max_steps, checkpoints_dir=None):
        self.global_step = load_checkpoint(self.model, self.optimizer, checkpoints_dir)
        train_pbar = tqdm(train_dataloader, initial=self.global_step, total=float('inf'), dynamic_ncols=True, unit='step')
        while self.global_step < max_steps:
            for batch in train_pbar:
                if self.global_step % save_interval == 0:
                    with torch.no_grad():
                        self.model.eval()
                        self.validate(val_dataloader, self.global_step)
                    save_checkpoint(self.model, self.optimizer, checkpoints_dir, self.global_step, 10)
            self.model.train()
            batch.to(self.device)
            losses, total_loss = self.training_step(batch)

    def training_step(self, batch):
        img_hr = batch[]

    def validate(self, val_loader):
        #todo

    def test(self):

#todo
