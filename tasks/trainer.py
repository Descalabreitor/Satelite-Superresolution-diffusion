import torch
from torch.utils.tensorboard import SummaryWriter



class Trainer:
    def __init__(self, logs_dir):
        self.logger = SummaryWriter(logs_dir)

    def build_optimizer(self, optimizer, params, lr=1e-3, momentum=0.9):
        optimizers = {
            'adam': torch.optim.Adam(params, lr=lr),
            'sgd': torch.optim.SGD(params, lr=lr, momentum=momentum)
        }

        if optimizer not in optimizers.keys():
            raise ValueError(
                f"Optimizador '{optimizer}' not suported. Valid Options: {', '.join(optimizers.keys())}")

        return optimizers[optimizer]

    def build_model(self, model):
        #todo

    def build_scheduler(self, scheduler_type, optimizer, lr_decay=0.1, step_size=10):
        schedulers = {
            'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=step_size,
                                                                            factor=lr_decay, verbose=True),
            'step_lr': torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr_decay)
        }

        # Elegir el scheduler basado en el argumento 'scheduler_type'
        if scheduler_type not in schedulers.keys():
            raise ValueError(
                f"Scheduler '{scheduler_type}' no soportado. Opciones válidas: {', '.join(schedulers.keys())}")

        return schedulers[scheduler_type]

    def train(self, train_loader, val_loader, epochs):
        #TODO

    def validate(self, model, val_loader):
        #todo
    def test(self):
        #todo