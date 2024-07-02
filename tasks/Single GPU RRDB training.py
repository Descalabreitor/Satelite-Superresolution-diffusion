import json

import PIL.Image
import torch
import wandb

from Dataset.StandartDaloader import setUpStandartDataloaders
from models.SRDIFFBuilder import SRDiffBuilder
from tasks.trainers.RRDBTrainer import RRDBTrainer
from utils import logger_utils
from utils.logger_utils import log_config
from utils.model_utils import load_model
from utils.tensor_utils import move_to_cuda, tensor2img


def setUpTrainingObjects(config):
    model_builder = SRDiffBuilder()
    model_builder = model_builder.set_standart()
    _, model = model_builder.build()

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=config['factor'], patience=config['patience'])

    return model, optimizer, scheduler, model_builder.get_hyperparameters()


def execute_check(config, test_dataloader, epoch, trainer, log_data_wandb, log_data_local):
    trainer.save_model(epoch)
    metrics = trainer.test()
    for metric in metrics.keys():
        log_data_wandb[metric] = float(metrics[metric])
        log_data_local[metric] = float(metrics[metric])

    return log_data_wandb, log_data_local


def execute(config):
    model, optimizer, scheduler, model_data = setUpTrainingObjects(config)

    model.to(config["device"])

    train_dataloader, val_dataloader, test_dataloader = setUpStandartDataloaders(config, config["dataset_path"])

    if config["start_epoch"] > 0:
        model = load_model(model, f"{config['model_name']} Epoch{config["start_epoch"]}.pt", config["save_dir"])

    trainer = RRDBTrainer(config)
    trainer.set_model(model)
    trainer.set_optimizer(optimizer)
    trainer.set_scheduler(scheduler)
    trainer.set_dataloaders(train_dataloader, val_dataloader, test_dataloader)

    wandb.login(relogin=True, key="e13381c1bc10ba98afb7a152e624e1fc4d097e54")
    wandb.init(project="SRDiff experiments", config=config.update(model_data),
               name=config['model_name'] + f"_{config['start_epoch']}")

    log_config(config, "RRDB")

    log_data_local = {}
    for epoch in range(config["start_epoch"], config['num_epochs'] + 1):
        log_data_wandb = {}
        with torch.no_grad():
            val_loss = trainer.validate()
            log_data_wandb["val_loss"] = val_loss
            log_data_local["val_loss"] = float(val_loss)
            torch.cuda.empty_cache()

        train_loss = trainer.train_epoch(epoch)
        log_data_wandb["train_loss"] = train_loss
        log_data_local["train_loss"] = float(train_loss)
        torch.cuda.empty_cache()

        if epoch % 100 == 0 and epoch != 0:
            log_data_wandb, log_data_local = execute_check(config, test_dataloader, epoch, trainer,
                                                           log_data_wandb, log_data_local)

        log_data_local["epoch"] = epoch
        logger_utils.dict_to_csv(log_data_local,
                                 f"{config["project_root"]}\\logs\\RRDB\\{config["model_name"]}")
        wandb.log(log_data_wandb)
        torch.cuda.empty_cache()

    log_data_wandb = {}
    log_data_wandb = execute_check(config, test_dataloader, config["n_epochs"], trainer, log_data_wandb, log_data_local)
    wandb.log(log_data_wandb)

    wandb.finish()


if __name__ == '__main__':
    config = {
        'num_epochs': 500,
        'lr': 1e-6,
        'patience': 10,
        'factor': 0.1,
        "losstype": "l1",
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 45,
        'grad_acum': 1,
        "num_workers": 1,
        "model_name": "RRDB pretrained",
        "lr_size": 64,
        "hr_size": 256,
        "save_dir": "C:\\Users\\adria\\Desktop\\TFG-code\\SR-model-benchmarking\\saved models\\RRDB",
        "metrics_used": ("psnr", "ssim"),
        "start_epoch": 0,
        "dataset_path": "E:\\TFG\\dataset_tfg",
        "project_root": "C:\\Users\\adria\\Desktop\\TFG-code\\SR-model-benchmarking",
        "grad_loss_weight": 0.1
    }
    execute(config)
