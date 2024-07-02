import json

import PIL.Image
import torch
import wandb

import utils.logger_utils
from tasks.trainers.SRDiffTrainer import SRDiffTrainer
from models.SRDIFFBuilder import SRDiffBuilder
from Dataset.StandartDaloader import setUpStandartDataloaders
from utils.model_utils import load_model
from utils.tensor_utils import tensor2img, move_to_cuda


def setUpTrainingObjects(config):
    model_builder = SRDiffBuilder()
    model_builder = model_builder.set_standart()
    model_builder = model_builder.set_timesteps(600)
    model_builder = model_builder.use_pretrained_rrdb(config['pretrained_rrdb'])
    model, _ = model_builder.build()

    optimizer = buildOptimizer(config, model)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=config['factor'], patience=config['patience'])

    return model, optimizer, scheduler, model_builder.get_hyperparameters()


def buildOptimizer(config, model):
    params = list(model.named_parameters())
    if not config['fix_rrdb']:
        params = [p for p in params if 'rrdb' not in p[0]]
    params = [p[1] for p in params]
    return torch.optim.Adam(params, lr=config['lr'])


def execute_check(config, test_dataloader, epoch, trainer, log_data_wandb, log_data_local):
    trainer.save_model(epoch)
    visualization_batch = next(iter(test_dataloader))
    sr_images = trainer.sample_test(move_to_cuda(visualization_batch, device=config["device"]), get_metrics=False)
    sr_images = [PIL.Image.fromarray(tensor2img(tensor.to('cpu'))) for tensor in sr_images]
    for id, image in enumerate(sr_images):
        image.save(
            f"C:\\Users\\adria\\Desktop\\TFG-code\\SR-model-benchmarking\\test pictures\\{config['model_name']}\\Epoch_{epoch}_{id}.png")
    metrics = trainer.test()
    log_data_wandb["examples"] = [wandb.Image(image) for image in sr_images]
    for metric in metrics.keys():
        log_data_wandb[metric] = float(metrics[metric])
        log_data_local[metric] = float(metrics[metric])

    return log_data_wandb, log_data_local


def execute(config):
    model, optimizer, scheduler, model_data = setUpTrainingObjects(config)

    model.to(config["device"])
    train_dataloader, val_dataloader, test_dataloader = setUpStandartDataloaders(config, config['dataset_path'])
    if config["start_epoch"] > 0:
        model = load_model(model, f"{config['model_name']} Epoch{config["start_epoch"]}.pt", config["save_dir"])

    trainer = SRDiffTrainer(config)
    trainer.set_model(model)
    trainer.set_optimizer(optimizer)
    trainer.set_scheduler(scheduler)
    trainer.set_dataloaders(train_dataloader, val_dataloader, test_dataloader)

    wandb.login(relogin=True, key="e13381c1bc10ba98afb7a152e624e1fc4d097e54")
    wandb.init(project="SRDiff experiments", config=config.update(model_data),
               name=config['model_name'] + f"_{config['start_epoch']}")
    utils.logger_utils.log_config(config, "SRDIFF")

    log_data_local = {}
    for epoch in range(config["start_epoch"] + 1, config['num_epochs']+1):
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

        if epoch % 25 == 0 and epoch != 0:
            log_data_wandb, log_data_local = execute_check(config, test_dataloader, epoch, trainer,
                                                           log_data_wandb, log_data_local)
        log_data_local["epoch"] = epoch
        utils.logger_utils.dict_to_csv(log_data_local,
                                       f"{config["project_root"]}\\logs\\SRDIFF\\{config["model_name"]}")
        wandb.log(log_data_wandb)
        torch.cuda.empty_cache()

    log_data_wandb = {}
    log_data_wandb = execute_check(config, test_dataloader, config["num_epochs"], trainer, log_data_wandb)
    wandb.log(log_data_wandb)

    wandb.finish()


if __name__ == "__main__":
    config = {
        'num_epochs': 50,
        'lr': 1e-6,
        'patience': 10,
        'factor': 0.1,
        'fix_rrdb': True,
        'use_rrdb': True,
        "aux_l1_loss": False,
        "aux_perceptual_loss": False,
        "aux_ssim_loss": False,
        "losstype": "l1",
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 10,
        'grad_acum': 1,
        "num_workers": 1,
        "model_name": "SRDiff_600_steps",
        "lr_size": 64,
        "hr_size": 256,
        "save_dir": "C:\\Users\\adria\\Desktop\\TFG-code\\SR-model-benchmarking\\saved models\\SRDiff\\600_steps",
        "project_root":"C:\\Users\\adria\\Desktop\\TFG-code\\SR-model-benchmarking",
        "dataset_path":"C:\\Users\\adria\\Desktop\\dataset_tfg",
        "metrics_used": ("psnr", "ssim"),
        "start_epoch": 0,
        "grad_loss_weight": 0.1,
        "pretrained_rrdb": "C:\\Users\\adria\\Desktop\\TFG-code\\SR-model-benchmarking\\saved models\\RRDB\\RRDB pretrained Epoch100.pt"
    }
    execute(config)
