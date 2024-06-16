import PIL.Image
import torch
import wandb

from tasks.trainers.SRDiffTrainer import SRDiffTrainer
from models.SRDIFFBuilder import SRDiffBuilder
from Dataset.StandartDaloader import setUpDataloaders
from utils.tensor_utils import tensor2img, move_to_cuda


def setUpTrainingObjects(config):
    model_builder = SRDiffBuilder()
    model_builder.set_standart()
    model = model_builder.build()

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


def execute_check(config, test_dataloader, epoch, trainer, log_data):
    trainer.save_model(epoch)
    visualization_batch = next(iter(test_dataloader))
    sr_images = trainer.sample_test(move_to_cuda(visualization_batch, device=config["device"]), get_metrics=False)
    sr_images = [PIL.Image.fromarray(tensor2img(tensor.to('cpu'))) for tensor in sr_images]
    metrics = trainer.test()
    log_data["examples"] = [wandb.Image(image) for image in sr_images]
    for metric in metrics.keys():
        log_data[metric] = metrics[metric]
    return log_data


def execute(config):
    model, optimizer, scheduler, model_data = setUpTrainingObjects(config)
    model.to(config["device"])
    train_dataloader, val_dataloader, test_dataloader = setUpDataloaders(config, "E:\\TFG\\dataset_tfg")

    trainer = SRDiffTrainer(config)
    trainer.set_model(model)
    trainer.set_optimizer(optimizer)
    trainer.set_scheduler(scheduler)
    trainer.set_dataloaders(train_dataloader, val_dataloader, test_dataloader)

    wandb.login()
    wandb.init(project="SRDiff experiments", config=config.update(model_data), name=config['model_name'])

    for epoch in range(config['num_epochs']):
        log_data = {}
        with torch.no_grad():
            val_loss = trainer.validate()
            log_data["val_loss"] = val_loss
            torch.cuda.empty_cache()

        train_loss = trainer.train_epoch(epoch)
        log_data["train_loss"] = train_loss
        torch.cuda.empty_cache()

        if epoch % 100 == 0 and epoch != 0:
            log_data = execute_check(config, test_dataloader, epoch, trainer, log_data)

        wandb.log(log_data)
        torch.cuda.empty_cache()

    log_data = {}
    log_data = execute_check(config, test_dataloader, config["n_epochs"], trainer, log_data)
    wandb.log(log_data)

    wandb.finish()

if __name__ == "__main__":
    config = {
        'num_epochs': 1000,
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
        "model_name": "SRDiff ver4",
        "lr_size": 64,
        "hr_size": 256,
        "save_dir": "C:\\Users\\adria\\Desktop\\TFG-code\\SR-model-benchmarking\\saved models\\SRDiff\\version 4",
        "metrics_used": ("psnr", "ssim")
    }
    execute(config)
