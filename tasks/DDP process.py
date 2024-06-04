import torch
from torch.utils.data import random_split
from torchvision.transforms.v2 import Compose, RandomApply, GaussianBlur, RandomEqualize

from models.SRDIFFBuilder import SRDiffBuilder
from trainers.SRDiffTrainerDDP import SRDiffTrainerDDP
import utils.DDP_utils as ddp_utils
from Dataset.AerialDataset import AerialDataset
from tasks import trainers
import torch.multiprocessing as mp


def set_up_data(hyperparams, dataset_dir, sat_dataset_dir):
    lr_size = 64
    hr_size = 256
    dataset_dir = dataset_dir

    transforms = Compose(
        [RandomApply(transforms=[GaussianBlur(7)], p=0.5),
         RandomEqualize()]
    )

    dataset = AerialDataset(dataset_dir, lr_size, hr_size, data_augmentation=None, aux_sat_prob=0.4,
                            sat_dataset_path=sat_dataset_dir)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.6, 0.2, 0.2],
                                                            generator=torch.Generator().manual_seed(420))

    train_dataloader = ddp_utils.prepare_data(train_dataset, hyperparams['batch_size'])
    val_dataloader = ddp_utils.prepare_data(val_dataset, hyperparams['batch_size'])
    test_dataloader = ddp_utils.prepare_data(test_dataset, hyperparams['batch_size'])

    return train_dataloader, val_dataloader, test_dataloader


def set_up_trainer(hyperparams, rank):
    return SRDiffTrainerDDP(metrics_used=("ssim", "psnr"), model_name=hyperparams["model_name"],
                            device=rank, use_rrdb=hyperparams["use_rrdb"],
                            fix_rrdb=hyperparams["fix_rrdb"], aux_ssim_loss=hyperparams["aux_ssim_loss"],
                            aux_l1_loss=hyperparams["aux_l1_loss"],
                            aux_perceptual_loss=hyperparams["aux_perceptual_loss"])


def set_up_model():
    model_builder = SRDiffBuilder()
    return model_builder.set_standart().build()


def set_up_optimizer(hyperparams, model):
    return torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])


def set_up_scheduler(hyperparams, optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=hyperparams["decay_steps"], gamma=hyperparams["factor"])


def execute(rank, hyperparams, dataset_dir=None, sat_dataset_dir=None):
    ddp_utils.ddp_setup(rank=rank, world_size=world_size)
    trainer = set_up_trainer(hyperparams, rank)
    train_dataloader, val_dataloader, test_dataloader = set_up_data(hyperparams, dataset_dir, sat_dataset_dir)
    model = set_up_model()
    optimizer = set_up_optimizer(hyperparams, model)
    scheduler = set_up_scheduler(hyperparams, optimizer)
    trainer.set_model(model)
    trainer.set_optimizer(optimizer)
    trainer.set_scheduler(scheduler)

    logs = {"train_loss": [], "val_loss": []}
    for epoch in range(hyperparams['num_epochs']):
        with torch.no_grad():
            val_loss = trainer.validate(val_dataloader, epoch)
            logs["val_loss"].append(val_loss)
        torch.cuda.empty_cache()
        train_loss = trainer.train(train_dataloader, epoch)
        logs["train_loss"].append(train_loss)
        torch.cuda.empty_cache()
        if epoch % 10 == 0:
            trainer.save_model("saved models\\SRDiff\\large")
        torch.cuda.empty_cache()
    test_metrics = trainer.test(test_dataloader, hyperparams['num_epochs'])
    logs.update(test_metrics)


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    hyperparams = {
        "lr": 0.00002,
        "epochs": 20,
        "eta_min": 1e-7,
        "decay_steps": 1000000,
        "mode": "min",
        "factor": 0.5,
        "model": "SRDiff",
        "batch_size": 20,
        "ddp": False,
        "grad_acum": 1,
        "use_rrdb": True,
        "fix_rrdb": True,
        "aux_l1_loss": False,
        "aux_perceptual_loss": False,
        "aux_ssim_loss": False
    }
    mp.spawn(execute, args=(world_size, hyperparams, 'C:\\Users\\adrianperera\\Desktop\\dataset_tfg',
                            "C:\\Users\\adrianperera\\Desktop\\dataset_tfg\\satelite_dataset\\64_256"))
