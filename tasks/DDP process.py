import torch
import wandb
from torch.distributed import destroy_process_group
from torch.utils.data import random_split
from torchvision.transforms.v2 import Compose, RandomApply, GaussianBlur, RandomEqualize

from models.SRDIFFBuilder import SRDiffBuilder
from SRDiffTrainerDDP import SRDiffTrainerDDP
import utils.DDP_utils as ddp_utils
from Dataset.AerialDataset import AerialDataset
import torch.multiprocessing as mp


def set_up_data(hyperparams, dataset_dir, sat_dataset_dir, world_size):
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

    train_dataloader = ddp_utils.prepare_data(train_dataset, hyperparams['batch_size'], world_size)
    val_dataloader = ddp_utils.prepare_data(val_dataset, hyperparams['batch_size'], world_size)
    test_dataloader = ddp_utils.prepare_data(test_dataset, hyperparams['batch_size'], world_size)

    return train_dataloader, val_dataloader, test_dataloader


def set_up_trainer(hyperparams, rank):
    return SRDiffTrainerDDP(metrics_used=("ssim", "psnr"), model_name=hyperparams["model_name"],
                            gpu_id=rank, use_rrdb=hyperparams["use_rrdb"],
                            fix_rrdb=hyperparams["fix_rrdb"], aux_ssim_loss=hyperparams["aux_ssim_loss"],
                            aux_l1_loss=hyperparams["aux_l1_loss"],
                            aux_perceptual_loss=hyperparams["aux_perceptual_loss"])


def set_up_model(hyperparams):
    model_builder = SRDiffBuilder()
    model_builder = model_builder.set_standart()
    model_builder = model_builder.set_losstype(hyperparams["losstype"], hyperparams["aux_l1_loss"], hyperparams["aux_perceptual_loss"])
    return model_builder.build(), model_builder.set_standart().get_hyperparameters()


def set_up_optimizer(hyperparams, model):
    return torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])


def set_up_scheduler(hyperparams, optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=hyperparams["patience"],
                                                      factor=hyperparams["factor"], mode=hyperparams["mode"])


def execute(rank, world_size, hyperparams, dataset_dir=None, sat_dataset_dir=None):
    ddp_utils.ddp_setup(rank=rank, world_size=world_size)
    save_dir = "C:\\Users\\adrianperera\\Desktop\\SR-model-benchmarking\\saved models\\SRDiff\\Distributed\\Version 3"

    print(f"seted up world_size:{world_size}")
    trainer = set_up_trainer(hyperparams, rank)

    train_dataloader, val_dataloader, test_dataloader = set_up_data(hyperparams, dataset_dir, sat_dataset_dir, world_size )

    model, model_data = set_up_model(hyperparams)
    optimizer = set_up_optimizer(hyperparams, model)
    scheduler = set_up_scheduler(hyperparams, optimizer)
    trainer.set_model(model)
    trainer.set_optimizer(optimizer)
    trainer.set_scheduler(scheduler)
    print(f"seted up training objects")

    logs = {"train_loss": [], "val_loss": []}

    wandb.init(project="SRDiff experiments", config=hyperparams.update(model_data), name=f"{hyperparams["model_name"]} GPU{rank} training",
               group=f"{hyperparams["model_name"]} ddp group")
    torch.backends.cudnn.benchmark = True

    for epoch in range(hyperparams['epochs']):
        with torch.no_grad():
            val_loss = trainer.validate(val_dataloader, epoch)
            logs["val_loss"].append(val_loss)

        torch.cuda.empty_cache()
        train_loss = trainer.train(train_dataloader, epoch)
        logs["train_loss"].append(train_loss)
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        torch.cuda.empty_cache()

        if epoch % 50 == 0:
            trainer.save_model(save_dir, epoch)
        torch.cuda.empty_cache()
    trainer.save_model(
        save_dir, epoch)
    destroy_process_group()
    wandb.finish()

if __name__ == '__main__':

    world_size = torch.cuda.device_count()
    hyperparams = {
        "lr": 0.00002,
        "model_name": f"SRDiff Percp loss",
        "epochs": 200,
        "eta_min": 1e-7,
        "num_workers":torch.cuda.device_count(),
        "mode": "min",
        "patience": 5,
        "factor": 0.1,
        "model": "SRDiff",
        "batch_size": 20,
        "ddp": True,
        "grad_acum": 1,
        "use_rrdb": True,
        "fix_rrdb": False,
        "aux_l1_loss": False,
        "aux_perceptual_loss": True,
        "aux_ssim_loss": False,
        "losstype": "l1"
    }
    wandb.login()
    mp.spawn(execute, args=(world_size, hyperparams, 'C:\\Users\\adrianperera\\Desktop\\dataset_tfg',
                            "C:\\Users\\adrianperera\\Desktop\\dataset_tfg\\satelite_dataset\\64_256"), nprocs=world_size)
