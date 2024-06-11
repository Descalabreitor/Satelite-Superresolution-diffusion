import torch
import wandb
from torch.distributed import destroy_process_group

from models.SR3Builder import SR3Builder
from tasks.trainers.SR3DDPTrainer import SR3DDPTrainer
from tasks.trainers.SRDiffTrainerDDP import SRDiffTrainerDDP
import utils.DDP_utils as ddp_utils
import torch.multiprocessing as mp


def set_up_model(hyperparams):
    model_builder = SR3Builder()
    model_builder = model_builder.set_standart()
    return model_builder.build(), model_builder.set_standart().get_hyperparameters()


def set_up_optimizer(hyperparams, model):
    return torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])


def set_up_scheduler(hyperparams, optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=hyperparams["patience"],
                                                      factor=hyperparams["factor"], mode=hyperparams["mode"])


def execute(rank, world_size, hyperparams, dataset_dir=None, sat_dataset_dir=None):
    hyperparams["device"] = rank
    ddp_utils.ddp_setup(rank=rank, world_size=world_size)
    save_dir = hyperparams["save_dir"]

    print(f"seted up world_size:{world_size}")
    trainer = SR3DDPTrainer(hyperparams)

    train_dataloader, val_dataloader, test_dataloader = ddp_utils.set_up_data(hyperparams, dataset_dir, sat_dataset_dir,
                                                                              world_size)

    model, model_data = set_up_model(hyperparams)
    optimizer = set_up_optimizer(hyperparams, model)
    scheduler = set_up_scheduler(hyperparams, optimizer)
    trainer.set_model(model)
    trainer.set_optimizer(optimizer)
    trainer.set_scheduler(scheduler)
    trainer.set_dataloaders(train_dataloader, val_dataloader, test_dataloader)

    wandb.init(project="SR3 experiments", config=hyperparams.update(model_data),
               name=f"{hyperparams["model_name"]} GPU{rank} training",
               group=f"{hyperparams["model_name"]} ddp group")
    torch.backends.cudnn.benchmark = True

    for epoch in range(hyperparams['epochs']):
        with torch.no_grad():
            val_loss = trainer.validate(val_dataloader, epoch)

        torch.cuda.empty_cache()
        train_loss = trainer.train_epoch(train_dataloader, epoch)
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
        torch.cuda.empty_cache()

        if epoch % 50 == 0:
            trainer.save_model(save_dir, epoch)
        torch.cuda.empty_cache()
    trainer.save_model(save_dir, epoch)
    destroy_process_group()
    wandb.finish()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    hyperparams = {
        "model_name": "SR3DDP fki",
        "lr": 1e-3,
        "patience": 5,
        "factor": 0.1,
        "mode": "min",
        "epochs": 1000,
        "save_dir": "saved models\\SR3\\literalmente el paper",
        "batch_size": 100
    }
    wandb.login()
    mp.spawn(execute, args=(world_size, hyperparams, 'C:\\Users\\adrianperera\\Desktop\\dataset_tfg',
                            "C:\\Users\\adrianperera\\Desktop\\dataset_tfg\\satelite_dataset\\64_256"),
             nprocs=world_size)
