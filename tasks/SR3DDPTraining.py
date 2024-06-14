import PIL.Image
import torch
import wandb
from torch.distributed import destroy_process_group

from models.SR3Builder import SR3Builder
from tasks.trainers.SR3DDPTrainer import SR3DDPTrainer
import utils.DDP_utils as ddp_utils
import torch.multiprocessing as mp

from utils.model_utils import load_model
from utils.tensor_utils import tensor2img, move_to_cuda


def set_up_model(hyperparams):
    model_builder = SR3Builder()
    model_builder = model_builder.set_papersm()
    return model_builder.build(), model_builder.set_standart().get_hyperparameters()


def set_up_optimizer(hyperparams, model):
    return torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])


def set_up_scheduler(hyperparams, optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=hyperparams["patience"],
                                                      factor=hyperparams["factor"], mode=hyperparams["mode"])


def execute(rank, world_size, hyperparams, dataset_dir=None, sat_dataset_dir=None, start_epoch=0):
    hyperparams["device"] = rank
    ddp_utils.ddp_setup(rank=rank, world_size=world_size)
    save_dir = hyperparams["save_dir"]

    print(f"seted up world_size:{world_size}")
    trainer = SR3DDPTrainer(hyperparams)

    train_dataloader, val_dataloader, test_dataloader = ddp_utils.set_up_data(hyperparams, dataset_dir, sat_dataset_dir,
                                                                              world_size)

    model, model_data = set_up_model(hyperparams)

    if start_epoch > 0:
        model = load_model(model, f"{hyperparams["model_name"]}{start_epoch}.pt", save_dir)

    optimizer = set_up_optimizer(hyperparams, model)
    scheduler = set_up_scheduler(hyperparams, optimizer)
    trainer.set_model(model)
    trainer.set_optimizer(optimizer)
    trainer.set_scheduler(scheduler)
    trainer.set_dataloaders(train_dataloader, val_dataloader, test_dataloader)

    wandb.init(project="SR3 experiments", config=hyperparams.update(model_data),
               name=f"{hyperparams["model_name"]} GPU{rank} training", group=f"{hyperparams["model_name"]} ddp group")
    torch.backends.cudnn.benchmark = True

    for epoch in range(start_epoch, hyperparams['epochs']):
        with torch.no_grad():
            val_loss = trainer.validate(epoch)

        torch.cuda.empty_cache()
        train_loss = trainer.train_epoch(epoch)
        torch.cuda.empty_cache()

        if epoch % 50 == 0:
            trainer.save_model(epoch)
            #Not a full test just to see how it's going
            test_dataloader.sampler.set_epoch(epoch)
            batch = next(iter(test_dataloader))
            sr = trainer.sample_test(move_to_cuda(batch, rank), get_metrics=False)
            images = [PIL.Image.fromarray(tensor2img(tensor.to('cpu'))) for tensor in sr]
            log_inf = {"train_loss": train_loss, "val_loss": val_loss,
                       "examples": [wandb.Image(image) for image in images]}
            metrics = trainer.test(epoch)
            log_inf['psnr'] = metrics['psnr']
            log_inf['ssim'] = metrics['ssim']
            wandb.log(log_inf)
        else:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        torch.cuda.empty_cache()
    trainer.save_model(epoch)
    destroy_process_group()
    wandb.finish()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    hyperparams = {
        "model_name": "SR3DDP fki2",
        "lr": 1e-6,
        "patience": 5,
        "factor": 0.1,
        "mode": "min",
        "epochs": 1000,
        "save_dir": "C:\\Users\\adrianperera\\Desktop\\SR-model-benchmarking\\saved models\\SR3\\DDPfkimodel2",
        "batch_size": 16,
        "metrics_used": ["psnr", "ssim"]
    }
    wandb.login()
    mp.spawn(execute, args=(world_size, hyperparams, 'C:\\Users\\adrianperera\\Desktop\\dataset_tfg',
                            "C:\\Users\\adrianperera\\Desktop\\dataset_tfg\\satelite_dataset\\64_256", 350),
             nprocs=world_size)
