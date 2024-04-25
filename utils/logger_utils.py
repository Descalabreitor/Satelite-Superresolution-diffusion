import os

import torch
import wandb


def log_metrics(metrics):
    metrics = metrics_to_scalars(metrics)
    wandb.log(metrics)


def metrics_to_scalars(metrics):
    new_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            v = v.item()

        if type(v) is dict:
            v = metrics_to_scalars(v)

        new_metrics[k] = v

    return new_metrics


def configure_wandb(project, hyperparams):
    wandb.init(project=project, config=hyperparams)
    wandb.log({"pinga": 10})
