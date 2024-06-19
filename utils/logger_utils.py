import csv
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


def dict_to_csv(data, filename):
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['val_loss', 'train_loss', 'psnr', 'ssim', 'epoch']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Si el archivo no existe, escribe el encabezado
        if not file_exists:
            writer.writeheader()

        # Escribe la fila de datos
        writer.writerow(data)
