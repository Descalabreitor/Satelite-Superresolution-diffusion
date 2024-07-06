import csv
import json
import os

import torch


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


def log_config(config, model_type):
    logged_config = config
    with open(
            f"{config["project_root"]}\\logs\\{model_type}\\config_{config["model_name"]}.json",
            "w") as archivo:
        if config["device"] == torch.device("cuda"):
            logged_config["device"] = "cuda"
        else:
            logged_config["device"] = "cpu"

        json.dump(logged_config, archivo, indent=4)
