import os
import sys
import numpy as np
from scipy.interpolate import interpn
from PIL import Image
from tqdm import tqdm

def bicubic_interpolation(image, objective_dim):  # De momento lo implementare para 1 sola foto a la vez
    # Calculo nuevas dimensiones
    height, width = image.shape[0], image.shape[1]
    new_width, new_height = objective_dim[0], objective_dim[1]
    new_image = np.zeros((new_height, new_width, image.shape[2]))

    # Generar cuadrículas para las coordenadas X e Y de la imagen original y la interpolada
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    new_x = np.linspace(0, width - 1, new_width)
    new_y = np.linspace(0, height - 1, new_height)
    new_image = interpn((y, x), image, (new_y[:, None], new_x), method='cubic', bounds_error=False, fill_value=0)
    return new_image

def __main__(args):
    dataset_dir = str(sys.argv[1])
    objective_dim = (int(sys.argv[3]), int(sys.argv[3]))
    original_dim = sys.argv[2]
    results_dir = dataset_dir + f"/{str(original_dim)}_{str(objective_dim[0])}"
    os.mkdir(results_dir)
    for image_file in tqdm(os.listdir(dataset_dir)):
        image = Image.open(os.path.join(dataset_dir, image_file))
        image = np.array(image)
        interpolated = bicubic_interpolation(image, objective_dim)
        interpolated_image = Image.fromarray(interpolated.astype(np.uint8))
        interpolated_image.save(results_dir + f"/{image_file}")
