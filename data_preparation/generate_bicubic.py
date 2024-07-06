import os
import sys

from PIL import Image
from tqdm import tqdm


def bicubic_interpolation(image, objective_dim):  # De momento lo implementare para 1 sola foto a la vez
    new_image = image.resize(objective_dim, Image.BICUBIC)
    return new_image

def process_images(dataset_dir, results_dir, objective_dim, method):
    for image_file in tqdm(os.listdir(dataset_dir)):
        image = Image.open(os.path.join(dataset_dir, image_file))
        interpolated_image = bicubic_interpolation(image, objective_dim)
        interpolated_image.save(results_dir + f"/{image_file}")

def __main__(args):
    dataset_dir = str(sys.argv[1])
    objective_dim = (int(sys.argv[3]), int(sys.argv[3]))
    original_dim = sys.argv[2]
    results_dir = dataset_dir + f"/{str(original_dim)}_{str(objective_dim[0])}"
    os.mkdir(results_dir)
    process_images(dataset_dir, results_dir, objective_dim)
