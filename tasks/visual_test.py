import torch
from PIL import Image
from torchvision import transforms


def visual_test_SR3(model, test_img_dir, objective_size, device):
    img = Image.open(test_img_dir)
    img = img.resize((objective_size, objective_size), Image.BICUBIC)
    img = transforms.ToTensor()(img)
    img.to(device)
    with torch.no_grad():
        sampled = model.sample(img)
    return sampled
