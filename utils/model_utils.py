import os

import torch


def save_model(model, model_file_name, models_dir):
    save_dir = os.path.join(models_dir, model_file_name)
    torch.save(model.state_dict(), save_dir)

def load_model(model, model_file_name, models_dir):
    save_dir = os.path.join(models_dir, model_file_name)
    try:
        model.load_state_dict(torch.load(save_dir))
        print(f"The model weights have been loaded from '{save_dir}'")
        return model
    except FileNotFoundError:
        print(f"Error! The file '{save_dir}' does not exist.")
    except Exception as e:
        print(f"Error! Failed to load the model: {e}")