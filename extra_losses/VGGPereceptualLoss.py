import torch
import torch.nn as nn
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=None):
        super(VGGPerceptualLoss, self).__init__()
        if layers is None:
            layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        self.vgg = models.vgg19(pretrained=True).features
        self.layers = layers
        self.layer_name_mapping = {
            '0': 'relu1_1',
            '5': 'relu2_1',
            '10': 'relu3_1',
            '19': 'relu4_1',
            '28': 'relu5_1'
        }
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = {}
        for name, module in self.vgg._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                layer_name = self.layer_name_mapping[name]
                if layer_name in self.layers:
                    outputs[layer_name] = x
        return outputs


class PerceptualLoss(nn.Module):
    def __init__(self, layers=None):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGGPerceptualLoss(layers)
        self.criterion = nn.L1Loss()

    def forward(self, output, target):
        output_features = self.vgg(output)
        target_features = self.vgg(target)
        loss = 0
        for layer in output_features:
            loss += self.criterion(output_features[layer], target_features[layer])
        return loss
