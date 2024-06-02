import torch

from models.SRDiff.diffusion import GaussianDiffusion
from models.SRDiff.diffsr_modules import Unet, RRDBNet


class SRDiffBuilder:

    def __init__(self):
        self.losstype = None
        self.aux_l1 = None
        self.aux_perceptual = None
        self.scale = None
        self.dim_mults = None
        self.timesteps = None
        self.rrdb_blocks = None
        self.rrdb_features = None
        self.hidden = None

    def set_hidden(self, hidden):
        self.hidden = hidden
        return self

    def set_rrdb_features(self, rrdb_features):
        self.rrdb_features = rrdb_features
        return self

    def set_rrdb_blocks(self, rrdb_blocks):
        self.rrdb_blocks = rrdb_blocks
        return self

    def set_timesteps(self, timesteps):
        self.timesteps = timesteps
        return self

    def set_dim_mults(self, dim_mults):
        self.dim_mults = dim_mults
        return self

    def set_scale(self, scale):
        self.scale = scale
        return self

    def set_losstype(self, losstype, aux_l1, aux_perceptual):
        self.losstype = losstype
        self.aux_l1 = aux_l1
        self.aux_perceptual = aux_perceptual
        return self

    def build(self):
        denoise_fn = Unet(self.hidden, out_dim=3, cond_dim=self.rrdb_features,
                          dim_mults=self.dim_mults, rrdb_num_block=self.rrdb_blocks,
                          sr_scale=self.scale)
        rrdb = RRDBNet(3, 3, self.rrdb_features, self.rrdb_blocks, self.rrdb_features // 2)

        model = GaussianDiffusion(denoise_fn=denoise_fn,
                                  rrdb_net=rrdb,
                                  timesteps=self.timesteps,
                                  loss_type=self.losstype,
                                  aux_l1_loss=self.aux_l1,
                                  aux_perceptual_loss=self.aux_perceptual)

        return model

    def set_standart(self):
        self.hidden = 64
        self.dim_mults = [1, 2, 2, 4]
        self.scale = 4
        self.losstype = 'l1'
        self.aux_l1 = False
        self.aux_perceptual = False
        self.rrdb_blocks = 8
        self.rrdb_features = 32
        self.timesteps = 100
        return self

    def set_small(self):
        self.hidden = 32
        self.dim_mults = [1, 2, 2, 4]
        self.scale = 4
        self.losstype = 'l1'
        self.aux_l1 = False
        self.aux_perceptual = False
        self.rrdb_blocks = 4
        self.rrdb_features = 16
        self.timesteps = 100
        return self

    def set_large(self):
        self.hidden = 96
        self.dim_mults = [1, 2, 2, 4]
        self.scale = 4
        self.losstype = 'l1'
        self.aux_l1 = False
        self.aux_perceptual = False
        self.rrdb_blocks = 10
        self.rrdb_features = 48
        self.timesteps = 100
        return self

    def get_hyperparameters(self):
        hyperparameters = {
            "losstype": self.losstype,
            "aux_l1": self.aux_l1,
            "aux_perceptual": self.aux_perceptual,
            "scale": self.scale,
            "dim_mults": self.dim_mults,
            "timesteps": self.timesteps,
            "rrdb_blocks": self.rrdb_blocks,
            "rrdb_features": self.rrdb_features,
            "hidden": self.hidden
        }
        return hyperparameters