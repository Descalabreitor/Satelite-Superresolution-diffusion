import torch

from models.SRDiff.diffusion import GaussianDiffusion
from models.SRDiff.diffsr_modules import Unet, RRDBNet


class SRDiffBuilder:
    def set_size(self, size):
        self.size = size
        return self

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
