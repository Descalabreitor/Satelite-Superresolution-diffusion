from .SR3_unofficial.sr3_modules import diffusion, unet


class SR3unofBuilder:
    def __init__(self):
        self.in_channel = None
        self.out_channel = None
        self.norm_groups = None
        self.inner_channel = None
        self.channel_mults = None
        self.attn_res = None
        self.res_blocks = None
        self.dropout = None
        self.image_size = 256

        self.channels = None
        self.losstype = "l1"
        self.conditional = True
        self.schedule_opt = None

    def set_in_channel(self, in_channel):
        self.in_channel = in_channel
        return self

    def set_out_channel(self, out_channel):
        self.out_channel = out_channel
        return self

    def set_norm_groups(self, norm_groups):
        self.norm_groups = norm_groups
        return self

    def set_inner_channel(self, inner_channel):
        self.inner_channel = inner_channel
        return self

    def set_channel_mults(self, channel_mults):
        self.channel_mults = channel_mults
        return self

    def set_attn_res(self, attn_res):
        self.attn_res = attn_res
        return self

    def set_res_blocks(self, res_blocks):
        self.res_blocks = res_blocks
        return self

    def set_dropout(self, dropout):
        self.dropout = dropout
        return self

    def set_image_size(self, image_size):
        self.image_size = image_size
        return self

    def set_channels(self, channels):
        self.channels = channels
        return self

    def set_losstype(self, losstype):
        self.losstype = losstype
        return self

    def set_conditional(self, conditional):
        self.conditional = conditional
        return self

    def set_schedule_opt(self, scheduler):
        self.schedule_opt = scheduler
        return self

    def build(self):
        model = unet.UNet(in_channel=self.in_channel,
                          out_channel=self.out_channel,
                          norm_groups=self.norm_groups,
                          inner_channel=self.inner_channel,
                          channel_mults=self.channel_mults,
                          attn_res=self.attn_res,
                          res_blocks=self.res_blocks,
                          dropout=self.dropout,
                          image_size=self.image_size,
                          )

        net_g = diffusion.GaussianDiffusion(model,
                                            image_size=self.image_size,
                                            channels=self.channels,
                                            loss_type=self.losstype,
                                            conditional=self.conditional,
                                            schedule_opt=self.schedule_opt)

        return net_g

    def set_default(self):
        self.in_channel = 6
        self.out_channel = 3
        self.inner_channel = 64
        self.norm_groups = 16
        self.channel_mults = [1, 2, 4, 4, 8, 8]
        self.res_blocks = 1
        self.dropout = 0

        self.schedule_opt = {
            "schedule": "linear",
            "n_timestep": 2000,
            "linear_start": 1e-6,
            "linear_end": 1e-2
        }

        self.losstype = "l1"
        self.conditional = True
        self.channels = 3
        self.attn_res = [8]

        return self

    def get_hyperparameters(self):
        hyperparameters = {
            "in_channel": self.in_channel,
            "out_channel": self.out_channel,
            "norm_groups": self.norm_groups,
            "channel_mults": self.channel_mults,
            "res_blocks": self.res_blocks,
            "dropout": self.dropout,
            "image_size": self.image_size,
            "channels": self.channels,
            "losstype": self.losstype,
            "conditional": self.conditional,
            "schedule_opt": self.schedule_opt,
        }
        return hyperparameters

