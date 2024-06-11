from models.SR3.diffusion import GaussianDiffusion
from models.SR3.model import UNet


class SR3Builder:

    def __init__(self):
        self.C = None
        self.losstype = None
        self.mid_atts = None
        self.drp_rate = None
        self.resbloks_downstage = None
        self.emb_expansions = None
        self.channels_expansions = None
        self.sample_steps = None
        self.steps = None
        self.up_att = None
        self.down_att = None

    def set_steps(self, steps):
        self.steps = steps
        return self

    def set_sample_steps(self, sample_steps):
        self.sample_steps = sample_steps
        return self

    def set_channels_expansions(self, channels_expansions):
        self.channels_expansions = channels_expansions
        return self

    def set_emb_expansions(self, emb_expansions):
        self.emb_expansions = emb_expansions
        return self

    def set_resbloks_downstage(self, resbloks_downstage):
        self.resbloks_downstage = resbloks_downstage
        return self

    def set_drp_rate(self, rate):
        self.drp_rate = rate
        return self

    def set_down_att(self, attn):
        self.down_att = attn
        return self

    def set_up_att(self, attn):
        self.up_att = attn
        return self

    def set_mid_atts(self, attns):
        self.mid_atts = attns
        return self

    def set_losstype(self, losstype):
        self.losstype = losstype
        return self

    def set_C(self, C):
        self.C = C
        return self

    def build(self):
        if not self.C:
            self.C = 3

        model = UNet(self.C, self.steps, self.channels_expansions, self.emb_expansions,
                     self.resbloks_downstage, self.drp_rate, self.down_att, self.mid_atts, self.up_att)
        diffusion = GaussianDiffusion(model, self.steps, self.sample_steps, self.losstype)
        return diffusion

    def set_standart(self):
        self.steps = 2000
        self.sample_steps = 100
        self.channels_expansions = (1, 2, 4, 4, 8, 8)
        self.emb_expansions = 4
        self.resbloks_downstage = 3
        self.drp_rate = 0
        self.down_att = True
        self.up_att = True
        self.mid_atts = (True, False)
        self.losstype = "l2"
        return self

    def set_sr3plus(self):
        self.steps = 20000
        self.sample_steps = 1000
        self.channels_expansions = (1, 2, 4, 4, 8, 8)
        self.emb_expansions = 4
        self.resbloks_downstage = 5
        self.drp_rate = 0
        self.down_att = False
        self.up_att = False
        self.mid_atts = (False, False)
        self.losstype = "l2"
        return self

    def set_papermodel(self):
        self.channels_expansions = (1, 2, 4, 4, 8, 8)
        self.emb_expansions = 4
        self.resbloks_downstage = 3
        self.C = 128
        self.steps = 1000
        self.sample_steps = 100
        self.losstype = "l2"
        self.drp_rate = 0
        self.down_att = True
        self.up_att = True
        self.mid_atts = (True, False)
        return self


    def get_hyperparameters(self):
        hyperparameters = {
            "losstype": self.losstype,
            "mid_atts": self.mid_atts,
            "drp_rate": self.drp_rate,
            "resbloks_downstage": self.resbloks_downstage,
            "emb_expansions": self.emb_expansions,
            "channels_expansions": self.channels_expansions,
            "sample_steps": self.sample_steps,
            "steps": self.steps,
            "up_att": self.up_att,
            "down_att": self.down_att
        }
        return hyperparameters
