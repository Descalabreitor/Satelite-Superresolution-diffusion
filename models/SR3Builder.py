from models.SR3.diffusion import GaussianDiffusion
from models.SR3.model import UNet


class SR3Builder:

    def __init__(self):
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

    def build(self):
        model = UNet(3, self.steps, self.channels_expansions, self.emb_expansions,
                     self.resbloks_downstage, self.drp_rate, self.down_att, self.mid_atts, self.up_att)
        diffusion = GaussianDiffusion(model, self.steps, self.sample_steps, self.losstype)
        return diffusion
