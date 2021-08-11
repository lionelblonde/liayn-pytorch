import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U


STANDARDIZED_OB_CLAMPS = [-5., 5.]


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Core.

def init(weight_scale=1., constant_bias=0.):
    """Perform orthogonal initialization"""

    def _init(m):

        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=weight_scale)
            if m.bias is not None:
                nn.init.constant_(m.bias, constant_bias)
        elif (isinstance(m, nn.BatchNorm2d) or
              isinstance(m, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    return _init


def snwrap(use_sn=False):
    """Spectral normalization wrapper"""

    def _snwrap(m):

        assert isinstance(m, nn.Linear)
        if use_sn:
            return U.spectral_norm(m)
        else:
            return m

    return _snwrap


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Models.

class Discriminator(nn.Module):

    def __init__(self, env, hps, rms_obs):
        super(Discriminator, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        if hps.wrap_absorb:
            ob_dim += 1
            ac_dim += 1
        self.hps = hps
        self.leak = 0.1
        apply_sn = snwrap(use_sn=self.hps.spectral_norm)
        if self.hps.d_batch_norm:
            # Define observation whitening
            self.rms_obs = rms_obs
        # Define the input dimension
        in_dim = ob_dim
        if self.hps.state_only:
            in_dim += ob_dim
        else:
            in_dim += ac_dim
        # Assemble the layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', apply_sn(nn.Linear(in_dim, 100))),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', apply_sn(nn.Linear(100, 100))),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
        ]))
        self.d_head = nn.Linear(100, 1)
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=math.sqrt(2) / math.sqrt(1 + self.leak**2)))
        self.d_head.apply(init(weight_scale=0.01))

    def D(self, input_a, input_b):
        return self.forward(input_a, input_b)

    def forward(self, input_a, input_b):
        if self.hps.d_batch_norm:
            # Apply normalization
            if self.hps.wrap_absorb:
                # Normalize state
                input_a_ = input_a.clone()[:, 0:-1]
                input_a_ = self.rms_obs.standardize(input_a_).clamp(*STANDARDIZED_OB_CLAMPS)
                input_a = torch.cat([input_a_, input_a[:, -1].unsqueeze(-1)], dim=-1)
                if self.hps.state_only:
                    # Normalize next state
                    input_b_ = input_b.clone()[:, 0:-1]
                    input_b_ = self.rms_obs.standardize(input_b_).clamp(*STANDARDIZED_OB_CLAMPS)
                    input_b = torch.cat([input_b_, input_b[:, -1].unsqueeze(-1)], dim=-1)
            else:
                # Normalize state
                input_a = self.rms_obs.standardize(input_a).clamp(*STANDARDIZED_OB_CLAMPS)
                if self.hps.state_only:
                    # Normalize next state
                    input_b = self.rms_obs.standardize(input_b).clamp(*STANDARDIZED_OB_CLAMPS)
        else:
            input_a = input_a.clamp(*STANDARDIZED_OB_CLAMPS)
            if self.hps.state_only:
                input_b = input_b.clamp(*STANDARDIZED_OB_CLAMPS)
        # Concatenate
        x = torch.cat([input_a, input_b], dim=-1)
        x = self.fc_stack(x)
        score = self.d_head(x)  # no sigmoid here
        return score


class Actor(nn.Module):

    def __init__(self, env, hps, rms_obs):
        super(Actor, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        self.ac_max = env.action_space.high[0]
        self.hps = hps
        # Define observation whitening
        self.rms_obs = rms_obs
        # Assemble the last layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim, 300)),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(300)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.a_fc_stack = nn.Sequential(OrderedDict([
            ('fc_block', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(300, 200)),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(200)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.a_head = nn.Linear(200, ac_dim)
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=math.sqrt(2)))
        self.a_fc_stack.apply(init(weight_scale=math.sqrt(2)))
        self.a_head.apply(init(weight_scale=0.01))

    def act(self, ob):
        out = self.forward(ob)
        return out[0]  # ac

    def forward(self, ob):
        ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x = self.fc_stack(ob)
        ac = float(self.ac_max) * torch.tanh(self.a_head(self.a_fc_stack(x)))
        out = [ac]
        return out

    @property
    def perturbable_params(self):
        return [n for n, _ in self.named_parameters() if 'ln' not in n]

    @property
    def non_perturbable_params(self):
        return [n for n, _ in self.named_parameters() if 'ln' in n]


class Critic(nn.Module):

    def __init__(self, env, hps, rms_obs):
        super(Critic, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        if hps.use_c51:
            num_heads = hps.c51_num_atoms
        elif hps.use_qr:
            num_heads = hps.num_tau
        else:
            num_heads = 1
        self.hps = hps
        # Define observation whitening
        self.rms_obs = rms_obs
        # Assemble the last layers and output heads
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim + ac_dim, 400)),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(400)),
                ('nl', nn.ReLU()),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(400, 300)),
                ('ln', (nn.LayerNorm if hps.layer_norm else nn.Identity)(300)),
                ('nl', nn.ReLU()),
            ]))),
        ]))
        self.head = nn.Linear(300, num_heads)
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=math.sqrt(2)))
        self.head.apply(init(weight_scale=0.01))

    def QZ(self, ob, ac):
        return self.forward(ob, ac)

    def forward(self, ob, ac):
        ob = self.rms_obs.standardize(ob).clamp(*STANDARDIZED_OB_CLAMPS)
        x = torch.cat([ob, ac], dim=-1)
        x = self.fc_stack(x)
        x = self.head(x)
        if self.hps.use_c51:
            # Return a categorical distribution
            x = F.log_softmax(x, dim=1).exp()
        return x

    @property
    def out_params(self):
        return [p for p in self.head.parameters()]
