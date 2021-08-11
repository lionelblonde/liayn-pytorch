import math
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import logger
from helpers.console_util import log_module_info
from agents.nets import init


STANDARDIZED_OB_CLAMPS = [-5., 5.]


class PredNet(nn.Module):

    def __init__(self, env, hps, rms_obs):
        super(PredNet, self).__init__()
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        if hps.wrap_absorb:
            ob_dim += 1
            ac_dim += 1
        self.hps = hps
        self.leak = 0.1
        if self.hps.forward_batch_norm:
            # Define observation whitening
            self.rms_obs = rms_obs
        # Assemble the layers
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(ob_dim + ac_dim, 100)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(100, 100)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
        ]))
        self.head = nn.Linear(100, ob_dim)
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=math.sqrt(2) / math.sqrt(1 + self.leak**2)))
        self.head.apply(init(weight_scale=0.01))

    def forward(self, obs, acs):
        if self.hps.forward_batch_norm:
            # Apply normalization
            if self.hps.wrap_absorb:
                obs_ = obs.clone()[:, 0:-1]
                obs_ = self.rms_obs.standardize(obs_).clamp(*STANDARDIZED_OB_CLAMPS)
                obs = torch.cat([obs_, obs[:, -1].unsqueeze(-1)], dim=-1)
            else:
                obs = self.rms_obs.standardize(obs).clamp(*STANDARDIZED_OB_CLAMPS)
        else:
            obs = obs.clamp(*STANDARDIZED_OB_CLAMPS)
        # Concatenate
        x = torch.cat([obs, acs], dim=-1)
        x = self.fc_stack(x)
        x = self.head(x)
        return x


class Forward(object):

    def __init__(self, env, device, hps, rms_obs):
        self.env = env
        self.device = device
        self.hps = hps
        self.rms_obs = rms_obs

        # Create nets
        self.pred_net = PredNet(self.env, self.hps, self.rms_obs).to(self.device)

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=self.hps.actor_lr)  # using actor lr

        log_module_info(logger, 'Forward Pred Network', self.pred_net)

    def update(self, state, action, next_state):
        """Update the dynamics predictor network"""

        # Container for all the metrics
        metrics = defaultdict(list)

        logger.info("updating forward predictor")

        # Compute loss
        _loss = F.mse_loss(
            self.pred_net(state, action),
            next_state,
            reduction='none',
        )
        loss = _loss.mean()
        metrics['loss'].append(loss)

        # Update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        metrics = {k: torch.stack(v).mean().cpu().data.numpy() for k, v in metrics.items()}
        return metrics
