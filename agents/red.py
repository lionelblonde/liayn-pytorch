import math
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from helpers import logger
from helpers.console_util import log_module_info
from agents.nets import init
from helpers.distributed_util import RunMoms


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
        if self.hps.red_batch_norm:
            # Define observation whitening
            self.rms_obs = rms_obs
        # Define the input dimension
        in_dim = ob_dim
        if self.hps.state_only:
            in_dim += ob_dim
        else:
            in_dim += ac_dim
        # Assemble the layers
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(in_dim, 100)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(100, 100)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
            ('fc_block_3', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(100, 100)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
        ]))
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=math.sqrt(2) / math.sqrt(1 + self.leak**2)))

    def forward(self, input_a, input_b):
        if self.hps.red_batch_norm:
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
        return x


class TargNet(PredNet):

    def __init__(self, env, hps, rms_obs):
        super(TargNet, self).__init__(env, hps, rms_obs)
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]
        if hps.wrap_absorb:
            ob_dim += 1
            ac_dim += 1
        self.hps = hps
        self.leak = 0.1
        if self.hps.red_batch_norm:
            # Define observation whitening
            self.rms_obs = rms_obs
        # Define the input dimension
        in_dim = ob_dim
        if self.hps.state_only:
            in_dim += ob_dim
        else:
            in_dim += ac_dim
        # Assemble the layers
        self.fc_stack = nn.Sequential(OrderedDict([
            ('fc_block_1', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(in_dim, 100)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
            ('fc_block_2', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(100, 100)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
            ('fc_block_3', nn.Sequential(OrderedDict([
                ('fc', nn.Linear(100, 100)),
                ('nl', nn.LeakyReLU(negative_slope=self.leak)),
            ]))),
        ]))
        # Perform initialization
        self.fc_stack.apply(init(weight_scale=math.sqrt(2) / math.sqrt(1 + self.leak**2)))

        # Prevent the weights from ever being updated
        for param in self.fc_stack.parameters():
            param.requires_grad = False


class RandomExpertDistillation(object):

    def __init__(self, env, device, hps, expert_dataset, rms_obs):
        self.env = env
        self.device = device
        self.hps = hps
        self.expert_dataset = expert_dataset
        self.rms_obs = rms_obs

        # Create nets
        self.pred_net = PredNet(self.env, self.hps, self.rms_obs).to(self.device)
        self.targ_net = TargNet(self.env, self.hps, self.rms_obs).to(self.device)  # fixed, not trained

        # Set up demonstrations dataset
        self.e_batch_size = min(len(self.expert_dataset), self.hps.batch_size)
        self.e_dataloader = DataLoader(
            self.expert_dataset,
            self.e_batch_size,
            shuffle=True,
            drop_last=True,
        )
        assert len(self.e_dataloader) > 0

        # Create optimizer
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=self.hps.red_lr)

        # Define reward normalizer
        self.rms_pred_losses = RunMoms(shape=(1,), use_mpi=False)

        log_module_info(logger, 'RED Pred Network', self.pred_net)
        log_module_info(logger, 'RED Targ Network', self.targ_net)

    def remove_absorbing(self, x):
        non_absorbing_rows = []
        for j, row in enumerate([x[i, :] for i in range(x.shape[0])]):
            if torch.all(torch.eq(row, torch.cat([torch.zeros_like(row[0:-1]),
                                                  torch.Tensor([1.]).to(self.device)], dim=-1))):
                # logger.info("removing absorbing row (#{})".format(j))
                pass
            else:
                non_absorbing_rows.append(j)
        return x[non_absorbing_rows, :], non_absorbing_rows

    def train(self):
        """Update the RED predictor network"""

        # Container for all the metrics
        metrics = defaultdict(list)

        for e_batch in self.e_dataloader:

            # Transfer to device
            e_input_a = e_batch['obs0'].to(self.device)
            if self.hps.state_only:
                e_input_b = e_batch['obs1'].to(self.device)
            else:
                e_input_b = e_batch['acs'].to(self.device)

            if self.hps.red_batch_norm:
                # Update running moments for observations
                _e_input_a = e_input_a.clone()
                if self.hps.wrap_absorb:
                    _e_input_a = self.remove_absorbing(_e_input_a)[0][:, 0:-1]
                self.pred_net.rms_obs.update(_e_input_a)
                self.targ_net.rms_obs.update(_e_input_a)

            # Compute loss
            _loss = F.mse_loss(
                self.pred_net(e_input_a, e_input_b),
                self.targ_net(e_input_a, e_input_b),
                reduction='none',
            )
            # Compute rewards for different loss scales (monitoring purposes only)
            e_pde = {'scale: {}'.format(str(10**i)): torch.exp(-10**i * _loss) for i in range(6)}
            # Only use the desired proportion of experience per update
            loss = _loss.mean(dim=-1)
            mask = loss.clone().detach().data.uniform_().to(self.device)
            mask = (mask < self.hps.proportion_of_exp_per_red_update).float()
            loss = (mask * loss).sum() / torch.max(torch.Tensor([1.]), mask.sum())
            metrics['loss'].append(loss)

            # Update running moments
            pred_losses = F.mse_loss(
                self.pred_net(e_input_a, e_input_b),
                self.targ_net(e_input_a, e_input_b),
                reduction='none',
            ).mean(dim=-1, keepdim=True).detach()
            self.rms_pred_losses.update(pred_losses)

            # Update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        metrics = {k: torch.stack(v).mean().cpu().data.numpy() for k, v in metrics.items()}
        return metrics, e_pde

    def get_syn_rew(self, input_a, input_b):
        # Compute synthetic reward
        pred_losses = F.mse_loss(
            self.pred_net(input_a, input_b),
            self.targ_net(input_a, input_b),
            reduction='none',
        ).mean(dim=-1, keepdim=True).detach()
        # Normalize synthetic
        pred_losses = self.rms_pred_losses.divide_by_std(pred_losses)
        syn_rews = (-pred_losses).exp()
        return syn_rews
