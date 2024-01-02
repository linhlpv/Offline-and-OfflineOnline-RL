import torch
from math import sqrt
from torch import nn
from torch.distributions import Distribution, Normal
import torch.nn.functional as F
import math

import numpy as np
from typing import Tuple  


class VectorizedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, ensemble_size):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ensemble_size = ensemble_size
        
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_dim, out_dim))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_dim))
        self.reset_parameters()
        
    def reset_parameters(self,):
        # default torch init for nn.Linear
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=sqrt(5))
            
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        # input: (ensemble_size, batch_size, in_dim)
        
        return x @ self.weight + self.bias
    
class Actor(nn.Module):
    def __init__(
        self, state_dim, action_dim, hidden_dim, max_action=1.0
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
        )
        
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # init as in the EDAC paper
        for layer in self.net[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)
        torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_std.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_std.bias, -1e-3, 1e-3)
        
        self.action_dim = action_dim
        self.max_action = max_action
        
    def forward(self, state, deterministic=False, need_log_prob=False):
        hidden = self.net(state)
        mu, log_std = self.mu(hidden), self.log_std(hidden)
        
        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_std = torch.clip(log_std, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_std))
        
        if deterministic:
            action = mu
        else:
            action = policy_dist.rsample()
            
        tanh_action, log_prob = torch.tanh(action), None
        if need_log_prob:
            # Change of the variables formular (SAC appendix C eq 21)
            log_prob = policy_dist.log_prob(action).sum(-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(-1)
        
        return tanh_action * self.max_action, log_prob 
    
    @torch.no_grad()
    def act(self, state, device):
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action = self(state, deterministic=True)[0].cpu().numpy()
        return action
    
    
class VectorizedCritic(nn.Module):
    def __init__(
        self, state_dim, action_dim, hidden_dim, num_critics
    ):
        super().__init__()
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, num_critics),
        )
        
        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)
        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)
        
        self.num_critics = num_critics
        
    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=-1)
        state_action = state_action.unsqueeze(0).repeat_interleave(
            self.num_critics, dim=0
        )
        q_values = self.critic(state_action).squeeze(-1)
        return q_values