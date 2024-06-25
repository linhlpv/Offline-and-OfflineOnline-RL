import torch
from math import sqrt
from torch import nn
from torch.distributions import Distribution, Normal
import torch.nn.functional as F

import numpy as np
from typing import Tuple  

MIN_LOG_STD = -20.0
MAX_LOG_STD = 2.0

class Squeeze(nn.Module):
    def __init__(
        self, dim=-1
    ):
        super(Squeeze, self).__init__()
        self.dim = dim
        
    def forward(self,x):
        return x.squeeze(self.dim)

class MLP(nn.Module):
    def __init__(
        self,
        dims, 
        activation_function=nn.ReLU,
        out_activation_function=None,
        squeeze_output=False,
        dropout=0.0,
        layer_norm=False,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("n_dims must be at least 2")
        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if layer_norm:
                layers.append(nn.LayerNorm(dims[i+1]))
            else:
                layers.append(nn.Identity())
            layers.append(activation_function())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
                
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if out_activation_function is not None:
            layers.append(out_activation_function())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("dims[-1] must be 1 if squeeze_output is True")
            layers.append(Squeeze(dim=-1))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)
        
        

class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim, 
        action_dim, 
        max_action,
        hidden_dim=256,
        n_hidden=2,
        dropout=0.0,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim]*n_hidden), action_dim],
            activation_function=nn.ReLU,
            out_activation_function=nn.Tanh,
            dropout=dropout
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self.max_action = max_action
        
    def get_policy(self, state):
        mean = self.net(state)
        
        
    def forward(self, state):
        mean = self.net(state)
        std = torch.exp(self.log_std.clamp(MIN_LOG_STD, MAX_LOG_STD))
        return Normal(mean, std)    
        
    @torch.no_grad()
    def act(self, state, device="cpu"):
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float32, device=device)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()
    

class DoubleQNetwork(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        n_hidden=2,
        layer_norm=True,
    ):
        super().__init__()
        self.qf1 = MLP(
            [state_dim + action_dim, *([hidden_dim]*n_hidden), 1],
            activation_function=nn.ReLU,
            dropout=0.0,
            squeeze_output=True,
            layer_norm=layer_norm,
        )        
        self.qf2 = MLP(
            [state_dim + action_dim, *([hidden_dim]*n_hidden), 1],
            activation_function=nn.ReLU,
            dropout=0.0,
            squeeze_output=True,
            layer_norm=layer_norm,
        )
        
    def both(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.qf1(sa), self.qf2(sa)
    
    def forward(self, state, action):
        return torch.min(*self.both(state, action))
    
class ValueNetwork(nn.Module):
    def __init__(
        self, 
        state_dim,
        hidden_dim=256,
        n_hidden=2,
        layer_norm=True,
    ):
        super().__init__()
        self.vf = MLP(
            [state_dim, *([hidden_dim]*n_hidden), 1],
            activation_function=nn.ReLU,
            dropout=0.0,
            squeeze_output=True,
            layer_norm=True
        )
        
    def forward(self, state):
        return self.vf(state)