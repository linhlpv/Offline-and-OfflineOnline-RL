import torch
import numpy as np 
import sys
import os 
import random

from torch  import nn 
import torch.nn.functional as F
from torch.distributions import Normal
from typing import List, Tuple, Union, Optional, Dict

### Backbone modules
class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super(Squeeze, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.squeeze(self.dim)
    

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Union[List[int], Tuple[int]], output_dim: Optional[int]=None, activation: nn.Module=nn.ReLU, dropout_rate: Optional[float]=None) -> None:
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model += [nn.Linear(in_dim, out_dim), activation()]
            if dropout_rate is not None:
                model += [nn.Dropout(dropout_rate)]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            model += [nn.Linear(hidden_dims[-1], output_dim)]
            self.output_dim = output_dim

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
## Distributional modules, used for actor
class NormalWrapper(torch.distributions.Normal):
    def log_prob(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)
    
    def entropy(self):
        return super().entropy().sum(-1)
    
    def mode(self):
        return self.mean
    

class TanhNormalWrapper(torch.distributions.Normal):
    def __init__(self, loc, scale, max_action):
        super().__init__(loc, scale)
        self._max_action = max_action

    def log_prob(self, action, raw_action=None):
        squashed_action = action/self._max_action
        if raw_action is None:
            raw_action = torch.atanh(squashed_action)

        log_prob = super().log_prob(raw_action).sum(-1, keepdim=True)
        eps = 1e-6
        log_prob = log_prob - torch.log(self._max_action*(1 - squashed_action.pow(2)) + eps).sum(-1, keepdim=True)
        return log_prob
    
    def mode(self):
        raw_action = self.mean
        action = self._max_action * torch.tanh(self.mean)
        return action, raw_action
    
    def arctanh(self, x):
        one_plus_x = (1+x).clamp(min=1e-6)
        one_minus_x = (1-x).clamp(min=1e-6)
        return 0.5*torch.log(one_plus_x/one_minus_x)
    
    def rsample(self):
        raw_action = super().rsample()
        action = self._max_action * torch.tanh(raw_action)
        return action, raw_action
    

class DiagGaussian(nn.Module):
    def __init__(self, latent_dim, output_dim, unbounded=False, conditional_sigmal=False, max_mu=1.0, sigma_min=-5.0, sigma_max=2.0):
        super().__init__()
        self.mu = nn.Linear(latent_dim, output_dim)
        self._c_sigma = conditional_sigmal
        if self._c_sigma:
            self.sigma = nn.Linear(latent_dim, output_dim)
        else:
            self.sigma_param = nn.Parameter(torch.zeros(output_dim, 1))\
            
        self._unbounded = unbounded
        self._max = max_mu
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max

    def forward(self, logits):
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)

        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), self._sigma_min, self._sigma_max).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1 
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return NormalWrapper(mu, sigma)
    

class TanhDiagGaussian(DiagGaussian):
    def __init__(self, latent_dim, output_dim, unbounded=False, conditional_sigma=False, max_mu=1.0, sigma_min=-5.0, sigma_max=2.0):
        super().__init__(latent_dim, output_dim, unbounded, conditional_sigma, max_mu, sigma_min, sigma_max)

    def forward(self, logits):
        mu = self.mu(logits)
        if not self._unbounded:
            mu = self._max * torch.tanh(mu)
        if self._c_sigma:
            sigma = torch.clamp(self.sigma(logits), self._sigma_min, self._sigma_max).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        return TanhNormalWrapper(mu, sigma, self._max)
    
## Actors
class Actor(nn.Module):
    def  __init__(self, backbone: nn.Module, dist_net: nn.Module, device: str="cpu") -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(self.device)
        self.dist_net = dist_net.to(self.device)
    
    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.distributions.Normal:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        dist = self.dist_net(logits)
        return dist
    
class DeterministicActor(nn.Module):
    def __init__(self, backbone: nn.Module, action_dim: int, max_action: float=1.0, device: str="cpu") -> None:
        super().__init__()
        self.device = torch.device(device)
        self.backbone = backbone.to(self.device)
        latent_dim = getattr(backbone, "output_dim")
        output_dim = action_dim
        self.last = nn.Linear(latent_dim, output_dim).to(self.device)
        self._max = max_action

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        logits = self.backbone(obs)
        actions = self._max * torch.tanh(self.last(logits))
        return actions
    
## Critic
class SingleCritic(nn.Module):
    def __init__(self, backbone: nn.Module, device: str="cpu") -> None:
        super().__init__()
        self.backbone = backbone
        self.device = torch.device(device)

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))
    
class DoubleCritic(nn.Module):
    def __init__(self, backbone_1: nn.Module, backbone_2: nn.Module, device: str="cpu") -> None:
        super().__init__()
        self.backbone_1 = backbone_1
        self.backbone_2 = backbone_2
        self.device = torch.device(device)

    def both(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.backbone_1(x), self.backbone_2(x)

    def forward(self, state, action):
        return torch.min(*self.both(state, action))
    
class Critic(nn.Module):
    """
    This implementation can be used for both Q and V function
    """
    def __init__(self, backbone: nn.Module, device: str="cpu") -> None:
        super().__init__()
        self.backbone = backbone
        self.device = torch.device(device)
        latent_dim = getattr(self.backbone, "output_dim")
        self.last = nn.Linear(latent_dim, 1).to(self.device)

    def forward(self, obs: Union[np.ndarray, torch.Tensor], actions: Optional[Union[np.ndarray, torch.Tensor]]=None) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32).flatten(1)
            obs = torch.cat([obs, actions], dim=-1)
        logits = self.backbone(obs)
        values = self.last(logits)
        return values
