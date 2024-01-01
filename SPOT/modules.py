import torch
from math import sqrt
from torch import nn
from torch.distributions import Distribution, Normal
import torch.nn.functional as F

import numpy as np
from typing import Tuple  

MIN_LOG_STD = -20.0
MAX_LOG_STD = 2.0

def weights_init(module, init_w=3e-3):
    if isinstance(module, nn.Linear):
        module.weight.data.uniform_(-init_w, init_w)
        module.bias.data.uniform_(-init_w, init_w)
        

class VAE(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        latent_dim, 
        max_action,
        device,
        hidden_dim=750,
    ):
        super().__init__()
        self.device=device
        if latent_dim is None:
            latent_dim = 2 * action_dim 
        self.encoder_shared = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.max_action = max_action
        self.latent_dim = latent_dim
        
    def encode(self, state, action):
        z = self.encoder_shared(torch.cat([state, action], dim=-1))
        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        return mean, std
    
    def decode(self, state, z):
        # clip the latent code to the range [-0.5, 0.5] when sample from the VAE
        if z is None:
            z = (torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5))
        x = torch.cat([state, z], -1)
        return self.max_action * self.decoder(x)
        
    def forward(self, state, action):
        mean, std = self.encode(state, action)
        z = mean + std * torch.randn_like(std)
        u = self.decode(state, z)
        return u, mean, std
        
        
    def importance_sampling_estimator(
        self, state, action, beta, num_samples=500,
    ):
        mean, std = self.encode(state, action)
        
        mean_enc = mean.repeat(num_samples, 1, 1).permute(1, 0, 2) # B X n_S X D
        std_enc = std.repeat(num_samples, 1, 1).permute(1, 0, 2) # B X n_S X D
        z = mean_enc + std_enc * torch.randn_like(std_enc) # B X n_S X D
        
        state = state.repeat(num_samples, 1, 1).permute(1, 0, 2) # B X n_S X D
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2) # B X n_S X D
        mean_dec = self.decode(state, z)
        std_dec = np.sqrt(beta/4)
        
        # q(z|x)
        log_qzx = torch.distribution.Normal(loc=mean_enc, scale=std_enc).log_prob(z)
        # p(z)
        mu_prior = torch.zeros_like(z).to(self.device)
        std_prior = torch.ones_like(z).to(self.device)
        log_pz = torch.distribution.Normal(loc=mu_prior, scale=std_prior).log_prob(z)
        # p(x|z)
        std_dec = std_dec * torch.ones_like(mean_dec).to(self.device)
        log_pxz = torch.distribution.Normal(loc=mean_dec, scale=std_dec).log_prob(action)
        
        w = log_pxz.sum(-1) + log_pz.sum(-1) - log_qzx.sum(-1) 
        ll = w.logsumexp(-1) - np.log(num_samples)
        return ll
    
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, init_w=None):
        super().__init__()
        head = nn.Linear(256, action_dim)
        if init_w is not None:
            weights_init(head, init_w)
            
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            head,
            nn.Tanh(),
        )
        
        self.max_action = max_action
        
    def forward(self, state):
        return self.net(state) *  self.max_action
    
    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()
        

class DoubleQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, init_w=None):
        super().__init__()
        
        head_1 = nn.Linear(256, 1)
        head_2 = nn.Linear(256, 1)
        if init_w is not None:
            weights_init(head_1, init_w)
            weights_init(head_2, init_w)
        
        self.qf1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            head_1,
        )
        self.qf2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            head_2,
        )
        
    def both(self, state, action):
        return self.qf1(torch.cat([state, action], dim=-1)), self.qf2(torch.cat([state, action], dim=-1))
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return torch.min(*(self.qf1(x), self.qf2(x)))
    
