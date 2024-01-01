import numpy as np
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import copy
from utils import TensorBatch, soft_update


class SPOT:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        double_q_network: nn.Module,
        double_q_network_optimizer: torch.optim.Optimizer,
        vae: nn.Module,
        vae_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float=0.5,
        policy_freq: int = 2,
        beta: float = 0.5,
        lambd: float = 1.0,
        num_samples: int = 1,
        iwae: bool = False,
        lambd_cool: bool = False,
        lambd_end: float = 0.2,
        max_online_steps: int = 1000000,
        device: str = "cpu",
    ):
        self.actor = actor.to(device)
        self.actor_optimizer = actor_optimizer
        self.actor_target = copy.deepcopy(actor).requires_grad_(False).to(device)
        self.qf = double_q_network.to(device)
        self.qf_optimizer = double_q_network_optimizer
        self.q_target = copy.deepcopy(double_q_network).requires_grad_(False).to(device)
        
        self.vae = vae.to(device)
        self.vae_optimizer = vae_optimizer
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_fred = policy_freq
        self.beta = beta
        self.lambd = lambd
        self.num_samples = num_samples
        self.iwae = iwae
        self.lambd_cool = lambd_cool
        self.lambd_end = lambd_end
        
        self.max_online_steps = max_online_steps
        
        self.is_online = False
        self.online_it = 0
        
        self.total_it = 0
        self.device = device
        
    def elbo_loss(self, state, action, beta, num_samples=1):
        mean, std = self.vae.encode(state, action)
        
        mean_s = mean.repeat(num_samples, 1, 1).permute(1, 0, 2) # (batch_size, num_samples, latent_dim)
        std_s = std.repeat(num_samples, 1, 1).permute(1, 0, 2) # (batch_size, num_samples, latent_dim)
        z = mean_s + std_s * torch.randn_like(std_s)
        
        state = state.repeat(num_samples, 1, 1).permute(1, 0, 2) # (batch_size, num_samples, state_dim)
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2) # (batch_size, num_samples, action_dim)
        u = self.vae.decode(state, z)
        recon_loss = ((u - action) ** 2).mean(dim=(1, 2))
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(-1)
        vae_loss = recon_loss + beta * KL_loss
        
        return vae_loss
    
    def iwae_loss(self, state, action, beta, num_samples=10):
        ll = self.vae.importance_sampling_estimator(state, action, beta, num_samples=num_samples)
        return -ll
    
    def vae_train(self, batch:TensorBatch):
        log_dict = {}
        self.total_it += 1
        
        state, action, _, _, _ = batch
        # VAE training
        recon, mean, std = self.vae(state, action)
        recon_loss = F.mse_loss(recon, action)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + self.beta * KL_loss
        
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        
        log_dict["vae/recon_loss"] = recon_loss.item()
        log_dict["vae/KL_loss"] = KL_loss.item()
        log_dict["vae/loss"] = vae_loss.item()
        
        return log_dict
    
    def train(self, batch:TensorBatch):
        log_dict = {}
        self.total_it += 1
        
        if self.is_online:
            self.online_it += 1
        
        state, action, reward, next_state, done = batch
        not_done = 1 - done
        
        with torch.no_grad():
            # Select action according to actor and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )
            
            # Compute the target Q value
            q_next = self.q_target(next_state, next_action)
            q_target = reward + self.discount * not_done * q_next
            
        # Get current Q estimate
        qs = self.qf.both(state, action)
        q_loss = sum(F.mse_loss(q, q_target) for q in qs) / len(qs)
        log_dict["critic_loss"] = q_loss.item()
        # Optimize the critic
        self.qf_optimizer.zero_grad()
        q_loss.backward()
        self.qf_optimizer.step()
        
        # Delayed actor updates
        if self.total_it % self.policy_fred == 0:
            pi_action = self.actor(state)
            q = self.qf(state, pi_action)
            
            if self.iwae:
                neg_log_beta = self.iwae_loss(state, pi_action, self.beta, self.num_samples)
            else:
                neg_log_beta = self.elbo_loss(state, pi_action, self.beta, self.num_samples)
            
            if self.lambd_cool:
                lambd = self.lambd * max(
                    self.lambd_end, (1.0 - self.online_it / self.max_online_steps)
                )
            else:
                lambd = self.lambd
                
            norm_q = 1 / q.abs().mean().detach()
            
            actor_loss = -norm_q * q.mean() + lambd * neg_log_beta.mean()
            log_dict["actor_loss"] = actor_loss.item()
            log_dict["neg_log_beta_max"] = neg_log_beta.max().item()
            log_dict["neg_log_beta_mean"] = neg_log_beta.mean().item()
            log_dict["lambd"] = lambd
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.q_target, self.qf, self.tau)
            
        return log_dict
    
    def state_dict(self):
        return {
            "vae": self.vae.state_dict(),
            "vae_optimizer": self.vae_optimizer.state_dict(),
            "qf": self.qf.state_dict(),
            "qf_optimizer": self.qf_optimizer.state_dict(),
            "q_target": self.q_target.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict):
        self.vae.load_state_dict(state_dict["vae"])
        self.vae_optimizer.load_state_dict(state_dict["vae_optimizer"])

        self.qf.load_state_dict(state_dict["qf"])
        self.qf_optimizer.load_state_dict(state_dict["qf_optimizer"])
        self.q_target.load_state_dict(state_dict["q_target"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_target.load_state_dict(state_dict["actor_target"])

        self.total_it = state_dict["total_it"]