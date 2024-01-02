import os
import gym
import d4rl
import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, List, Tuple
import copy

from utils import asymmetric_l2_loss, soft_update, TensorBatch, ENVS_WITH_GOAL, EXP_ADV_MAX

class SAC_N:
    def __init__(
        self,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic: nn.Module,
        critic_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        tau: float = 0.005,
        alpha_lr: float = 1e-4,
        device: str = "cpu",
    ):
        self.device = device 
        
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.critic = critic
        self.critic_optimizer = critic_optimizer 
        self.target_critic = copy.deepcopy(self.critic).to(device)
        
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        
        # adaptive alpha setup
        self.target_entropy = -float(self.actor.action_dim)
        self.log_alpha = torch.tensor(
            [0.0], dtype=torch.float32, requires_grad=True, device=self.device
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp().detach()
        
    def update_alpha(self, state, log_dict):
        with torch.no_grad():
            action, log_prob = self.actor(state, need_log_prob=True)
        
        loss = (-self.log_alpha *(log_prob + self.target_entropy)).mean()
        log_dict["alpha_loss"] = loss.item()
        self.alpha_optimizer.zero_grad()
        loss.backward()
        self.alpha_optimizer.step()
        return log_dict

    def update_actor(self, state, log_dict):
        action, log_prob = self.actor(state, need_log_prob=True)
        q_value_dist = self.critic(state, action)
        assert q_value_dist.shape[0] == self.critic.num_critics
        q_value_min = q_value_dist.min(0).values 
        
        q_value_std = q_value_dist.std(0).mean().item()
        batch_entropy = -log_prob.mean().item()
        assert log_prob.shape == q_value_min.shape
        loss = (self.alpha * log_prob - q_value_min).mean()
        log_dict["actor_loss"] = loss.item()
        log_dict["q_value_std"] = q_value_std
        log_dict["batch_entropy"] = batch_entropy
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        
        return log_dict
    
    def update_critic(self, state, action, reward, next_state, done, log_dict):
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(next_state, need_log_prob=True)
            q_next = self.target_critic(next_state, next_action).min(0).values
            q_next = q_next - self.alpha * next_action_log_prob
            
            assert q_next.unsqueeze(-1).shape == done.shape == reward.shape
            q_target = reward + self.discount * (1 - done) * q_next.unsqueeze(-1)
        
        q_values = self.critic(state, action)
        
        loss = ((q_values - q_target.view(1, -1)) ** 2).mean(dim=1).sum(dim=0)
        log_dict["critic_loss"] = loss.item()
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        return log_dict
                    
        
    def train(self, batch: TensorBatch):
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        log_dict = self.update_alpha(observations, log_dict)
        log_dict = self.update_actor(observations, log_dict)
        log_dict = self.update_critic(observations, actions, rewards, next_observations, dones, log_dict)
        with torch.no_grad():
            soft_update(self.target_critic, self.critic, self.tau)
            max_action = self.actor.max_action
            random_actions = -max_action + 2 * max_action * torch.rand_like(actions)
            q_random_std = self.critic(observations, random_actions).std(0).mean().item()
            log_dict["q_random_std"] = q_random_std
        
        return log_dict

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.item(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.target_critic = copy.deepcopy(self.actor)

        self.critic.load_state_dict(state_dict["critic"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])

        self.log_alpha[0] = state_dict["log_alpha"]
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
        

        self.total_it = state_dict["total_it"]