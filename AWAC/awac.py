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

from utils import soft_update, TensorBatch, ENVS_WITH_GOAL, EXP_ADV_MAX

class AdvantageWeightedActorCritic:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        awac_lambda: float = 1.0,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.q_optimizer = q_optimizer
        self.discount = discount
        self.awac_lambda = awac_lambda
        self.tau = tau
        self.device = device
        self.total_it = 0
        
    def update_critic(self, observations, actions, rewards, next_observations, dones, log_dict):
        with torch.no_grad():
            next_actions, _ = self.actor(next_observations)
            q_next = self.q_target(next_observations, next_actions)
            targets = rewards + self.discount * (1.0 - dones) * q_next
        
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
    def update_actor(self, observations, actions, log_dict):
        with torch.no_grad():
            next_actions, _ = self.actor(observations)
            v = self.qf(observations, next_actions)
            q = self.qf(observations, actions)
            adv = q - v 
            weights = torch.clamp_max(
                torch.exp(adv / self.awac_lambda), EXP_ADV_MAX
            )
        
        action_log_prob = self.actor.log_prob(observations, actions)
        policy_loss = -(weights * action_log_prob).mean()
        log_dict["policy_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
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
        self.update_critic(observations, actions, rewards, next_observations, dones, log_dict)
        self.update_actor(observations, actions, log_dict)
        soft_update(self.q_target, self.qf, self.tau)
        return log_dict
    
    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "qf": self.qf.state_dict(),
            "q_target": self.q_target.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.qf.load_state_dict(state_dict["qf"])
        self.q_target.load_state_dict(state_dict["q_target"])