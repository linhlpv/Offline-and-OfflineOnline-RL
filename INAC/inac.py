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

from utils import soft_update, TensorBatch, ENVS_WITH_GOAL, EXP_ADV_MAX, ADV_MAX, ADV_MIN

class InAC:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        behavior_policy: nn.Module,
        behavior_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        inac_tau: float = 0.33,
        discount: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.behavior = behavior_policy
        self.behavior_optimizer = behavior_optimizer
        
        self.discount = discount
        self.tau = tau
        self.inac_tau = inac_tau

        self.total_it = 0
        self.device = device
        
    def _update_behavior(self, observations, actions, log_dict):
        behavior_loss = self.behavior.log_prob(observations, actions).mean()
        log_dict['behavior'] = behavior_loss.item()
        self.behavior_optimizer.zero_grad()
        behavior_loss.backward()
        self.behavior_optimizer.step()

    def _update_v(self, observations, log_dict):
        # Update value function
        with torch.no_grad():
            actions, log_probs = self.actor(observations)
            target_q = self.q_target(observations, actions)
            target_q = target_q - self.inac_tau * log_probs

        v = self.vf(observations)
        v_loss = F.mse_loss(v, target_q)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

    def _update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        with torch.no_grad():
            # based on the author implementation to compute target 
            next_actions, next_log_probs = self.actor(next_observations)
            target = self.q_target(next_observations, next_actions)
            target = target - self.inac_tau * next_log_probs 
            # # based on the paper Eq (16)
            # target = self.vf(next_observations)
            target = rewards + self.discount * (1.0 - terminals.float()) * target
            
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, target) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        # soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        log_prob = self.actor.log_prob(observations, actions)
        with torch.no_grad():
            target_q = self.qf(observations, actions)
            v = self.vf(observations)
            behavior_log_prob = self.behavior.log_prob(observations, actions)
            exp_adv = torch.exp((target_q - v) / self.inac_tau - behavior_log_prob).clamp(ADV_MIN, ADV_MAX)    
        
        policy_loss = torch.mean(-exp_adv * log_prob)
        log_dict["actor_loss"] = policy_loss.item()
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

        # Update behavior policy
        self._update_behavior(observations, actions, log_dict)
        # Update value function
        adv = self._update_v(observations, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(observations, actions, rewards, next_observations, dones, log_dict)
        # Update actor
        self._update_policy(observations, actions, log_dict)
        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)


        return log_dict

    def state_dict(self):
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "behavior": self.behavior.state_dict(),
            "behavior_optimizer": self.behavior_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.behavior_optimizer.load_state_dict(state_dict["behavior_optimizer"])
        self.behavior.load_state_dict(state_dict["behavior"])
        self.total_it = state_dict["total_it"]