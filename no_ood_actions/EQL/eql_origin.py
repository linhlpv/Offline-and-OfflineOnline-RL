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

from utils import asymmetric_l2_loss, soft_update, TensorBatch, ENVS_WITH_GOAL, EXP_ADV_MAX, ADV_MAX

class ExpQLearning:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        alpha: float = 2.0,
        beta: float = 10.0,
        max_steps: int = 1000000,
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
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
 
        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        self.beta = beta

        self.total_it = 0
        self.device = device
        
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

        # with torch.no_grad():
        #     next_v = self.vf(next_observations)
        # Update value function
        self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update actor
        self._update_policy(observations, actions, log_dict)
        # Update Q function
        self._update_q(observations, actions, rewards, next_observations, dones, log_dict)


        return log_dict

    def _update_v(self, observations, actions, log_dict):
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = ((target_q - v) / self.alpha).clamp_max(5.0) # taken from the original paper,
        # apply max normalized exp trick for stable compute the exp. Take it from the original implementation in jax
        max_adv = torch.max(adv, dim=0).values.detach().clamp_min(-1.0)
        
        v_loss = (torch.exp(adv - max_adv) + torch.exp(-max_adv) * v / self.alpha).mean()
        
        # exp_term = torch.exp(adv / self.alpha)
        # v_loss = (exp_term  + v / self.alpha).mean()
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
            next_v = self.vf(next_observations)
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        log_dict: Dict,
    ):
        # compute adv using the updated V and Q_target
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v.detach()
        # exp_term = torch.exp(adv.detach() / self.alpha)
        exp_term = torch.exp(self.beta * adv / self.alpha)
        exp_term = exp_term.clamp(0, ADV_MAX)
        
        policy_out = self.actor(observations)
        log_prob = -policy_out.log_prob(actions).sum(-1, keepdim=False)
        
        policy_loss = torch.mean(exp_term * log_prob)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()
    

    def state_dict(self):
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
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
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]