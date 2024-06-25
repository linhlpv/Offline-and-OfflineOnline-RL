import numpy as np 
import torch 
import torch.nn as nn 
import gym 
import copy 
from copy import deepcopy

import torch.nn.functional as F
from typing import Dict, Union, Tuple
from collections import defaultdict
from dynamics import BaseDynamics

class BasePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def train(self) -> None:
        raise NotImplementedError

    def eval() -> None:
        raise NotImplementedError
    
    def select_action(self, obs: np.ndarray, deterministic: bool=False) -> np.ndarray:
        raise NotImplementedError
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        raise NotImplementedError
    
class SACPolicy(BasePolicy):
    """
    SAC implementation
    """
    def __init__(self, actor: nn.Module, critic_1: nn.Module, critic_2: nn.Module, actor_optim: torch.optim.Optimizer, critic_1_optim: torch.optim.Optimizer, critic_2_optim: torch.optim.Optimizer, tau: float=0.005, gamma: float=0.99, alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2) -> None:
        super().__init__()

        self.actor = actor
        self.critic_1, self.critic_1_target = critic_1, deepcopy(critic_1)
        self.critic_1_target.eval()
        self.critic_2, self.critic_2_target = critic_2, deepcopy(critic_2)
        self.critic_2_target.eval()

        self.actor_optim = actor_optim
        self.critic_1_optim = critic_1_optim
        self.critic_2_optim = critic_2_optim

        self._tau = tau
        self._gamma = gamma

        self._is_auto_alpha = False
        if isinstance(alpha, tuple):
            self._is_auto_alpha = True
            self._target_entropy, self._log_alpha, self.alpha_optim = alpha
            self._alpha = self._log_alpha.detach().exp()
        else:
            self.alpha = alpha

    
    def train(self) -> None:
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()

    def soft_update(self) -> None:
        for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)
        
        for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

    def action_forward(self, obs: torch.Tensor, deterministic: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor(obs)
        if deterministic:
            squashed_action, raw_action = dist.mode()
        else:
            squashed_action, raw_action = dist.rsample()
        log_prob = dist.log_prob(squashed_action, raw_action)
        return squashed_action, log_prob
    
    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool=False) -> np.ndarray:
        action, _ = self.action_forward(obs, deterministic)
        return action.cpu().numpy()
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        obss, actions, next_obss, rewards, terminals = batch["observations"], batch["actions"], batch["next_observations"], batch["rewards"], batch["terminals"]

        # update critic
        q1, q2 = self.critic_1(obss, actions), self.critic_2(obss, actions)
        with torch.no_grad():
            next_actions, next_log_prob = self.action_forward(next_obss)
            next_q = torch.min(self.critic_1_target(next_obss, next_actions), self.critic_2_target(next_obss, next_actions)) - self._alpha * next_log_prob
            target_q = rewards + self._gamma * (1 - terminals) * next_q

        critic_1_loss = ((q1 - target_q).pow(2)).mean()
        self.critic_1_optim.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optim.step()

        critic_2_loss = ((q2 - target_q).pow(2)).mean()
        self.critic_2_optim.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optim.step()

        # update actor
        a, log_probs = self.action_forward(obss)
        current_q1, current_q2 = self.critic_1(obss, a), self.critic_2(obss, a)

        actor_loss = - torch.min(current_q1, current_q2).mean() + self._alpha * log_probs.mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = torch.clamp(self._log_alpha.detach().exp(), 0.0, 1.0)

        self.soft_update()
        result = {
            "actor_loss": actor_loss.item(),
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item()
        }
        if self._is_auto_alpha:
            result["alpha_loss"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()
        
        return result

class MOPOPolicy(SACPolicy):
    """
    MOPO implementation
    """
    def __init__(self, dynamics: BaseDynamics, actor: nn.Module, critic_1: nn.Module, critic_2: nn.Module, actor_optim: torch.optim.Optimizer, critic_1_optim: torch.optim.Optimizer, critic_2_optim: torch.optim.Optimizer, tau: float=0.005, gamma: float=0.99, alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]]=0.2) -> None:
        super().__init__(actor, critic_1, critic_2, actor_optim, critic_1_optim, critic_2_optim, tau, gamma, alpha)

        self.dynamics = dynamics

    def rollout(self, init_obss: np.ndarray, rollout_length: int) -> Tuple[Dict[str, np.ndarray], Dict]:
        num_transitions = 0
        rewards_arr = np.array([])
        rollout_transitions = defaultdict(list)

        # rollout
        observations = init_obss 
        for _ in range(rollout_length):
            actions = self.select_action(observations)
            next_observations, rewards, terminals, _ = self.dynamics.predict(observations, actions)
            rollout_transitions['obss'].append(observations)
            rollout_transitions['next_obss'].append(next_observations)
            rollout_transitions['actions'].append(actions)
            rollout_transitions['rewards'].append(rewards)
            rollout_transitions['terminals'].append(terminals)

            num_transitions += len(observations)
            rewards_arr = np.append(rewards_arr, rewards.flatten())

            nonterm_mask = (~terminals).flatten()
            if nonterm_mask.sum() == 0:
                break

            observations = next_observations

        for k, v in rollout_transitions.items():
            rollout_transitions[k] = np.concatenate(v, axis=0)

        return rollout_transitions, {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}
    
    def learn(self, batch: Dict) -> Dict[str, float]:
        real_batch, fake_batch = batch["real"], batch["fake"]
        mix_batch = {k: torch.cat([real_batch[k], fake_batch[k]], dim=0) for k in real_batch.keys()}
        return super().learn(mix_batch)
    
    


