import torch 
import numpy as np 
import sys
import os
import random 

from torch import nn 
import torch.nn.functional as F
from torch.distributions import Normal 
from typing import List, Tuple, Union, Dict, Optional


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    
def soft_clamp(x, min=None, max=None):
    """
    Clamp tensor while maintaining the gradient
    """
    if max is not None:
        x = max - F.softplus(max - x)
    if min is not None:
        x = min + F.softplus(x - min)

    return x

class EnsembleLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_ensemble: int, weight_decay: float=0.0) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble

        self.register_parameter("weight", nn.Parameter(torch.zeros(num_ensemble, input_dim, output_dim)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(num_ensemble, 1, output_dim)))

        nn.init.trunc_normal_(self.weight, std=1/(2*input_dim**0.5))

        self.register_parameter("saved_weight", nn.Parameter(self.weight.detach().clone()))
        self.register_parameter("saved_bias", nn.Parameter(self.bias.detach().clone()))

        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias 

        if len(x.shape) == 2:
            x = torch.einsum('ij, bjk->bik', x, weight)
        else:
            x = torch.einsum("bij, bjk->bik", x, weight)
        
        x = x + bias 

        return x

    def load_save(self) -> None:
        self.weight.data.copy_(self.saved_weight.data)
        self.bias.data.copy_(self.saved_bias.data)

    def update_save(self, indexes: List[int]) -> None:
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]

    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = self.weight_decay * (0.5 *((self.weight**2).sum()))
        return decay_loss
    
class EnsembleDynamicsModel(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Union[List[int], Tuple[int]], num_ensemble: int=7, num_elites: int=5, activation: nn.Module=Swish, weight_decays: Optional[Union[List[float], Tuple[float]]]=None, with_reward: bool=True, device: str="cpu"):
        super().__init__()

        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        self._with_reward = with_reward
        self.device = torch.device(device)

        self.activation = activation()

        assert len(weight_decays) == (len(hidden_dims) + 1)

        module_list = []
        hidden_dims = [obs_dim + action_dim] + list(hidden_dims)
        if weight_decays is None:
            weight_decays = [0.0] * (len(hidden_dims) + 1)
        
        for in_dim, out_dim, weight_decay in zip(hidden_dims[:-1], hidden_dims[1:], weight_decays[:-1]):
            module_list.append(EnsembleLinear(in_dim, out_dim, num_ensemble, weight_decay))
        self.backbone = nn.Sequential(*module_list)

        self.output_layer = EnsembleLinear(
            hidden_dims[-1],
            2 * (obs_dim + self._with_reward),
            num_ensemble,
            weight_decays[-1]
        )
        
        self.register_parameter("max_logvar", nn.Parameter(torch.ones(obs_dim + self._with_reward) * 0.5, requires_grad=True))
        self.register_parameter("min_logvar", nn.Parameter(torch.ones(obs_dim + self._with_reward) * -10, requires_grad=True))

        self.register_parameter("elites", nn.Parameter(torch.tensor(list(range(0, self.num_elites))), requires_grad=False))

        self.to(device)

    def forward(self, obs_action: np.ndarray):
        obs_action = torch.as_tensor(obs_action, dtype=torch.float32).to(self.device)
        output = obs_action
        for layer in self.backbone:
            output = self.activation(layer(output))
        
        mean, logvar = torch.chunk(self.output_layer(output), 2, dim=-1)
        logvar = soft_clamp(logvar, min=self.min_logvar, max=self.max_logvar)
        return mean, logvar
    
    def load_save(self):
        for layer in self.backbone:
            layer.load_save()
        
        self.output_layer.load_save()

    def update_save(self, indexes: List[int]):
        for layer in self.backbone:
            layer.update_save(indexes)
        
        self.output_layer.update_save(indexes)

    def get_decay_loss(self):
        decay_loss = 0
        for layer in self.backbone:
            decay_loss += layer.get_decay_loss()
        decay_loss += self.output_layer.get_decay_loss()
        return decay_loss
    
    def set_elites(self, indexes: List[int]):
        assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
        self.register_parameter('elites', nn.Parameter(torch.tensor(indexes), requires_grad=False))

    def random_elite_idxs(self, batch_size: int):
        idxs = np.random.choice(self.elites.data.cpu().numpy(), size=batch_size)
        return idxs
    