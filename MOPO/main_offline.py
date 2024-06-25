import argparse
import os  
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import sys
import random 

import gym
import d4rl 
import numpy as np
import torch 

from modules import MLP, Actor, Critic, TanhDiagGaussian
from ensemble_dynamics_model import EnsembleDynamicsModel
from dynamics import EnsembleDynamics
from utils import StandardScaler, get_termination_fn, qlearning_dataset 
from buffer import ReplayBuffer
from logger import Logger, make_log_dirs
from trainer import Trainer
from mopo import MOPOPolicy
import yaml

import pyrallis
from typing import List, Tuple
import datetime
from dataclasses import asdict, dataclass
from pyrallis import field

"""
suggested hypers

halfcheetah-medium-v2: rollout-length=5, penalty-coef=0.5
hopper-medium-v2: rollout-length=5, penalty-coef=5.0
walker2d-medium-v2: rollout-length=5, penalty-coef=0.5
halfcheetah-medium-replay-v2: rollout-length=5, penalty-coef=0.5
hopper-medium-replay-v2: rollout-length=5, penalty-coef=2.5
walker2d-medium-replay-v2: rollout-length=1, penalty-coef=2.5
halfcheetah-medium-expert-v2: rollout-length=5, penalty-coef=2.5
hopper-medium-expert-v2: rollout-length=5, penalty-coef=5.0
walker2d-medium-expert-v2: rollout-length=1, penalty-coef=2.5
"""

@dataclass
class TrainConfig:
    project: str = "off_offon_debug_1"
    group: str = ""
    name: str = "MOPO"
    algo_name: str = "mopo"
    env: str = "walker2d-medium-expert-v2"
    seed: int = 1
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    hidden_dims: List[int] = field(default=[256, 256], is_mutable=True)
    obs_shape: Tuple[int] = None
    action_dim: int = None
    max_action: float = None

    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: int = None
    alpha_lr: float = 1e-4
    dynamics_lr: float = 1e-3
    dynamics_hidden_dim: List[int] = field(default=[200, 200, 200, 200], is_mutable=True)
    dynamics_weight_decay: List[float] = field(default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4], is_mutable=True)
    n_ensemble: int = 7
    n_elites: int = 5
    rollout_freq: int = 1000
    rollout_bactch_size = 50000
    rollout_length: int = 1
    penalty_coef: float = 2.5
    model_retain_epochs: int = 5
    real_ratio: float = 0.05
    load_dynamics_path: str = None
    epoch: int = 3000
    step_per_epoch: int = 1000
    eval_episodes: int = 10
    batch_size: int = 256 
    device: str = "cuda"

    # Logger
    use_wandb: bool = True
    is_test: bool = False
    checkpoints_path: str = "../checkpoints"

    def __post_init__(self):
        if self.is_test:
            self.checkpoints_path = "../test_checkpoints/"
        self.name = f"{self.name}--{self.env}--{self.seed}--{str(datetime.datetime.now())}"
        if self.checkpoints_path is None:
            self.checkpoints_path = "./checkpoints/"
        if self.checkpoints_path is not None:
            os.makedirs(self.checkpoints_path, exist_ok=True)
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.project)
            os.makedirs(self.checkpoints_path, exist_ok=True)
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.group)
            os.makedirs(self.checkpoints_path, exist_ok=True)
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
            os.makedirs(self.checkpoints_path, exist_ok=True)


@pyrallis.wrap()
def train(config: TrainConfig):
    # create env and dataset
    env = gym.make(config.env)
    dataset = qlearning_dataset(env)
    config.obs_shape = env.observation_space.shape
    config.action_dim = np.prod(env.action_space.shape)
    config.max_action = env.action_space.high[0]

    # seed
    random.seed(config.seed)
    np.random.seed(config.seed) 
    torch.manual_seed(config.seed)  
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(config.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(config.obs_shape), hidden_dims=config.hidden_dims)
    critic_1_backbone = MLP(input_dim=np.prod(config.obs_shape) + config.action_dim, hidden_dims=config.hidden_dims)
    critic_2_backbone = MLP(input_dim=np.prod(config.obs_shape) + config.action_dim, hidden_dims=config.hidden_dims)
    print(actor_backbone)
    print(getattr(actor_backbone, "output_dim"))
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=config.action_dim,
        unbounded=True,
        conditional_sigma=True,
        max_mu=config.max_action
    )
    actor = Actor(actor_backbone, dist, config.device).to(config.device)
    critic_1 = Critic(critic_1_backbone, config.device).to(config.device)
    critic_2 = Critic(critic_2_backbone, config.device).to(config.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_1_optim = torch.optim.Adam(critic_1.parameters(), lr=config.critic_lr)
    critic_2_optim = torch.optim.Adam(critic_2.parameters(), lr=config.critic_lr)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, config.epoch)

    if config.auto_alpha:
        target_entropy = config.target_entropy if config.target_entropy else -np.prod(env.action_space.shape)
        config.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=config.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=config.alpha_lr)
        alpha = [target_entropy, log_alpha, alpha_optim]
    
    else:
        alpha = config.alpha
    
    # create dynamics model
    load_dynamics_model = True if config.load_dynamics_path else False
    dynamics_model = EnsembleDynamicsModel(
        obs_dim = np.prod(config.obs_shape),
        action_dim = config.action_dim,
        hidden_dims = config.dynamics_hidden_dim,
        num_ensemble = config.n_ensemble,
        num_elites = config.n_elites,
        weight_decays = config.dynamics_weight_decay,
        device = config.device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr = config.dynamics_lr
    )
    scaler = StandardScaler()
    termination_fn = get_termination_fn(task=config.env)
    dynamics = EnsembleDynamics(
        model = dynamics_model,
        optim = dynamics_optim,
        scaler = scaler,
        terminal_fn = termination_fn,
        penalty_coef = config.penalty_coef,
    )

    if config.load_dynamics_path:
        dynamics.load(config.load_dynamics_path)

    # create MOPO
    policy =  MOPOPolicy(
        dynamics=dynamics,
        actor=actor,
        critic_1=critic_1,
        critic_2=critic_2,
        actor_optim=actor_optim,
        critic_1_optim=critic_1_optim,
        critic_2_optim=critic_2_optim,
        tau=config.tau,
        gamma=config.gamma,
        alpha=alpha
    )

    # create buffer
    real_buffer = ReplayBuffer(
        buffer_size=len(dataset['observations']),
        obs_shape=config.obs_shape,
        obs_dtype=np.float32,
        action_dim=config.action_dim,
        action_dtype=np.float32,
        device=config.device
    )
    real_buffer.load_dataset(dataset)
    fake_buffer = ReplayBuffer(
        buffer_size = config.rollout_bactch_size*config.rollout_length*config.model_retain_epochs,
        obs_shape=config.obs_shape,
        obs_dtype=np.float32,
        action_dim=config.action_dim,
        action_dtype=np.float32,
        device=config.device
    )

    # log
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            yaml.dump(asdict(config), f)
    logger = Logger(config.checkpoints_path, asdict(config), use_wandb=config.use_wandb)

    # create policy trainer
    policy_trainer = Trainer(
        policy=policy,
        eval_env=env,
        real_buffer=real_buffer,
        fake_buffer=fake_buffer,
        logger=logger,
        rollout_setting=(config.rollout_freq, config.rollout_bactch_size, config.rollout_length),
        epoch=config.epoch,
        step_per_epoch=config.step_per_epoch,
        batch_size=config.batch_size,
        real_ratio=config.real_ratio,
        eval_episodes=config.eval_episodes,
        lr_scheduler=lr_scheduler,
    )

    # train
    if not load_dynamics_model:
        dynamics.train(real_buffer.sample_all(), logger, max_epochs_since_update=5)

    policy.train()

if __name__ == "__main__":
    train()