import os  
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random
from dataclasses import asdict, dataclass 
from pathlib import Path 
from typing import Optional

import d4rl 
import gym
import torch   
import pyrallis  
import datetime  
import numpy as np

from modules import Actor, VectorizedCritic
from replay_buffer import ReplayBuffer
from logger import Logger
from sacn import SAC_N
from utils import *
from tqdm import trange

@dataclass
class TrainConfig:
    project: str = "off_offon_debug_3"
    group: str = ""
    name: str = "SAC-N"
    env: str = "halfcheetah-medium-v2"
    discount: float = 0.99
    tau: float = 0.005
    num_epochs: int = 3000
    buffer_size: int = int(2e6)
    hidden_dim: int = 256
    batch_size: int = 256
    normalize: bool = True
    normalize_reward: bool = False
    num_critics: int = 10
    alpha_lr: float = 3e-4
    qf_lr: float = 3e-4
    actor_lr: float = 3e-4
    eval_episodes: int = 10
    num_updates_on_epoch: int = 1000
    eval_every: int = 5
    deterministic_torch: bool = True
    checkpoints_path: str = "../checkpoints"
    load_model: str = "" # file name for loading a model
    seed: int = 21
    device: str = "cuda"
    log_every: int = 10
    
    # Logger
    use_wandb: bool = True
    is_test: bool = False
    
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
    set_seed(config.seed, deterministic_torch=config.deterministic_torch)
    logger = Logger(config.checkpoints_path, asdict(config), use_wandb=config.use_wandb)

    # data, evaluation, env setup
    eval_env = wrap_env(gym.make(config.env))
    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    max_action = float(eval_env.action_space.high[0])

    d4rl_dataset = d4rl.qlearning_dataset(eval_env)

    if config.normalize_reward:
        modify_reward(d4rl_dataset, config.env_name)

    buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=config.buffer_size,
        device=config.device,
    )
    buffer.load_d4rl_dataset(d4rl_dataset)

    # Actor & Critic setup
    actor = Actor(state_dim, action_dim, config.hidden_dim, max_action)
    actor.to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic = VectorizedCritic(
        state_dim, action_dim, config.hidden_dim, config.num_critics
    )
    critic.to(config.device)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=config.qf_lr
    )

    trainer = SAC_N(
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic=critic,
        critic_optimizer=critic_optimizer,
        discount=config.discount,
        tau=config.tau,
        alpha_lr=config.alpha_lr,
        device=config.device,
    )

    total_updates = 0.0
    for epoch in trange(config.num_epochs, desc="Training"):
        # training
        for _ in trange(config.num_updates_on_epoch, desc="Epoch", leave=False):
            batch = buffer.sample(config.batch_size)
            update_info = trainer.train(batch)

            if total_updates % config.log_every == 0:
                logger.log_dict({"epoch": epoch, **update_info}, int(total_updates))

            total_updates += 1

        # evaluation
        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            eval_returns = eval_actor(
                env=eval_env,
                actor=actor,
                n_episodes=config.eval_episodes,
                seed=config.seed,
                device=config.device,
            )
            eval_log = {
                "eval/reward_mean": np.mean(eval_returns),
                "eval/reward_std": np.std(eval_returns),
                "epoch": epoch,
            }
            if hasattr(eval_env, "get_normalized_score"):
                normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
                eval_return = eval_returns.mean()
                eval_log["eval/normalized_score_mean"] = np.mean(normalized_score)
                eval_log["eval/normalized_score_std"] = np.std(normalized_score)
                print("--------------------------")
                print(config.eval_episodes)
                print(f"Evaluation over {config.eval_episodes} episodes: "
                    f"{eval_return:.3f} , D4RL score: {normalized_score.mean():.3f}")
                print("--------------------------")

            logger.log_dict(eval_log, int(total_updates))

            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"{epoch}.pt"),
                )



if __name__ == "__main__":
    train()
