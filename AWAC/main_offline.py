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

from modules import GaussianPolicy, DoubleQNetwork
from utils import *
from replay_buffer import ReplayBuffer
from logger import Logger
from awac import AdvantageWeightedActorCritic

@dataclass
class TrainConfig:
    project: str = "off_offon_debug_1"
    group: str = ""
    name: str = "AWAC"
    env: str = "halfcheetah-medium-v2"
    discount: float = 0.99
    tau: float = 0.005
    awac_lambda: float = 1.0
    max_timesteps: int = int(1e6)
    buffer_size: int = int(2e6)
    batch_size: int = 256
    normalize: bool = True
    normalize_reward: bool = True
    vf_lr: float = 3e-4
    qf_lr: float = 3e-4
    actor_lr: float = 3e-4
    actor_dropout: Optional[float] = 0.0  
    eval_freq: int = int(5e3)
    n_episodes: int = 10
    checkpoints_path: str = "../checkpoints"
    load_model: str = "" # file name for loading a model
    seed: int = 21
    device: str = "cuda"
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
    env = gym.make(config.env)
    is_env_with_goal = config.env.startswith(ENVS_WITH_GOAL)
    
    # prepare dataset
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    dataset = d4rl.qlearning_dataset(env)
    
    if config.normalize_reward:
        modify_reward(dataset, config.env)
        
    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1
        
    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean, state_std)
    replay_buffer = ReplayBuffer(
        state_dim, action_dim, config.buffer_size, config.device
    )
        
    replay_buffer.load_d4rl_dataset(dataset)
    
    max_action = float(env.action_space.high[0])
    
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)
            
    # Set seeds
    seed = config.seed
    set_seed(seed, env, deterministic_torch=True)
    
    q_network = DoubleQNetwork(state_dim, action_dim).to(config.device)
    actor = GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        ).to(config.device)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    
    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # SQL
        "awac_lambda": config.awac_lambda,
    }
    
    print("-------------------------")
    print(f"Traing IQL, Env: {config.env}, Seed: {seed}")
    print("-------------------------")

    # Initialize actor
    trainer = AdvantageWeightedActorCritic(**kwargs)
    
    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor
    logger = Logger(config.checkpoints_path, asdict(config), use_wandb=config.use_wandb)
    
    
    eval_successes = []
    
    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        logger.log_dict(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t+1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores, success_rate = eval_actor(env, actor, config.device, config.n_episodes, config.seed) 
            eval_score = eval_scores.mean()
            eval_log = {}
            normalized = env.get_normalized_score(eval_score)
            # Valid only for envs with goal, e.g. AntMaze, Adroit
            if is_env_with_goal:
                eval_successes.append(success_rate)
                eval_log["eval/regret"] = np.mean(1 - np.array(eval_successes))
                eval_log["eval/is_success"] = success_rate.mean()     
            normalized_eval_score = normalized * 100
            evaluations.append(normalized_eval_score)
            eval_log["eval/d4rl_normalized_score"] = normalized_eval_score
            print("--------------------------")
            print(f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}")
            print("--------------------------")
            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoints_{t}.pt"),
                )
            logger.log_dict(eval_log, trainer.total_it)
        
if __name__ == "__main__":
    train()