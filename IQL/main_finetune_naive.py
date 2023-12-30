import os  
os.environ["CULAB_WORKSPACE_CONFIG"] = ":4096:8"
import random
from dataclasses import asdict, dataclass 
from pathlib import Path 
from typing import Optional

import d4rl 
import gym
import torch   
import pyrallis  
import datetime  

from modules import GaussianPolicy, DeterministicPolicy, DoubleQNetwork, ValueNetwork
from utils import *
from replay_buffer import ReplayBuffer
from logger import Logger
from iql import ImplicitQLearning

@dataclass
class TrainConfig:
    device: str = "cuda"
    env: str = "antmaze-umaze-v2"
    seed: int = 21 
    eval_seed: int = 21
    eval_freq: int = int(5e4)
    n_episodes: int = 100
    offline_iterations: int = int(1e6)
    online_iteractions: int = int(1e6)
    checkpoints_path: str = "../checkpoints"
    load_model: str = ""
    # IQL 
    actor_dropout: float = 0.0
    buffer_size: int = int(2e6)
    batch_size: int = 256
    discount: float = 0.99
    tau: float = 0.005
    beta: float = 3.0
    iql_tau: float = 0.7
    expl_noise: float = 0.03 # Std of Gaussian exploration noise
    noise_clip: float = 0.5
    iql_deterministic: bool = False
    normalize: bool = True # Normalize states
    normalize_reward: bool = False # Normalize rewards
    vf_lr: float = 3e-4
    qf_lr: float = 3e-4
    actor_lr: float = 3e-4
    # Logging
    project: str = "off_offon"
    group: str = ""
    name: str = "IQL-finetune"
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
    eval_env = gym.make(config.env)
    
    is_env_with_goal = config.env.startswith(ENVS_WITH_GOAL)
    
    max_steps = env._max_episode_steps
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    dataset = d4rl.qlearning_dataset(env)
    
    reward_mod_dict = {}
    if config.normalize_reward:
        reward_mod_dict = modify_reward(dataset, config.env)
    
    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], 1e-3)
    else:
        state_mean, state_std = 0, 1
        
    # dataset["observations"] = normalize_states(dataset["observations"], state_mean, state_std)
    # dataset["next_observations"] = normalize_states(dataset["next_observations"], state_mean, state_std)
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(state_dim, action_dim, config.buffer_size, device=config.device)
    replay_buffer.load_d4rl_dataset(dataset)
    if config.normalize:
        replay_buffer.normalize_states()
    max_action = float(env.action_space.high[0])
    
    if config.checkpoints_path is not None:
        print(f"Chechpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.json"), "w") as f:
            pyrallis.dump(config, f)
    
    # Set seeds
    seed = config.seed
    set_seed(seed)
    set_env_seed(eval_env, seed)
    set_env_seed(env, seed)
    q_network = DoubleQNetwork(state_dim, action_dim).to(config.device)
    v_network = ValueNetwork(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(state_dim, action_dim, max_action, dropout=config.actor_dropout)
        if config.iql_deterministic
        else GaussianPolicy(state_dim, action_dim, max_action, dropout=config.actor_dropout)
    ).to(config.device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    
    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.offline_iterations,
    }
    
    print(f"----------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {seed}")
    print(f"----------------------------")
    
    # Initialize IQL
    trainer = ImplicitQLearning(**kwargs)
    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    logger = Logger(config.checkpoints_path, asdict(config), use_wandb=config.use_wandb)
    
    evaluations = []
    
    state, done = env.reset(), False
    episode_return = 0
    episode_step = 0
    goal_achieved = False
    
    eval_successes = []
    train_successes = []
    
    print("Offline pretraining")
    for t in range(int(config.offline_iterations) + int(config.online_iteractions)):
        if t == config.offline_iterations:
            print("Online finetuning")
        online_log = {}
        if t >= config.offline_iterations:
            episode_step += 1
            action = actor(torch.tensor(state.reshape(1, -1), dtype=torch.float32, device=config.device))
            if not config.iql_deterministic:
                action = action.sample()
            else:
                noise = (torch.randn_like(action) * config.expl_noise).clamp(-config.noise_clip, config.noise_clip)
                action += noise
            action = torch.clamp(max_action * action, -max_action, max_action)
            action = action.cpu().data.numpy().flatten()
            next_state, reward, done, env_infos = env.step(action)
            
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)

            real_done = False # Episode can timeout which is different from done
            if done and episode_step < max_steps:
                real_done = True
            if config.normalize_reward:
                reward = modify_reward_online(reward, config.env, **reward_mod_dict)
                
            replay_buffer.add_transition(state, action, reward, next_state, real_done)
            state = next_state
            if done:
                state, done = env.reset(), False
                # Valid only for envs with goal, e.g. AntMaze, Adroit
                if is_env_with_goal:
                    train_successes.append(goal_achieved)
                    online_log["train/regret"] = np.mean(1 - np.array(train_successes))
                    online_log["train/is_success"] = float(goal_achieved)
                online_log["train/episode_return"] = episode_return
                normalize_return = eval_env.get_normalized_score(episode_return)
                online_log["train/d4rl_normalized_return"] = (normalize_return * 100 )
                online_log["train/episode_length"] = episode_step
                episode_return = 0
                episode_step = 0
                goal_achieved = False
            
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        log_dict["offline_iter" if t <  config.offline_iterations else "online_iter"] = (
            t if t < config.offline_iterations else t - config.offline_iterations
        )             
        log_dict.update(online_log)
        logger.log_dict(log_dict, trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores, success_rate = eval_actor(eval_env, actor, config.device, config.n_episodes, config.eval_seed) 
            eval_score = eval_scores.mean()
            eval_log = {}
            normalized = eval_env.get_normalized_score(eval_score)
            # Valid only for envs with goal, e.g. AntMaze, Adroit
            if t >= config.offline_iterations and is_env_with_goal:
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