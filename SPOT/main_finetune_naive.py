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

from modules import Actor, VAE, DoubleQNetwork
from utils import *
from replay_buffer import ReplayBuffer
from logger import Logger
from spot import SPOT
from tqdm import tqdm

@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "halfcheetah-medium-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_seed: int = 0  # Eval environment seed
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    offline_iterations: int = int(1e6)  # Number of offline updates
    online_iterations: int = int(1e6)  # Number of online updates
    checkpoints_path: Optional[str] =  "../checkpoints"   # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # TD3
    actor_lr: float = 1e-4  # Actor learning ratev
    critic_lr: float = 3e-4  # Actor learning rate
    buffer_size: int = 20_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    expl_noise: float = 0.1  # Std of Gaussian exploration noise
    tau: float = 0.005  # Target network update rate
    policy_noise: float = 0.2  # Noise added to target actor during critic update
    noise_clip: float = 0.5  # Range to clip target actor noise
    policy_freq: int = 2  # Frequency of delayed actor updates
    # SPOT VAE
    vae_lr: float = 1e-3  # VAE learning rate
    vae_hidden_dim: int = 750  # VAE hidden layers dimension
    vae_latent_dim: Optional[int] = None  # VAE latent space, 2 * action_dim if None
    beta: float = 0.5  # KL loss weight
    vae_iterations: int = 100_000  # Number of VAE training updates
    # SPOT
    actor_init_w: Optional[float] = None  # Actor head init parameter
    critic_init_w: Optional[float] = None  # Critic head init parameter
    lambd: float = 1.0  # Support constraint weight
    num_samples: int = 1  # Number of samples for density estimation
    iwae: bool = False  # Use IWAE loss
    lambd_cool: bool = False  # Cooling lambda during fine-tune
    lambd_end: float = 0.2  # Minimal value of lambda
    normalize: bool = False  # Normalize states
    normalize_reward: bool = True  # Normalize reward
    online_discount: float = 0.995  # Discount for online tuning

    project: str = "off_offon_finetune_debug_2"
    group: str = ""
    name: str = "spot"
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
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
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
    set_seed(seed, env)
    set_env_seed(eval_env, config.eval_seed)

    vae = VAE(
        state_dim, action_dim, config.vae_latent_dim, max_action, config.vae_hidden_dim
    ).to(config.device)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=config.vae_lr)

    actor = Actor(state_dim, action_dim, max_action, config.actor_init_w).to(
        config.device
    )
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    q_network = DoubleQNetwork(state_dim, action_dim, config.critic_init_w).to(config.device)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.critic_lr)

    kwargs = {
        "max_action": max_action,
        "vae": vae,
        "vae_optimizer": vae_optimizer,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "double_q_network": q_network,
        "double_q_network_optimizer": q_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # TD3
        "policy_noise": config.policy_noise * max_action,
        "noise_clip": config.noise_clip * max_action,
        "policy_freq": config.policy_freq,
        # SPOT
        "beta": config.beta,
        "lambd": config.lambd,
        "num_samples": config.num_samples,
        "iwae": config.iwae,
        "lambd_cool": config.lambd_cool,
        "lambd_end": config.lambd_end,
        "max_online_steps": config.online_iterations,
    }

    print("---------------------------------------")
    print(f"Training SPOT, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = SPOT(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    logger = Logger(config.checkpoints_path, asdict(config), use_wandb=config.use_wandb)
    evaluations = []

    print("Training VAE")
    for t in tqdm(range(int(config.vae_iterations))):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.vae_train(batch)
        log_dict["vae_iter"] = t
        logger.log_dict(log_dict, step=trainer.total_it)

    vae.eval()
    state, done = env.reset(), False
    episode_return = 0
    episode_step = 0
    goal_achieved = False

    eval_successes = []
    train_successes = []

    print("Offline pretraining")
    for t in range(int(config.offline_iterations) + int(config.online_iterations)):
        if t == config.offline_iterations:
            print("Online tuning")
            trainer.is_online = True
            trainer.discount = config.online_discount
            # Resetting optimizers
            trainer.actor_optimizer = torch.optim.Adam(
                actor.parameters(), lr=config.actor_lr
            )
            trainer.qf_optimizer = torch.optim.Adam(
                q_network.parameters(), lr=config.critic_lr
            )
        online_log = {}
        if t >= config.offline_iterations:
            episode_step += 1
            action = actor(
                torch.tensor(
                    state.reshape(1, -1), device=config.device, dtype=torch.float32
                )
            )
            noise = (torch.randn_like(action) * config.expl_noise).clamp(
                -config.noise_clip, config.noise_clip
            )
            action += noise
            action = torch.clamp(max_action * action, -max_action, max_action)
            action = action.cpu().data.numpy().flatten()
            next_state, reward, done, env_infos = env.step(action)

            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
            episode_return += reward
            real_done = False  # Episode can timeout which is different from done
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
                normalized_return = eval_env.get_normalized_score(episode_return)
                online_log["train/d4rl_normalized_episode_return"] = (
                    normalized_return * 100.0
                )
                online_log["train/episode_length"] = episode_step
                episode_return = 0
                episode_step = 0
                goal_achieved = False

        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        log_dict["offline_iter" if t < config.offline_iterations else "online_iter"] = (
            t if t < config.offline_iterations else t - config.offline_iterations
        )
        log_dict.update(online_log)
        logger.log_dict(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores, success_rate = eval_actor(
                eval_env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            eval_log = {}
            normalized = eval_env.get_normalized_score(np.mean(eval_scores))
            # Valid only for envs with goal, e.g. AntMaze, Adroit
            if t >= config.offline_iterations and is_env_with_goal:
                eval_successes.append(success_rate)
                eval_log["eval/regret"] = np.mean(1 - np.array(train_successes))
                eval_log["eval/success_rate"] = success_rate
            normalized_eval_score = normalized * 100.0
            eval_log["eval/d4rl_normalized_score"] = normalized_eval_score
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")
            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            logger.log_dict(eval_log, step=trainer.total_it)


if __name__ == "__main__":
    train()


