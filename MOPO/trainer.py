import time
import os 
import numpy as np 
import torch 
import gym

from typing import Optional, Dict, List, Tuple
from tqdm import tqdm 
from collections import deque
from buffer import ReplayBuffer
from logger import Logger
from mopo import BasePolicy

class Trainer:
    def __init__(self, policy: BasePolicy, eval_env: gym.Env, real_buffer: ReplayBuffer, fake_buffer: ReplayBuffer, logger: Logger, rollout_setting: Tuple[int, int, int], epoch: int=1000, step_per_epoch: int=1000, batch_size=256, real_ratio: float=0.05, eval_episodes: int=10, lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]=None, dynamics_update_freq: int=0) -> None:
        self.policy = policy
        self.eval_env = eval_env
        self.real_buffer = real_buffer
        self.fake_buffer = fake_buffer
        self.logger = logger
        self._rollout_freq, self._rollout_batch_size, self._rollout_length = rollout_setting
        self._dynamics_update_freq = dynamics_update_freq   
        
        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._real_ratio = real_ratio
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        num_timesteps = 0
        last_10_performance = deque(maxlen=10)
        for e in range(1, self._epoch + 1):
            self.policy.train()
            pbar = tqdm(range(self._step_per_epoch), desc=f"Epoch #{e}/{self.epoch}")
            for it in pbar:
                if num_timesteps % self._rollout_freq == 0:
                    init_obss = self.real_buffer.sample(self._rollout_batch_size)["observations"].cpu().numpy()
                    rollout_transitions, rollout_info = self.policy.rollout(init_obss, self._rollout_length)
                    self.fake_buffer.add_batch(**rollout_transitions)
                    self.logger.log_dict({"num_rollout_transitions": rollout_info["num_transitions"], "reward_mean": rollout_info["reward_mean"]}, num_timesteps)

                real_sample_size = int(self._batch_size * self._real_ratio)
                fake_sample_size = self._batch_size - real_sample_size 
                real_batch = self.real_buffer.sample(real_sample_size)
                fake_batch = self.fake_buffer.sample(fake_sample_size)  

                batch = {"real": real_batch, "fake": fake_batch}
                loss = self.policy.learn(batch)
                pbar.set_postfix(**loss)
                self.logger.log_dict(loss, num_timesteps)

                # update dynamics model if necessary
                if 0 < self._dynamics_update_freq and (num_timesteps + 1) % self._dynamics_update_freq == 0:
                    dynamics_update_info = self.policy.update_dynamics_model(self.real_buffer)
                    self.logger.log_dict(dynamics_update_info, num_timesteps)
                
                num_timesteps += 1
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # evaluate policy
            eval_info = self._evaluate()
            ep_reward_mean, ep_reward_std = np.mean(eval_info["eval/episode_reward"]), np.std(eval_info["eval/episode_reward"])
            ep_length_mean, ep_length_std = np.mean(eval_info["eval/episode_length"]), np.std(eval_info["eval/episode_length"])
            norm_ep_reward_mean = self.eval_env.get_normalized_reward(ep_reward_mean) * 100
            norm_ep_reward_std = self.eval_env.get_normalized_reward(ep_reward_std) * 100
            last_10_performance.append(norm_ep_reward_mean)
            self.logger.add_scalar("eval/normalized_episode_reward", norm_ep_reward_mean, num_timesteps)
            self.logger.add_scalar("eval/normalized_episode_reward_std", norm_ep_reward_std, num_timesteps)
            self.logger.add_scalar("eval/episode_length", ep_length_mean, num_timesteps)
            self.logger.add_scalar("eval/episode_length_std", ep_length_std, num_timesteps)

        self.logger.add_scalar("total_time", time.time() - start_time, num_timesteps)  
        torch.save(self.policy.state_dict(), os.path.join(self.logger.model_dir, "policy.pth"))
        self.policy.dynamics.save(self.logger.model_dir)

        return {"last_10_performance": np.mean(last_10_performance)}
    

    def _evaluate(self) -> Dict[str, List[float]]:
        self.policy.eval()
        obs = self.eval_env.reset()
        eval_ep_info_buffer = []
        num_episodes = 0 
        episode_reward, episode_length = 0, 0

        while num_episodes <  self._eval_episodes:
            action = self.policy.select_action(obs.reshape(-1, 1), deterministic=True)
            next_obs, reward, terminal, _ = self.eval_env.step(action.flatten())
            episode_reward += reward
            episode_length += 1

            obs = next_obs 

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                num_episodes += 1
                episode_reward, episode_length = 0, 0
                obs = self.eval_env.reset()
            
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer], 
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }

