import torch
import numpy as np 
from typing import List, Tuple
import os

class ReplayBuffer:
    def __init__(
        self, 
        state_dim,
        action_dim,
        buffer_size=int(1e6),
        device="cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.device = device
        
        self.states = torch.zeros((buffer_size, state_dim),dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((buffer_size, action_dim),dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((buffer_size, 1),dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((buffer_size, state_dim),dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((buffer_size, 1),dtype=torch.float32, device=self.device)
        
        self.pointer = 0
        self.size = 0 # current size of the replay buffer
        
    def to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float32, device=self.device)
    
    def load_from_json(self, json_file):
        import json
        
        if not json_file.endswith(".json"):
            json_file += ".json"
            
        json_file = os.path.join("json_dataset", json_file)
        output = dict()
        with open(json_file, "r") as f:
            output = json.load(f)
            
        for k, v in output.items():
            v = np.array(v)
            if k != "terminals":
                v = v.astype(np.float32)
            
            output[k] = v
        
        self.from_d4rl(output) 
    
    
    def load_d4rl_dataset(self, dataset):
        if self.size != 0:
            raise ValueError("buffer is not empty")
        
        n_transitions = dataset["observations"].shape[0]
        if n_transitions > self.buffer_size:
            raise ValueError("dataset is larger than buffer")
        self.states[:n_transitions] = self.to_tensor(dataset["observations"])
        self.actions[:n_transitions] = self.to_tensor(dataset["actions"])
        self.rewards[:n_transitions] = self.to_tensor(dataset["rewards"][...,None])
        self.next_states[:n_transitions] = self.to_tensor(dataset["next_observations"])
        self.dones[:n_transitions] = self.to_tensor(dataset["terminals"][..., None])
        self.size += n_transitions
        self.pointer = min(self.size, n_transitions)
        
        print(f"Dataset size: {n_transitions}")
        
    def sample(self, batch_size):
        indices = np.random.randint(0, min(self.size, self.pointer), size=batch_size)
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]
        return [states, actions, rewards, next_states, dones]
    
    def get_statistic(self):
        state_mean, state_std = self.states.mean(dim=0), self.states.std(dim=0)
        action_mean, action_std = self.actions.mean(dim=0), self.actions.std(dim=0)
        return {
            "state_mean": state_mean,
            "state_std": state_std,
            "action_mean": action_mean, 
            "action_std": action_std
            }
        
    def add_transition(
        self, 
        state, 
        action, 
        reward,
        next_state, 
        done,
    ):
        self.states[self.pointer] = self.to_tensor(state)
        self.actions[self.pointer] = self.to_tensor(action)
        self.rewards[self.pointer] = self.to_tensor(reward)
        self.next_states[self.pointer] = self.to_tensor(next_state)
        self.dones[self.pointer] = self.to_tensor(done)
        self.pointer = (self.pointer + 1) % self.buffer_size
        self.size = min(self.size+1, self.buffer_size)
        
    def add_batch_transitions(
        self, 
        states, 
        actions,
        rewards,
        next_states,
        dones,
    ):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.add_transition(state, action, reward, next_state, done)
            
    
    

    
