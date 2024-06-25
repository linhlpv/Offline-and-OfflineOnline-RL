import numpy as np
import os.path as path
import torch


class StandardScaler(object):
    def __init__(self, mu=None, std=None):
        self.mu = mu
        self.std = std

    def fit(self, data):

        self.mu = np.mean(data, axis=-0, keepdims=True)
        self.std = np.std(data, axis=-0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """"
        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.std
    
    def inverse_transform(self, data):
        """
        Arguments:
        data (np.array): A numpy array containing the points to be transformed.
        
        Returns: (np.array) The transformed dataset.
        """
        return self.std * data + self.mu
    
    def save_scaler(self, save_path):
        mu_path = path.join(save_path, "mu.npy")
        std_path = path.join(save_path, "std.npy")
        np.save(mu_path, self.mu)
        np.save(std_path, self.std)
    
    def load_scaler(self, load_path):
        mu_path = path.join(load_path, "mu.npy")
        std_path = path.join(load_path, "std.npy")
        self.mu = np.load(mu_path)
        self.std = np.load(std_path)

    def transform_tensor(self, data: torch.Tensor):
        device = data.device
        data = self.transform(data.cpu().numpy())
        data = torch.tensor(data, device=device)
        return data
    

###### Terminal functions

def obs_unnormalization(termination_fn, obs_mean, obs_std):
    def thunk(obs, act, next_obs):
        obs = obs * obs_std + obs_mean
        next_obs = next_obs * obs_std + obs_mean
        return termination_fn(obs, act, next_obs)
    
    return thunk

def termination_fn_halfcheetah(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    not_done = np.logical_and(np.all(next_obs > -100, axis=-1), np.all(next_obs < 100, axis=-1))
    done = ~not_done
    done = done[:, None]
    return done


def termination_fn_hopper(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done =  np.isfinite(next_obs).all(axis=-1) \
                    * np.abs(next_obs[:,1:] < 100).all(axis=-1) \
                    * (height > .7) \
                    * (np.abs(angle) < .2)

    done = ~not_done
    done = done[:,None]
    return done

def termination_fn_halfcheetahveljump(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False]).repeat(len(obs))
    done = done[:,None]
    return done

def termination_fn_antangle(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    x = next_obs[:, 0]
    not_done = 	np.isfinite(next_obs).all(axis=-1) \
                * (x >= 0.2) \
                * (x <= 1.0)

    done = ~not_done
    done = done[:,None]
    return done

def termination_fn_ant(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    x = next_obs[:, 0]
    not_done = 	np.isfinite(next_obs).all(axis=-1) \
                * (x >= 0.2) \
                * (x <= 1.0)

    done = ~not_done
    done = done[:,None]
    return done

def termination_fn_walker2d(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done =  np.logical_and(np.all(next_obs > -100, axis=-1), np.all(next_obs < 100, axis=-1)) \
                * (height > 0.8) \
                * (height < 2.0) \
                * (angle > -1.0) \
                * (angle < 1.0)
    done = ~not_done
    done = done[:,None]
    return done

def termination_fn_point2denv(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False]).repeat(len(obs))
    done = done[:,None]
    return done

def termination_fn_point2dwallenv(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False]).repeat(len(obs))
    done = done[:,None]
    return done

def termination_fn_pendulum(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.zeros((len(obs), 1))
    return done

def termination_fn_humanoid(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    z = next_obs[:,0]
    done = (z < 1.0) + (z > 2.0)

    done = done[:,None]
    return done

def termination_fn_pen(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    obj_pos = next_obs[:, 24:27]
    done = obj_pos[:, 2] < 0.075

    done = done[:,None]
    return done

def terminaltion_fn_door(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False] * obs.shape[0])

    done = done[:, None]
    return done

def get_termination_fn(task):
    if 'halfcheetahvel' in task:
        return termination_fn_halfcheetahveljump
    elif 'halfcheetah' in task:
        return termination_fn_halfcheetah
    elif 'hopper' in task:
        return termination_fn_hopper
    elif 'antangle' in task:
        return termination_fn_antangle
    elif 'ant' in task:
        return termination_fn_ant
    elif 'walker2d' in task:
        return termination_fn_walker2d
    elif 'point2denv' in task:
        return termination_fn_point2denv
    elif 'point2dwallenv' in task:
        return termination_fn_point2dwallenv
    elif 'pendulum' in task:
        return termination_fn_pendulum
    elif 'humanoid' in task:
        return termination_fn_humanoid
    elif 'pen' in task:
        return termination_fn_pen
    elif 'door' in task:
        return terminaltion_fn_door
    else:
        raise np.zeros



##### Offline Dataset
def qlearning_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for using by standard Q-learning algorithms, 
    with observations, actions, next_observations, rewards, and a terminal flag.

    Arguments:
        env (gym.Env): An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None, the dataset
            will be default to env.get_dataset().
        terminate_on_end (bool): Set done=True on the last timestep in a trajectory. 
            Default is False, and will discard the last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().
    
    Returns: 
        A dictionary containing the following keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_act array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of terminals.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)
    
    has_next_obs = True if "next_observations" in dataset else False

    N = dataset['rewards'].shape[0]
    obss = []
    actions = []
    next_obss = []
    rewards = []
    dones = []

    # The newer version of the dataset adds an explicit timeouts field.
    # Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True
    
    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        if has_next_obs:
            new_obs = dataset['next_observations'][i].astype(np.float32)
        else:
            new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # skip this tranition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0
            if not has_next_obs:
                continue

        obss.append(obs)
        next_obss.append(new_obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(done_bool)
        episode_step += 1

    return {
        "observations": np.array(obss),
        "actions": np.array(actions),
        "next_observations": np.array(next_obss),
        "rewards": np.array(rewards),
        "terminals": np.array(dones)
    }

