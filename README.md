# Offline and Offline to Online RL algorithms' implementations

This repository includes the implementation (in [Pytorch](https://pytorch.org/docs/stable/torch.html)) for the common offline RL baselines and Offline-Online RL algorithms too. This repo is inspired by the [CORL](https://github.com/corl-team/CORL). Highly recommend checking the CORL repo too.

The repo contains many folders that include the implementation code for each algorithm. To run the code, please follow the instructions below.

# Install
First, create the conda environment with ```python==3.9.16```
```
conda create -n off_offon python=3.9.16
```
Then, please download and install ```mujoco=2.1```, and setup follow [this](https://github.com/openai/mujoco-py) intruction.
Finally, install all the dependencies 
```
conda activate off_offon
pip install -r requirements.txt
```

# Training
For training, please go to the algorithm folder.
```
cd <./algorithm_folder/>
```
Then, for training in the offline mode, please run the command below.
```
python main_offline.py
```

To run the online finetuning, please follow the command below.
```
python main_finetune_naive.py
```

# Directory Hierarchy
The project is organized by algorithm, with the root directory containing global configuration and setup files.
Below is the directory hierarchy of the repo.

```
Offline-and-OfflineOnline-RL/
├── AWAC/                                   # Algorithm: Advantage Weighted Actor Critic
│   ├── awac.py                             # Pytorch implementation of the algorithm
│   ├── modules.py                          # PyTorch modules defining the policy (Actor) and value functions (Critic)
│   ├── main_<offline/finetune_naive>.py    # Main implementation of the algorithm
│   └── replay_buffer.py                    # Implementation of the data storage and sampling strategies
│   └── logger.py                           # Implementation of the Logger class
│   └── utils.py                            # Implementation os the utilization functions.
│
├── IQL/                                    # Algorithm: Implicit Q-Learning
│   ├── iql.py                              # Main implementation of the algorithm
│   └── ...
```

# Algorithm Implementation Structure
The primary purpose of this repository is to provide a well-structured, clear, and easily modifiable code structure, allowing for the addition of algorithm components and testing of new research ideas. 

To ensure readability and ease of debugging, each algorithm folder (e.g., IQL/) typically contains a self-contained implementation. This means that the network architecture, RL algorithm logic, and training loop are grouped to allow the algorithm to run as a standalone script.

# How to add a new algorithm
To add a new algorithm, please follow the instructions below.
- Create a folder: create a new folder with the name of your algorithm.
- Implement the scripts: create a main Python file that contains the logic of the algorithms (e.g., your_algo.py)
    - Copy the structure from an existing robust implementation.
    - Replace the specific loss functions and update rules. 
- Dependencies: If the new algorithm requires unique libraries, try to keep them local or document them clearly. Standard dependencies should be put in the root requirements.txt file.

# List of the algorithms
Model-free algorithms
- [AWAC]().
- [IQL]().
- [SQL]().
- [EQL]().
- [SAC-N/EDAC]().
- [SPOT]().
- [INAC]()

Model-based algorithms
- [MOPO]()

# References

- [AWAC: Accelerating Online Reinforcement Learning with Offline Datasets](https://arxiv.org/abs/2006.09359)
    - Ashvin Nair, Abhishek Gupta, Murtaza Dalal, Sergey Levine. 2020.
- [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169)
    - Ilya Kostrikov, Ashvin Nair, Sergey Levine. 2021.
- [The In-Sample Softmax for Offline Reinforcement Learning](https://arxiv.org/abs/2302.14372)
    - Chenjun Xiao, Han Wang, Yangchen Pan, Adam White, Martha White. 2023.
- [Offline RL with No OOD Actions: In-Sample Learning via Implicit Value Regularization](https://arxiv.org/abs/2303.15810)
    - Haoran Xu, Li Jiang, Jianxiong Li, Zhuoran Yang, Zhaoran Wang, Victor Wai Kin Chan, Xianyuan Zhan. 2023.
- [MOPO: Model-based Offline Policy Optimization](https://arxiv.org/abs/2005.13239)
    - Tianhe Yu, Garrett Thomas, Lantao Yu, Stefano Ermon, James Zou, Sergey Levine, Chelsea Finn, Tengyu Ma.
-  [Supported Policy Optimization for Offline Reinforcement Learning](https://arxiv.org/abs/2202.06239)
    - Jialong Wu, Haixu Wu, Zihan Qiu, Jianmin Wang, Mingsheng Long
-   [Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble](https://arxiv.org/abs/2110.01548)
    - Gaon An, Seungyong Moon, Jang-Hyun Kim, Hyun Oh Song
