# Offline and Offline to Online RL algorithm's implementations

This repository includes the implementation (in [Pytorch](https://pytorch.org/docs/stable/torch.html)) for the common offline RL baselines and Offline-Online RL algorithms too. The code (and coding style) is mainly inspired by the [CORL](https://github.com/corl-team/CORL). Highly recommend to check CORL repo too.

The repo contains many folder that include the implementation code for each algorithms. To run the code, please follow the instruction below.

# Install
First, create the conda environment with ```python==3.9.16```
```
conda create -n off_offon python=3.9.16
```
Then, please download and install ```mujoco=2.1```, and setup follow [this](https://github.com/openai/mujoco-py) intruction.
Finally, install all the dependences 
```
conda activate off_offon
pip install -r requirements.txt
```

# List of the algorithms
- [AWAC]().
- [IQL]().
- [SQL]().
- [EQL]().


# References

- [AWAC: Accelerating Online Reinforcement Learning with Offline Datasets](https://arxiv.org/abs/2006.09359)
    - Ashvin Nair, Abhishek Gupta, Murtaza Dalal, Sergey Levine. 2020.
- [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169)
    - Ilya Kostrikov, Ashvin Nair, Sergey Levine. 2021.
- [Offline RL with No OOD Actions: In-Sample Learning via Implicit Value Regularization](https://arxiv.org/abs/2303.15810)
    - Haoran Xu, Li Jiang, Jianxiong Li, Zhuoran Yang, Zhaoran Wang, Victor Wai Kin Chan, Xianyuan Zhan. 2023.