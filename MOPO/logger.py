from torch.utils.tensorboard import SummaryWriter
import torch 
import wandb
import uuid 
import os 
import yaml
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union
from tokenize import Number
import argparse
import datetime
ROOT_DIR = "log"
import json

def wandb_init(config):
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
        dir=config["checkpoints_path"],
    )
    wandb.run.save()

class Logger:
    def __init__(self, logdir, config, use_wandb=True):
        self.logdir = logdir
        self.config = config
        self.use_wandb = use_wandb
        self.writer = SummaryWriter(logdir)
        if self.use_wandb:
            wandb_init(self.config)


    def log_dict(self, log_dict, step):
        if self.use_wandb:
            wandb.log(log_dict, step=step)
        for key, value in log_dict.items():
            self.writer.add_scalar(key, value, global_step=step)
        
    def add_scalar(self, key, value, step):
        self.writer.add_scalar(key, value, global_step=step)
        wandb.log({key: value}, step=step)

    def write_to_file(self, data):
        with open(os.path.join(self.logdir, "log.txt"), "a") as f:
            f.write(data + "\n")

    def log_hyperparams(self):
        log_path = os.path.join(self.logdir, "hyperparams.yaml")
        with open(log_path, "w") as f:
            yaml.dump(self.config, f)   


def make_log_dirs(
    task_name: str,
    algo_name: str,
    seed: int,
    args: Dict,
    record_params: Optional[List]=None
) -> str:
    if record_params is not None:
        for param_name in record_params:
            algo_name += f"&{param_name}={args[param_name]}"
    timestamp = datetime.datetime.now().strftime("%y-%m%d-%H%M%S")
    exp_name = f"seed_{seed}&timestamp_{timestamp}"
    log_dirs = os.path.join(ROOT_DIR, task_name, algo_name, exp_name)
    os.makedirs(log_dirs)
    return log_dirs


def load_args(load_path: str) -> argparse.ArgumentParser:
    args_dict = {}
    with open(load_path,'r') as f:
        args_dict.update(json.load(f))
    return argparse.Namespace(**args_dict)
