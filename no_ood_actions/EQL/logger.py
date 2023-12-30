from torch.utils.tensorboard import SummaryWriter
import torch
import wandb
import uuid


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
        self.writter = SummaryWriter(logdir)
        if self.use_wandb:
            wandb_init(self.config)
            
    def log_dict(self, log_dict, step):
        if self.use_wandb:
            wandb.log(log_dict, step=step)
        for key, value in log_dict.items():
            self.writter.add_scalar(key, value, global_step=step)