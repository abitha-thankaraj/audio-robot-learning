import wandb
import logging
import sys
import os
import omegaconf
from hydra.core.hydra_config import HydraConfig

class Logger:
    def __init__(self, log_wandb = True, simple_log = None, log_dir = None, cfg = None) -> None:
        self.log_wandb = log_wandb
        self.simple_log = simple_log
        
        cfg.checkpoint_dir= HydraConfig.get().run.dir

        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        if log_dir is not None:
            self._configure_simple_log(log_dir)
        
        if log_wandb:
            wandb_exp_name = '-'.join(HydraConfig.get().run.dir.split('/')[-2:])
            wandb.init(project = cfg.wandb_project, 
                name=wandb_exp_name,
                config = omegaconf.OmegaConf.to_container(cfg, resolve=True), 
                settings=wandb.Settings(start_method="thread"))

    def _configure_simple_log(self, log_dir):
        self.simple_log.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(os.path.join(log_dir, 'logs.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        self.simple_log.addHandler(file_handler)
        self.simple_log.addHandler(stdout_handler)
    
    def log(self, msg):
        if self.log_wandb:
            if type(msg) is dict:
                wandb.log(msg)
            # else:
            #     raise Warning('The logger can currently only log dictionaries to wandb')
        self.simple_log.info(msg)