import hydra
from aurl.utils.logger import Logger
from aurl.utils.utils import save_pkl_file
from aurl.datasets.audio_dataset import load_data
from aurl.datasets.video_dataset import load_video_data
import torch
import logging
import os
from tqdm import tqdm
from easydict import EasyDict
from aurl.utils.normalization import get_norm_stats
from torchvision import transforms as T
import torch.nn as nn

from aurl.models.audio_regressor import AudioRegressor
from aurl.models.mlp import MLP

log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg, logger) -> None:

        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        
        # Baselines
        if cfg.encoder_dir is None: 

            if cfg.model.encoder_name == 'data2vec': # Pretrained audio model
                encoder = hydra.utils.instantiate(cfg.model,
                    num_mics=cfg.dataset.num_mics,
                    latent_dim = cfg.latent_dim, device=self.device).to(self.device)
            
            elif cfg.video: # Pretrained vision model
                encoder = hydra.utils.instantiate(
                    cfg.model,
                    latent_dim = cfg.latent_dim, device=self.device).to(self.device)
            
            
            else: # Simple supervised training
                encoder = hydra.utils.instantiate(cfg.model, 
                    num_classes = cfg.latent_dim,
                    num_mics =cfg.dataset.num_mics).to(self.device)

        # AuRL : Load BYOL pretrained encoder
        else:
            logger.log("Loading encoder from : {}".format(cfg.encoder_dir))
            encoder = torch.load(cfg.encoder_dir, map_location = self.device)


        regression_head = MLP(cfg.latent_dim , cfg.dataset.action_dim, cfg.hidden_dims).to(self.device) # Linear probe
        self.model = AudioRegressor(encoder, regression_head, freeze_encoder = cfg.freeze_encoder).to(self.device)
        self.optim = hydra.utils.instantiate(cfg.optimizer, params = self.model.parameters())

        # Note: Loss function has reduction = sum configured (To make logging easier)
        
        self.criterion = torch.nn.MSELoss(reduction='sum').to(self.device)
        
        self.cfg = cfg


    def train_one_epoch(self, dataloader):
        epoch_train_loss = 0.
        for _, (data, label) in enumerate(tqdm(dataloader)):
            data, label = self._apply_data_transforms(data, label)

            self.model.train()
            self.optim.zero_grad()
            train_loss = self.criterion(self.model(data), label.float())
            train_loss.backward()
            self.optim.step()

            epoch_train_loss += train_loss.item() #Assumption : reduction == sum
        
        return epoch_train_loss / len(dataloader.dataset)

    def evaluate_one_epoch(self, dataloader):
        epoch_val_loss = 0.
        for _, (data, label) in enumerate(tqdm(dataloader)):
            data, label = self._apply_data_transforms(data, label)
            self.model.eval()
            with torch.no_grad():
                val_loss = self.criterion(self.model(data), label.float())
                epoch_val_loss += val_loss.item()
        
        return epoch_val_loss / len(dataloader.dataset)

    def _set_norm_stats(self, train_loader, norm_audio_data, norm_action):        
        self.audio_norm_stats, self.action_min_max = get_norm_stats(train_loader, norm_audio_data, norm_action)
        if norm_audio_data:
            self.audio_data_normalizer = nn.Sequential(
                T.Normalize(mean = self.audio_norm_stats.mean, std = self.audio_norm_stats.std)
            )
        if norm_action:
            self.action_normalizer = lambda actions : (actions - self.action_min_max.min) / (self.action_min_max.max - self.action_min_max.min)
            self.action_denormalizer  = lambda normalized_actions : (normalized_actions * (self.action_min_max.max - self.action_min_max.min)) + self.action_min_max.min

    def _apply_data_transforms(self, data, label):
        
        if self.cfg.norm_audio_data:
            data = self.audio_data_normalizer(data)

        if self.cfg.norm_action:
            label = self.action_normalizer(label)
        
        if self.cfg.apply_augmentation: #Used for supervised with augmentations baseline
        # For mel spectrograms -> data shape = batch_size x n_mics x n_mels x t            
            h, w = data.shape[2], data.shape[3]
            transforms = torch.nn.Sequential(
                T.RandomResizedCrop((h//4, w//8))
            )
            data = transforms(data)        
        return data.to(self.device), label.to(self.device)

def train(cfg, logger):

    if cfg.video: # Video model -> Loads images
        train_loader, val_loader = load_video_data(cfg)     
    else: # Other models -> Load audio [mel spectrogram or raw signal]
        train_loader, val_loader = load_data(cfg)
    
    logger.log("Loaded data")

    best_loss = torch.inf
    trainer = Trainer(cfg, logger)
    
    logger.log("Set normalization stats")
    trainer._set_norm_stats(train_loader, cfg.norm_audio_data, cfg.norm_action)
    
    logger.log("Saving normalization stats")

    # Save configs - to run model on robot
    if cfg.norm_audio_data:
        save_pkl_file(os.path.join(cfg.checkpoint_dir, 'audio_norm_stats.pkl'), trainer.audio_norm_stats)
    if cfg.norm_action:
        save_pkl_file(os.path.join(cfg.checkpoint_dir, 'action_min_max_norm_stats.pkl'), trainer.action_min_max)
    
    # Save all cfgs
    save_pkl_file(os.path.join(cfg.checkpoint_dir, 'run_cfgs.pkl'), EasyDict(cfg))

    for i in tqdm(range(cfg.train_epochs)):

        epoch_train_loss = trainer.train_one_epoch(dataloader = train_loader)
            
        if i%cfg.log_frequency == 0:
            logger.log({'epoch':i, 'Train loss':epoch_train_loss})

        if i%cfg.eval_frequency == 0:
            logger.log("Validation")
            epoch_val_loss = trainer.evaluate_one_epoch(dataloader = val_loader)
            logger.log({'epoch':i, 
                    'Validation loss':epoch_val_loss})
                    
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                logger.log("Best loss :{}".format(best_loss))
                torch.save(trainer.model, os.path.join(cfg.checkpoint_dir, 'model.pth'))
                logger.log("Saved model : {}/model.pth".format(cfg.checkpoint_dir))
            
            logger.log({'Best loss':best_loss, 'epoch':i})
    
    return best_loss


@hydra.main(version_base='1.2',config_path='configs', config_name = 'train')

def main(cfg):
    logger = Logger(log_wandb=cfg.log_wandb, simple_log = log, cfg=cfg)
    train(cfg, logger)

if __name__ == '__main__':
    main()