import torch
from aurl.self_supervised.byol import BYOL
from aurl.utils.normalization import get_norm_stats
from aurl.models.resnets import *
import logging
import os
import hydra
from aurl.utils.logger import Logger
from aurl.datasets.byol_dataset import *
from tqdm import tqdm
from aurl.optimizers.lars import *
from aurl.utils.utils import save_pkl_file
from torchvision import transforms as TV

log = logging.getLogger(__name__)


def train_byol(cfg, logger):

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")    
    
    dataloader = load_byol_data(cfg) 
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    
    logger.log('Saving configs.')
    save_pkl_file(os.path.join(cfg.checkpoint_dir, 'data_cfg.pkl'), cfg)
    
    logger.log("Loaded data.")

    # Compute mean and std dev for dataset
    logger.log("Calculating norm stats")

    audio_norm_stats, _ = get_norm_stats(dataloader, action_norm=False)
    logger.log("Norm stats : {}".format(audio_norm_stats))
    
    audio_data_normalizer = nn.Sequential(
        TV.Normalize(mean = audio_norm_stats.mean, std = audio_norm_stats.std)
    )
    
    logger.log('\n Saving norm stats')
    save_pkl_file(os.path.join(cfg.checkpoint_dir, 'norm_stats.pkl'), audio_norm_stats)

    encoder = hydra.utils.instantiate(cfg.model, 
            num_classes = cfg.latent_dim,
            num_mics =cfg.dataset.num_mics).to(device)

    learner = BYOL(
        encoder,
        n_mics = cfg.dataset.num_mics,
        h = dataloader.dataset[0][0].shape[1],
        w = dataloader.dataset[0][0].shape[2], # Data shaped as (n_mics x n_mels x t)
        hidden_layer = 'avgpool',
        apply_augs = cfg.aurl.apply_augs # Determined by aurl variant
    ).to(device)

    if cfg.optimizer == 'adam':
        opt = torch.optim.Adam(learner.parameters(), lr=cfg.lr)
    
    elif cfg.optimizer == 'LARS':
        opt = LARS(
            learner.parameters(),
            lr = 0,
            weight_decay = cfg.weight_decay,
            weight_decay_filter = exclude_bias_and_norm,
            lars_adaptation_filter = exclude_bias_and_norm,
        )

    for epoch in tqdm(range(cfg.epochs+1), desc="BYOL epochs"):
        epoch_loss = 0.0
        for i, (data, data_prime) in tqdm(enumerate(dataloader)):
            
            if cfg.optimizer == 'LARS':
                lr = adjust_learning_rate(cfg, opt, dataloader, epoch)
                logger.log({'lr': lr, 'epoch': epoch, 'idx': i})
                        
            #Apply norm tfs
            data, data_prime = audio_data_normalizer(data).to(device), audio_data_normalizer(data_prime).to(device)

            x = data
            y = data if cfg.aurl.same_img else data_prime

            loss = learner(x, y) # Apply augs is passed into byol learner.

            epoch_loss += loss.item()        
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average() # update moving average of target encoder

        logger.log({'epoch_loss': epoch_loss/len(dataloader), 'epoch': epoch})
    
    # save your improved network
        if epoch%cfg.eval_frequency ==0:
            logger.log("Saving: {}".format(os.path.join(cfg.checkpoint_dir, 'improved-net-{}.pt'.format(epoch))))
            torch.save(encoder, os.path.join(cfg.checkpoint_dir, 'improved-net-{}.pt'.format(epoch)))

@hydra.main(version_base='1.2',config_path='configs', config_name = 'pretrain_byol')
def main(cfg):
    logger = Logger(log_wandb=cfg.log_wandb, simple_log = log, cfg=cfg)
    train_byol(cfg, logger)

if __name__ == '__main__':
    main()