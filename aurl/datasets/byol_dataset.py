
import torch
from aurl.datasets.dataset import *

class BYOLAudioDataset(AuRLDataset):
    """Used in pretraining encoder.
    """
    def __init__(self, root_dir, cfg = None, transform = None, return_fname = False) -> None:             
        super().__init__(root_dir = root_dir, 
                cfg = cfg, 
                transform = transform, 
                return_fname = return_fname)
    
    def __getitem__(self, index):
        data = self._load_audio_data(index)
        index_prime = self.sample_same_act_pair(index)
        data_prime = self._load_audio_data(index_prime)
        return data, data_prime
    
    def sample_same_act_pair(self, idx):
        return random.choice(self.act_idx_map[idx])

def load_byol_data(cfg):
    
    dataset = load_dataset(cfg)
    if cfg.num_train_pts is None: # Experiment uses enntire dataset
        train_set = dataset
    else: # Ablations for ssl in low data settings
        train_set, _ = split_dataset(cfg, dataset)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, 
                                               num_workers=cfg.num_workers, pin_memory=False) 
    return train_loader

def load_dataset(cfg):
    dataset = BYOLAudioDataset(cfg.dataset.data_dir, transform = cfg.dataset.name, cfg=cfg.dataset)
    return dataset
