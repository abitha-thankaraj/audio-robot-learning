import torch

import operator
from aurl.utils.utils import save_pkl_file

from aurl.datasets.dataset import *

class AudioDataset(AuRLDataset):
    """Used to finetune AuRL and train all audio based baselines.
    """
    def __init__(self, root_dir, cfg = None, transform = None, return_fname = False) -> None:    
        super().__init__(root_dir = root_dir, 
                cfg = cfg, 
                transform = transform, 
                return_fname = return_fname)

    
    def __getitem__(self, index):
        data = self._load_audio_data(index)
        label = self._load_params(self.dirs[index]).float()

        if self.return_fname:
            return data, label, self.dirs[index]

        return data, label
    

def load_data(cfg):
    
    dataset = load_dataset(cfg)

    train_set, val_set = split_dataset(cfg, dataset)
    if cfg.save_test_fnames: # Save for robot evals
        save_pkl_file(os.path.join(cfg.checkpoint_dir, 'test_fnames.pkl'), operator.itemgetter(*val_set.indices)(val_set.dataset.dirs))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, 
                                               num_workers=cfg.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, 
                                             num_workers=cfg.num_workers, pin_memory=True) 
    return train_loader, val_loader

def load_dataset(cfg, return_fname=False):
    dataset = AudioDataset(cfg.dataset.data_dir, transform = cfg.dataset.name, cfg=cfg.dataset, return_fname=return_fname)
    return dataset
