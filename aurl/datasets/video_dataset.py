import torch

import operator
from aurl.utils.utils import save_pkl_file
from aurl.datasets.dataset import *

import torchvision
import numpy as np


class VideoDataset(AuRLDataset):
    """Used for vision based baseline"""
    def __init__(self, root_dir, cfg = None, transform = None, return_fname = False) -> None:
        
        super().__init__(root_dir = root_dir, 
            cfg = cfg, 
            transform = transform, 
            return_fname = return_fname)
    
    def __len__(self):
        return len(self.dirs)
    
    def __getitem__(self, index):
        data = self._load_video_data(index)  
        label = self._load_params(self.dirs[index]).float()

        if self.return_fname:
            return data, label, self.dirs[index]
        
        return data, label
    
    def _load_video_data(self, index, n_frames = 5):
        """ Equally spaced 5 frames chosen from the video clip."""
        
        if 'imgs' in os.listdir(self.dirs[index]): #For vertical probing; we have raw images and not video
            L = len(os.listdir(os.path.join(self.dirs[index],'imgs')))
            step = (L - 1)//(n_frames)
            idxs = np.arange(0, L-1, step = step)
            idxs = idxs[:5]
        
            frames = []
            for i in idxs:
                frames.append(torchvision.io.read_image(os.path.join(self.dirs[index],'imgs', '{}.jpg'.format(i))))
            return torch.stack(frames)
        
        video_path = os.path.join(self.dirs[index], 'video.mp4')
        frames = torchvision.io.read_video(video_path, output_format='TCHW')[0]
        step = (len(frames) - 1)//(n_frames)
        idxs = np.arange(0, len(frames)-1, step = step)
        idxs = idxs[:5]
        
        return frames[idxs]
    

def load_video_data(cfg):
    
    dataset = load_dataset(cfg)
    train_set, val_set = split_dataset(cfg, dataset)
    
    if cfg.save_test_fnames: # Save for robot evals
        save_pkl_file(os.path.join(cfg.checkpoint_dir, 'test_fnames.pkl'), operator.itemgetter(*val_set.indices)(val_set.dataset.dirs))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, 
                                               num_workers=cfg.num_workers, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, 
                                             num_workers=cfg.num_workers, pin_memory=False) 
    return train_loader, val_loader

def load_dataset(cfg, return_fname=False):
    dataset = VideoDataset(cfg.dataset.data_dir,  cfg=cfg.dataset, return_fname=return_fname)
    return dataset
    