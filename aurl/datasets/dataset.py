import torch
import torchaudio
from torch.utils.data import Dataset

import random
import os
from collections import defaultdict
import torch.nn.functional as F
import torchaudio.transforms as T

from aurl.datasets.transforms import *
from tqdm import tqdm

TRANSFORMS = {
    'wav' : get_signal,
    'mel': to_mel_spectrogram
}

class AuRLDataset(Dataset):
    """Superclass for all datasets used in AuRL
    """
    def __init__(self, root_dir, cfg = None, transform = None, return_fname = False) -> None:        
        
        self.cfg = cfg

        # Lazy loading; sorted for repeatability
        self.dirs = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir)])
        
        
        self.create_action_audio_dict()
        
        if transform is not None: # Only for audio datasets 
            self.resampler = T.Resample(orig_freq = self.cfg.sample_rate, new_freq = self.cfg.resample_rate)
            self.transform = TRANSFORMS[transform]

        self.return_fname = return_fname        
        super().__init__()

    def __len__(self):
        return len(self.dirs)
    
    def __getitem__(self, index):
        raise NotImplementedError


    def _load_audio_data(self, index):
        wav, sample_rate = torchaudio.load(os.path.join(self.dirs[index], 'audio-clip.wav'))
        
        #Load only valid channels; Clip longer sequences
        wav = wav[:self.cfg.num_mics, :self.cfg.max_num_frames].float()
            
        #Pad the shorter signal to make sequences equal length [wav.shape -> n_mics x signal_length]
        wav = F.pad(wav, pad=(0, self.cfg.max_num_frames - wav.shape[1]), mode='constant', value=0.)
        
        # Resample audio to reduce size    
        wav = self.resampler(wav)
        
        # convert data into wav/ mel/ spec
        data = self.transform(self.cfg, wav.float())
        
        return data

    def _load_params(self, folder):
        return torch.load(os.path.join(folder, 'params.pt')).float()
        
    def create_action_audio_dict(self):
    # Precompute. Pay all the preprocessing costs when you load
        self.action_map = defaultdict(set) 
        self.act_idx_map = defaultdict(list) 
        for i, folder in enumerate(tqdm(self.dirs, desc = "Action map")):
            #Tuple to make hashable. Numpy for precision to make it uniquely mapable.
            self.action_map[tuple(self._load_params(folder).numpy())].add(i)
        
        for k, v in  self.action_map.items():
            for i in v:
                self.act_idx_map[i] = list(v.difference(set([i])))
        
    
def split_dataset(cfg, dataset):

    random.seed(42)
    act_map_keys = set(dataset.action_map.keys())
    
    train_set_size, test_set_size,  = cfg.num_train_pts, cfg.num_test_pts

    test_acts = random.sample(act_map_keys, test_set_size) #Test size is always fixed -> Sample this first.
    train_acts = random.sample(act_map_keys.difference(test_acts), train_set_size)
    
    train_idxs, test_idxs = set(), set()

    for k in train_acts:
        train_idxs = train_idxs.union(dataset.action_map[k])
    
    for k in test_acts:
        test_idxs = test_idxs.union(dataset.action_map[k])
    
    assert train_idxs.isdisjoint(test_idxs), "Train and test idxs are not disjoint"

    train_set = torch.utils.data.dataset.Subset(dataset, list(train_idxs))
    test_set = torch.utils.data.dataset.Subset(dataset, list(test_idxs))
    
    return train_set, test_set