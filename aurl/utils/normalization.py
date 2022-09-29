from aurl.datasets.audio_dataset import *
import torch
from easydict import EasyDict
from tqdm import tqdm


class StatsRecorder:
    def __init__(self, red_dims=(0,2,3)):
        """Accumulates normalization statistics across mini-batches.
        ref: http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        """
        self.red_dims = red_dims # which mini-batch dimensions to average over
        self.nobservations = 0   # running number of observations

    def update(self, data):
        """
        data: ndarray, shape (nobservations, ndimensions)
        """
        # initialize stats and dimensions on first batch
        if self.nobservations == 0:
            self.mean = data.mean(dim=self.red_dims, keepdim=True)
            self.std  = data.std (dim=self.red_dims,keepdim=True)
            self.nobservations = data.shape[0]
            self.ndimensions   = data.shape[1]

        else:
            if data.shape[1] != self.ndimensions:
                raise ValueError('Data dims do not match previous observations.')
            
            # find mean of new mini batch
            newmean = data.mean(dim=self.red_dims, keepdim=True)
            newstd  = data.std(dim=self.red_dims, keepdim=True)
            
            # update number of observations
            m = self.nobservations * 1.0
            n = data.shape[0]

            # update running statistics
            tmp = self.mean
            self.mean = m/(m+n)*tmp + n/(m+n)*newmean
            self.std  = m/(m+n)*self.std**2 + n/(m+n)*newstd**2 +\
                        m*n/(m+n)**2 * (tmp - newmean)**2
            self.std  = torch.sqrt(self.std)
                                 
            # update total number of seen samples
            self.nobservations += n

class MinMaxRecorder:
    def __init__(self, reduce_dims = (0)) -> None:
        self.nobservations = 0
        self.reduce_dims = reduce_dims
    
    def update(self, data):
        if self.nobservations == 0:
            # Hardcoded for now
            self.max = -1. * torch.inf * torch.ones((1, data.shape[1]))
            self.min = torch.inf * torch.ones((1, data.shape[1]))

        self.min = torch.min(torch.concat([self.min, data]), dim = self.reduce_dims).values.unsqueeze(0)
        self.max = torch.max(torch.concat([self.max, data]), dim = self.reduce_dims).values.unsqueeze(0)

        n = data.shape[0]
        self.nobservations += n

        return self.min.squeeze(0), self.max.squeeze(0)


def get_norm_stats(dataloader,audio_norm=True,  action_norm=True):
    audio_norm_recorder = StatsRecorder() # Normalizes for mu, sigma in spectrograms.
    action_min_max_norm_recorder = MinMaxRecorder()
    
    for _, (data, label) in enumerate(tqdm(dataloader, desc = "Normalization stats")):
        # data - batch x n_mics x n_mels x t  or batch x n_mics x h x w
        if audio_norm:
            audio_norm_recorder.update(data)
        if action_norm:
            action_min_max_norm_recorder.update(label)

    audio_normalizer, action_normalizer = None, None
    
    if audio_norm:
        audio_normalizer =  EasyDict(dict(mean = audio_norm_recorder.mean[0,:,0,0], std = audio_norm_recorder.std[0,:,0,0]))
    if action_norm:
        action_normalizer=  EasyDict(dict(min = action_min_max_norm_recorder.min, max = action_min_max_norm_recorder.max))

    return audio_normalizer, action_normalizer

    