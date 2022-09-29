import torch
from torch import nn
from transformers import Data2VecAudioForXVector

class Data2VecWrapper(nn.Module):
    def __init__(self, num_mics, latent_dim, device, **kwargs) -> None:
        super(Data2VecWrapper, self).__init__()
        self.data2vec_model = Data2VecAudioForXVector.from_pretrained("hf-internal-testing/tiny-random-data2vec-xvector").to(device)
        self.linear_layer = torch.nn.Linear(512 * num_mics, latent_dim).to(device) 
        self.latent_dim = latent_dim
        self.num_mics = num_mics
        # Freeze encoder
        for param in self.data2vec_model.parameters():
            param.requires_grad = False
    
    
    def forward(self, x):
        # get representation for each channel/ mic
        embeddings = []
        for i in range(self.num_mics):
            with torch.no_grad():
                embeddings.append(self.data2vec_model(x[:, i, :]).embeddings)
        t_embeddings = torch.concat(embeddings, dim=-1)
        # default output in pretrained model has 512 dim embedding
        return self.linear_layer(t_embeddings)
