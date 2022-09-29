import torch
from torch import nn
import torchvision
from torchvision.models import ResNet18_Weights

class ResNet18VideoModel(nn.Module):
    def __init__(self, latent_dim, device, n_frames=5, **kwargs) -> None:
        super(ResNet18VideoModel, self).__init__()
        self.encoder = torchvision.models.resnet18(pretrained=True).to(device)
        self.latent_dim = latent_dim
        self.n_frames = n_frames
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.linear_layer = torch.nn.Linear(1000 * n_frames, latent_dim).to(device) 

        self.weights = ResNet18_Weights.DEFAULT
        self.preprocess = self.weights.transforms()
    
    
    def forward(self, x, n_frames = 5):
        #!! x.shape ->  batch_size x n_frames x h x w x c 
        embeddings = []
        for i in range(n_frames):
            with torch.no_grad():
                embeddings.append(self.encoder(self.preprocess(x[:, i, :, :, :])))        
        
        t_embeddings = torch.concat(embeddings, dim=-1)

        return self.linear_layer(t_embeddings)

