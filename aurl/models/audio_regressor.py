import torch.nn as nn


class AudioRegressor(nn.Module):
    def __init__(self, encoder = None, regression_head = None,  return_latents = False, freeze_encoder=False) -> None:
        super().__init__()
        
        self.return_latents = return_latents
        self.encoder = encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.regression_head = regression_head

        modules = [self.encoder, self.regression_head]        
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        if self.return_latents: # Returns representation from encoder
            return self.model[:1](x)

        return self.model(x)