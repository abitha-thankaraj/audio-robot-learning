import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden_sizes, activation=nn.ReLU):
        super(MLP, self).__init__()
        if len(hidden_sizes) == 0:
            self.model = nn.Linear(d_in, d_out)
        else:
            modules = [nn.Linear(d_in, hidden_sizes[0])]
            for i in range(len(hidden_sizes) - 1):
                modules.append(activation())
                modules.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            modules.append(activation())    
            modules.append(nn.Linear(hidden_sizes[-1], d_out))

            self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)