import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    MLP Layer used after graph vector representation
"""

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        hidden_dims = [(input_dim//2**l , input_dim//2**(l+1)) for l in range(L)] #[(64, 32), (32, 16)]
        list_FC_layers = [ nn.Linear(hidden_dims[l][0] , hidden_dims[l][1], bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.BN_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dims[l][1]) for l in range(L)])
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = self.BN_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y