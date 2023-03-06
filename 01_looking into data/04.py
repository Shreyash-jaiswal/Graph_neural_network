#implementing GNN

#importing dataset with molecular representation
import rdkit
from torch_geometric.datasets import MoleculeNet
 
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
data = MoleculeNet(root=".", name="ESOL")

import torch
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
#GCNConv graph convolutional layer 
# message passing layer
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
embedding_size = 64     #size of feature set after message passing itteration

class GCN(torch.nn.Module):
    def __init__(self):
        # here we define our layers 
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)

        # GCN layers
        # these are 4 message passing layers
        self.initial_conv = GCNConv(data.num_features, embedding_size)
        # first is transformation layer this converts our 9 feature set to 64 embedding vector

        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)

        # Output layer
        self.out = Linear(embedding_size*2, 1)
        # this the our output layer with shape 1
        # gmp + gap therefore embedding_size*2

    def forward(self, x, edge_index, batch_index):
        # providing edge information with node features 
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden) # activation function

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden)
          
        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)
        
        # gmp global max pooling and gap global mean pooling

        # Apply a final (linear) classifier.
        out = self.out(hidden)

        return out, hidden

model = GCN()
print(model)
print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

