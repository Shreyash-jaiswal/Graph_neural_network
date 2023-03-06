from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

# #### What is the Cora Dataset?
# The Cora dataset consists of 2708 scientific publications classified into one of seven classes. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words.

# - Nodes = Publications (Papers, Books ...)
# - Edges = Citations
# - Node Features = word vectors
# - 7 Labels = Pubilcation type e.g. Neural_Networks, Rule_Learning, Reinforcement_Learning, 	Probabilistic_Methods...

# We normalize the features using torch geometric's transform functions.

###################################################################
#investigating the dataset

# Get some basic info about the dataset
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print(50*'=')

# There is only one graph in the dataset, use it as new data object
data = dataset[0]  

# Gather some statistics about the graph.
print(data)
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Is undirected: {data.is_undirected()}')

#printing each data point 
print(data.x.shape) # [No. Nodes x Features]

# Print some of the normalized word counts of the first datapoint
print(data.x[0][:50])

#looking at the label
print(data.y)

#example of binary mask 
print(len(data.test_mask) == data.num_nodes)
print(data.test_mask)

#example for edge connection
print(data.edge_index.t())