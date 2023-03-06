import rdkit
from torch_geometric.datasets import MoleculeNet
 
# Load the ESOL dataset
data = MoleculeNet(root=".", name="ESOL")

# print("Dataset type: ", type(data))
# print("Dataset features: ", data.num_features)
# print("Dataset target: ", data.num_classes)
# print("Dataset length: ", data.len)
# print("Dataset sample: ", data[0])
# print("Sample  nodes: ", data[0].num_nodes)
# print("Sample  edges: ", data[0].num_edges)

# output :-
# Dataset type:  <class 'torch_geometric.datasets.molecule_net.MoleculeNet'>
# Dataset features:  9
# Dataset target:  734
# Dataset length:  <bound method InMemoryDataset.len of ESOL(1128)>
# Dataset sample:  Data(x=[32, 9], edge_index=[2, 68], edge_attr=[68, 3], smiles='OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O ', y=[1, 1])'
# x => node features y=> target value
# Sample  nodes:  32    // 32 nodes for molecule present at data[0]
# Sample  edges:  68    // 68 edges for molecule present at data[0]


# print(data[0].x)  # printing node features

# # output 32 instances with 9 features each
# tensor([[8, 0, 2, 5, 1, 0, 4, 0, 0],
#         [6, 0, 4, 5, 2, 0, 4, 0, 0],



# edge infomation is present in Coordinate List (COO) Format
# print(data[0].edge_index.t())
# [ 0,  1], node 0 is connected to node 1
# [ 1,  0], node 1 is connected to node 0

print(data[0].y) #target value at the label -0.7700 which is water solubility

# print(data[0]["smiles"])
# edge_index = graph connections
# smiles = molecule with its atoms
# x = node features (32 nodes have each 9 features)
# y = labels (dimension)