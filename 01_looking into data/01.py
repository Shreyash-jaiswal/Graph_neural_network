import rdkit
from torch_geometric.datasets import MoleculeNet    # importing data set from torch_geometric
 

# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.MoleculeNet.html#torch_geometric.datasets.MoleculeNet
# contains information about diffrent dataset avaiable and parameters

# root (str) – Root directory where the dataset should be saved.

# name (str) – The name of the dataset ("ESOL", "FreeSolv", "Lipo", "PCBA", "MUV", "HIV", "BACE", "BBPB", "Tox21", "ToxCast", "SIDER", "ClinTox").


# Load the ESOL dataset
data = MoleculeNet(root=".", name="ESOL")

# ESOL
# ESOL is a small dataset consisting of water solubility data for 1128 compounds. 13 The dataset
# has been used to train models that estimate solubility directly from chemical structures (as
# encoded in SMILES strings). 22 Note that these structures don’t include 3D coordinates, since
# solubility is a property of a molecule and not of its particular conformers.

print(data)