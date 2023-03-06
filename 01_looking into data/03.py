# converting smiles to molucule form from data from pyG
import rdkit
from torch_geometric.datasets import MoleculeNet
 
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole

data = MoleculeNet(root=".", name="ESOL")

molecule = Chem.MolFromSmiles(data[0]["smiles"])
print(molecule)

print(type(molecule))
