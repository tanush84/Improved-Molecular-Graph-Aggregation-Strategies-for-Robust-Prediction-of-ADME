import pandas as pd
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from chemprop.data.datapoints import MoleculeDatapoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def log10_transform_safe(x, eps=1e-5):
    """
    Safe log10 transform:
    - clips values <=0
    - applies log10
    """
    return np.log10(np.clip(x, eps, None))

# Define a function to compute the scaffold
def compute_scaffold(smiles: str):
    mol = MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.GetScaffoldForMol(mol)
    
def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None
def smiles_to_mol(smiles):
    return Chem.MolFromSmiles(smiles)

def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }
    
# Define the function to create MoleculeDatapoints
def create_molecule_datapoints(smiles, targets):
    return [
        MoleculeDatapoint.from_smi(smi, y=target)
        for smi, target in zip(smiles, targets)
    ]
