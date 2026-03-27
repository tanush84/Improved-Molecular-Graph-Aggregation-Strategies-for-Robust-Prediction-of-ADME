import pandas as pd
import numpy as np
from rdkit import Chem
import pandas as pd
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt
import pandas as pd
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from mordred import Calculator, descriptors as mordred_descriptors
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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

def calc_fcfp6(mol, n_bits=2048):
    return np.array(
        AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=3, nBits=n_bits, useFeatures=True
        ),
        dtype=np.float32
    )

def calc_maccs(mol):
    return np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)

def calc_rdkit_desc(mol):
    return np.array([fn(mol) for _, fn in Descriptors.descList], dtype=np.float32)

# Mordred calculator (heavy but accurate)
mordred_calc = Calculator(mordred_descriptors, ignore_3D=True)

def calc_mordred(mol):
    vals = mordred_calc(mol)
    return np.array(
        [float(v) if v is not None else np.nan for v in vals],
        dtype=np.float32
    )
def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }
