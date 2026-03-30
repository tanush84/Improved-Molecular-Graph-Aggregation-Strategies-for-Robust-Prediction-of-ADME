import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from chemprop.data.datapoints import MoleculeDatapoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from chemprop.data import datapoints, dataloader, MoleculeDataset
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.data.datapoints import MoleculeDatapoint
from chemprop import data, featurizers, models, nn
import torch
import chemprop.nn.metrics as chem_metrics
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from chemprop.nn.agg import (
    MultiHeadAttentiveAggregation,
    GatedAttentiveAggregation,
    AttentiveAggregationv1,
)
from assets.functionchem import *
def run_chemprop_mp_agg_benchmark(
    train_df,
    val_df,
    test_df,
    target_column,
    smiles_column="smiles",
    max_epochs=200,
    num_workers=12,
    checkpoint_dir="./checkpoint/checkpoints_variants/",
    seed=42
    ):
    # ======================================================
    # Reproducibility
    # ======================================================
    pl.seed_everything(seed, workers=True)

    # ======================================================
    # Prepare data
    # ======================================================
    train_smis = train_df[smiles_column].values
    val_smis   = val_df[smiles_column].values
    test_smis  = test_df[smiles_column].values

    train_targets = train_df[[target_column]].values
    val_targets   = val_df[[target_column]].values
    test_targets  = test_df[[target_column]].values

    train_dp = create_molecule_datapoints(train_smis, train_targets)
    val_dp   = create_molecule_datapoints(val_smis, val_targets)
    test_dp  = create_molecule_datapoints(test_smis, test_targets)

    train_dset = MoleculeDataset(train_dp)
    val_dset   = MoleculeDataset(val_dp)
    test_dset  = MoleculeDataset(test_dp)

    scaler = train_dset.normalize_targets()
    val_dset.normalize_targets(scaler)

    train_loader = data.build_dataloader(train_dset, num_workers=num_workers, shuffle=True)
    val_loader   = data.build_dataloader(val_dset,   num_workers=num_workers, shuffle=False)
    test_loader  = data.build_dataloader(test_dset,  num_workers=num_workers, shuffle=False)

    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

    # ======================================================
    # Message Passing Variants
    # ======================================================
    mp_variants = {
        "BondMP3": lambda: nn.BondMessagePassing(depth=3, dropout=0.1),
        "BondMP4": lambda: nn.BondMessagePassing(depth=4, dropout=0.1),
        "BondMP5": lambda: nn.BondMessagePassing(depth=5, dropout=0.1),
        "BondMP6": lambda: nn.BondMessagePassing(depth=6, dropout=0.1),
        "AtomMP3": lambda: nn.AtomMessagePassing(depth=3, dropout=0.1),
        "AtomMP4": lambda: nn.AtomMessagePassing(depth=4, dropout=0.1),
        "AtomMP5": lambda: nn.AtomMessagePassing(depth=5, dropout=0.1),
        "AtomMP6": lambda: nn.AtomMessagePassing(depth=6, dropout=0.1),
    }

    # ======================================================
    # Aggregation Variants
    # ======================================================
    agg_variants = {
        "Mean": lambda: nn.MeanAggregation(),
        "GatedAttentive": lambda: GatedAttentiveAggregation(output_size=300),
        "Attentive1": lambda: AttentiveAggregationv1(output_size=300),
        "MultiHead4": lambda: MultiHeadAttentiveAggregation(output_size=300, num_heads=4),
        "MultiHead8": lambda: MultiHeadAttentiveAggregation(output_size=300, num_heads=8),
        "MultiHead12": lambda: MultiHeadAttentiveAggregation(output_size=300, num_heads=12),
    }

    # ======================================================
    # Run combinations
    # ======================================================
    all_results = []

    BASE_CKPT_DIR = Path(checkpoint_dir) / target_column
    BASE_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    for mp_name, mp_fn in mp_variants.items():
        for agg_name, agg_fn in agg_variants.items():

            print(f"\nTraining: {mp_name} + {agg_name}")

            mp = mp_fn()
            agg = agg_fn()
            ffn = nn.RegressionFFN(output_transform=output_transform)

            mpnn = models.MPNN(
                mp,
                agg,
                ffn,
                batch_norm=True,
                metrics=[
                    chem_metrics.MAE(),
                    chem_metrics.RMSE(),
                    chem_metrics.R2Score()
                ]
            )

            combo_ckpt_dir = BASE_CKPT_DIR / f"{mp_name}_{agg_name}"
            combo_ckpt_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_cb = ModelCheckpoint(
                dirpath=combo_ckpt_dir,
                filename="best-{epoch}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_last=True
            )

            trainer = pl.Trainer(
                accelerator="auto",
                devices=1,
                max_epochs=max_epochs,
                logger=False,
                callbacks=[checkpoint_cb],
                enable_progress_bar=True
            )

            trainer.fit(mpnn, train_loader, val_loader)

            test_metrics = trainer.test(
                mpnn,
                dataloaders=test_loader,
                verbose=False
            )[0]

            result = {
                "Target": target_column,
                "MessagePassing": mp_name,
                "Aggregation": agg_name,
                "Test_MAE": test_metrics["test/mae"],
                "Test_RMSE": test_metrics["test/rmse"],
                "Test_R2": test_metrics["test/r2"],
            }

            all_results.append(result)
            print("Test:", result)

    results_df = pd.DataFrame(all_results).sort_values("Test_MAE")

    csv_path = f"chemprop_mpall_agg_comparison_{target_column}.csv"
    results_df.to_csv(csv_path, index=False)

    print("\nFinal comparison:")
    print(results_df)

    return results_df
