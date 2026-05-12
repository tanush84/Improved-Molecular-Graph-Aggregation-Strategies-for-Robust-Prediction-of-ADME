Improved Molecular Graph Aggregation Strategies for Robust Prediction of ADME and Pharmacokinetic Properties

This repository contains the implementation, benchmarking framework, and reproducible training workflows associated with the manuscript:

**“Improved Molecular Graph Aggregation Strategies for Robust Prediction of ADME and Pharmacokinetic Properties”**

The study systematically evaluates descriptor-based machine learning models and graph neural network (GNN)-based molecular representation learning strategies for predicting multiple ADME and pharmacokinetic endpoints. Particular emphasis is placed on investigating the impact of adaptive aggregation mechanisms within the Directed Message Passing Neural Network (D-MPNN) framework.

To enhance representation learning, we systematically investigate adaptive aggregation mechanisms, including attentive, multi-head attention, and gated attentive pooling, which enable context-aware weighting of atom-level features. Model performance was evaluated across six pharmacokinetic endpoints namely plasma protein binding (LogFup), CaCO2 permeability (CaCO2Papp), lipophilicity (LogD), clearance (LogCL), and ionization constants (pKa and pKb) using standardized external validation metrics (MAE, RMSE, and R²). Compared to descriptor-based baselines, GNN-based models demonstrated consistent improvements in predictive accuracy and error reduction across most endpoints. Adaptive aggregation strategies provided enhanced flexibility and improved performance for structurally complex and chemically heterogeneous endpoints.

The following endpoints were benchmarked:

| Endpoint  | Description                |
| --------- | -------------------------- |
| LogFup    | Plasma Protein Binding     |
| CaCO2Papp | Caco-2 Permeability        |
| LogD      | Lipophilicity              |
| LogCL     | Intrinsic Clearance        |
| pKa       | Acid Dissociation Constant |
| pKb       | Base Dissociation Constant |


Molecular Representation Strategies
Descriptor-Based Features

The descriptor-based benchmarking pipeline includes:

RDKit descriptors
Mordred descriptors
MACCS fingerprints
Circular fingerprints (FCFP/ECFP)
Graph Neural Network Representations

Graph-based models were implemented using the Chemprop D-MPNN framework with modified aggregation strategies:

Mean Aggregation
Attentive Aggregation
Multi-Head Attention Aggregation
Gated Attentive Aggregation

Installation
Create Conda Environment for descriptor based models and chemprop based model using descriptorenv.yaml and chempropenv.yaml files.
conda env create -f NameOfEnvironment.yml.                                
Training notebooks are provided for: Base descriptor based model training as well as Chemprop based Model training for each 6 parameters. 
Due to proprietary software integration and internal deployment restrictions associated with in-house ADMET modeling workflows, pretrained production-level model checkpoints and certain advanced deployment utilities cannot currently be redistributed publicly.

However:

Complete model training workflows,
Aggregation mechanism implementations,
Evaluation pipelines,
Environment configuration files,
and benchmarking notebooks

are fully provided to ensure reproducibility of the reported experiments.

Part of the code currently used for advanced in-house ADMET modeling can be supplied upon reasonable request. Researchers interested in advanced aggregation implementations or extended deployment functionality are encouraged to contact the corresponding authors.
