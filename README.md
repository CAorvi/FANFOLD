# FANFOLD: Graph Normalization Flow-Induced Asymmetric Network for Unsupervised Graph-Level Anomaly Detection
![Framework](fig1.png)

## Overview
FANFOLD consists of four stages:
1. **Preparation**  
   Node attributes are combined with structural encodings based on random walks and node degrees.
2. **Teacher Pre-training**  
   A teacher GNN is pre-trained on normal graphs using attribute and structure reconstruction.
3. **Density-aware Teacher Modeling**  
   A normalizing-flow module is applied exclusively to the teacher branch to model the latent distribution of normal graph representations.
4. **Asymmetric Distillation and Anomaly Scoring**  
   A student GNN learns to mimic the flow-enhanced teacher representations on normal graphs. During inference, the teacher-student discrepancy is used as the anomaly score.
   
## Requirements
* Python==3.8
* Pytorch==2.2.1+cu121
* Pytorch Geometric==2.5.2
* Numpy==1.24.3
* Scikit-learn==1.3.2
* OGB==1.3.6
* NetworkX==3.1

## Datasets
We evaluate FANFOLD on eleven benchmark datasets:
- **Small molecules:** AIDS, DHFR, BZR, COX2, HSE, MMP, PPAR-gamma
- **Bioinformatics:** PROTEINS_full, DD
- **Social networks:** IMDB-BINARY, REDDIT-BINARY
