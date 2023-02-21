#!/bin/bash

env_name="robustness_ml_mp18"
conda create -n $env_name -y
conda activate $env_name
conda install -y -c conda-forge numpy pandas
conda install -y -c conda-forge scikit-learn umap-learn
conda install -y -c conda-forge py-xgboost-gpu # xgboost with gpu support
conda install -y -c conda-forge pymatgen
conda install -y -c conda-forge matminer 
conda install -y -c conda-forge jarvis-tools
pip install linear-tree

# --- Install kernel to switch between env in spyder
# conda install -y spyder-kernels=2.3 ipywidgets # installed with spyder-kernels=2.2 if any issue




