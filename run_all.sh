#!/bin/bash
set -e
python scripts/01_fetch_and_prepare.py
python scripts/02_harmonize.py
python scripts/03_build_graphs.py
python scripts/04_train_evaluate.py
# python scripts/05_gnn_train_evaluate.py  # GCN training; requires torch and torch_geometric (not in requirements.txt) -- run manually if needed
python scripts/06_figures.py
echo "Pipeline complete. Figures saved to figures/"
