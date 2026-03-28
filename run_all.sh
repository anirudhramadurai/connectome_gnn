#!/bin/bash
set -e
python scripts/01_fetch_and_prepare.py
python scripts/02_build_graphs.py
python scripts/03_train_evaluate.py
python scripts/04_figures.py
echo "Pipeline complete. Figures saved to figures/"
