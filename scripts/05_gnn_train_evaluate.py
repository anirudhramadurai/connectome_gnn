"""
05_gnn_train_evaluate.py
------------------------
Trains a two-layer Graph Convolutional Network (GCN) to classify ASD from
resting-state fMRI functional connectivity graphs using PyTorch Geometric.

This script is the full implementation of what models/gcn_numpy.py sketches
mathematically. It uses PyTorch's autograd for stable gradient computation
and PyTorch Geometric's GCNConv for efficient graph convolution.

Graph construction for GCN
---------------------------
This script builds its own sparse graph representation directly from the
harmonized connectomes, using a higher edge threshold (|z| > 0.50) than
the gradient boosting pipeline (|z| > 0.20). The rationale is important:

GCNConv's message passing aggregates each node's features with those of
its neighbors. With a 0.20 threshold, graphs have ~60% density (~24,000
edges per subject), meaning each node averages over ~120 neighbors in
every message passing step. After two layers, every node's representation
converges toward the global mean -- the local structure that GCNs are
designed to exploit is washed out.

At |z| > 0.50, graphs retain only the strongest ~15% of connections
(~4,000-6,000 edges), making the graph topology meaningful and allowing
the GCN to learn which local neighborhoods differ between ASD and controls.

Model architecture
------------------
  Input (200 nodes x 5 features, L2-normalized per node)
    --> GCNConv(5 -> 64) + BatchNorm + ReLU + Dropout
    --> GCNConv(64 -> 64) + BatchNorm + ReLU + Dropout
    --> Global mean pool (200 nodes -> 1 graph embedding of dim 64)
    --> Linear(64 -> 32) + ReLU + Dropout
    --> Linear(32 -> 1)
    --> BCEWithLogitsLoss with class-balanced weighting

Training details
----------------
  Optimizer    : Adam (lr=0.001, weight decay=1e-4)
  Scheduler    : ReduceLROnPlateau (patience=10, factor=0.5)
  Early stopping: patience=25 epochs on validation AUC
  Validation   : 20% of training set held out per fold
  Epochs       : up to 200 per fold
  Batch size   : 16 graphs
  Dropout      : 0.4

Note on device
--------------
PyTorch Geometric sparse operations have known numerical instabilities on
Apple MPS. This script runs on CPU for reliable gradient flow.

Usage
-----
  python scripts/05_gnn_train_evaluate.py

Inputs
------
  data/connectomes_harmonized.npy   from 02_harmonize.py
  data/labels.npy                   from 01_fetch_and_prepare.py
  data/roi_meta.pkl                 from 01_fetch_and_prepare.py

Outputs
-------
  results/gnn_cv_results.pkl   fold results, probabilities, node importance
  results/gnn_metrics.csv      summary table

References
----------
Kipf TN, Welling M. (2017). Semi-supervised classification with graph
  convolutional networks. ICLR 2017. arXiv:1609.02907.

Li X, et al. (2021). BrainGNN: Interpretable brain graph neural network
  for fMRI analysis. Medical Image Analysis, 74:102233.
  doi:10.1016/j.media.2021.102233.

Fey M, Lenssen JE. (2019). Fast graph representation learning with
  PyTorch Geometric. ICLR Workshop on Representation Learning on Graphs
  and Manifolds. arXiv:1903.02428.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# GCN-specific edge threshold -- higher than the GB pipeline (0.20)
# to keep graphs sparse enough for meaningful local message passing
GCN_THRESHOLD = 0.50

N_FOLDS      = 5
EPOCHS       = 200
LR           = 0.001
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.4
BATCH_SIZE   = 16
HIDDEN_DIM   = 64
PATIENCE     = 25
VAL_FRAC     = 0.20
RANDOM_SEED  = 42

DEVICE = torch.device("cpu")

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class BrainGCN(nn.Module):
    """
    Two-layer GCN with batch normalization and global mean pooling
    for graph-level binary classification.
    """

    def __init__(self, in_dim=5, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.conv1 = GCNConv(in_dim,     hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.lin1  = nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin2  = nn.Linear(hidden_dim // 2, 1)
        self.drop  = dropout

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop, training=self.training)

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.lin2(x)

        return x.squeeze(-1)


def build_sparse_graph(mat, threshold):
    """
    Build a sparse graph from a connectivity matrix using a higher threshold
    than the gradient boosting pipeline. Returns edge_index, edge_weight,
    and a 5-dimensional node feature vector per ROI.

    Node features (same as 03_build_graphs.py, recomputed at new threshold):
      0  mean_fc    -- average Fisher z to all other ROIs
      1  degree     -- number of edges above threshold
      2  clustering -- local clustering coefficient
      3  pos_fc     -- mean positive FC
      4  neg_fc     -- mean negative FC

    Parameters
    ----------
    mat       : (200, 200) harmonized Fisher z matrix
    threshold : float

    Returns
    -------
    edge_index  : (2, E) int array
    edge_weight : (E,) float array (absolute values)
    x           : (200, 5) float array of node features
    n_edges     : int
    """
    mask = np.abs(mat) > threshold
    np.fill_diagonal(mask, False)

    src, dst    = np.where(mask)
    edge_weight = np.abs(mat[src, dst])
    edge_index  = np.stack([src, dst], axis=0)

    adj = mask.astype(np.float32)
    np.fill_diagonal(adj, 0)

    mean_fc  = mat.mean(axis=1)
    degree   = adj.sum(axis=1)
    adj2     = adj @ adj
    triangles = (adj2 * adj).sum(axis=1)
    possible  = degree * (degree - 1)
    clust    = np.where(possible > 0, triangles / possible, 0.0)
    pos_fc   = np.where(mat > threshold,  mat, 0.0).mean(axis=1)
    neg_fc   = np.where(mat < -threshold, mat, 0.0).mean(axis=1)

    x = np.stack([mean_fc, degree, clust, pos_fc, neg_fc], axis=1).astype(np.float32)

    return edge_index.astype(np.int32), edge_weight.astype(np.float32), x, len(src)


def connectomes_to_pyg(connectomes, labels, threshold):
    """
    Build PyTorch Geometric Data objects from harmonized connectomes.

    Parameters
    ----------
    connectomes : (N, 200, 200) harmonized Fisher z matrices
    labels      : (N,) int array
    threshold   : float, edge inclusion threshold

    Returns
    -------
    data_list   : list of torch_geometric.data.Data
    edge_counts : list of int
    """
    data_list   = []
    edge_counts = []

    for mat, label in zip(connectomes, labels):
        ei, ew, x, n_edges = build_sparse_graph(mat, threshold)

        x_t  = torch.tensor(x,  dtype=torch.float)
        x_t  = F.normalize(x_t, p=2, dim=1)
        ei_t = torch.tensor(ei, dtype=torch.long)
        ew_t = torch.tensor(ew, dtype=torch.float)
        y    = torch.tensor(float(label), dtype=torch.float)

        data_list.append(Data(x=x_t, edge_index=ei_t, edge_attr=ew_t, y=y))
        edge_counts.append(n_edges)

    return data_list, edge_counts


def train_epoch(model, loader, optimizer, criterion):
    """Run one training epoch and return mean loss."""
    model.train()
    total_loss, total_n = 0.0, 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_n    += batch.num_graphs
    return total_loss / total_n


@torch.no_grad()
def evaluate(model, loader):
    """Return AUC, predicted probabilities, and true labels."""
    model.eval()
    probs, trues = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        out   = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        probs.extend(torch.sigmoid(out).cpu().numpy().tolist())
        trues.extend(batch.y.cpu().numpy().tolist())
    auc = roc_auc_score(trues, probs) if len(set(trues)) > 1 else 0.5
    return auc, probs, trues


def run_cv(data_list, labels):
    """
    5-fold stratified cross-validation with early stopping.

    Uses 20% of training subjects for validation (larger than the previous
    10% to give more stable early stopping signals with this dataset size).
    """
    skf          = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []
    all_probs    = np.zeros(len(labels))

    n_pos      = labels.sum()
    n_neg      = len(labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float).to(DEVICE)

    for fold, (tr_val_idx, te_idx) in enumerate(skf.split(np.arange(len(labels)), labels)):
        print(f"\nFold {fold + 1}/{N_FOLDS}  (train={len(tr_val_idx)}, test={len(te_idx)})")

        n_val   = max(1, int(len(tr_val_idx) * VAL_FRAC))
        rng     = np.random.default_rng(RANDOM_SEED + fold)
        val_idx = rng.choice(tr_val_idx, size=n_val, replace=False)
        tr_idx  = np.setdiff1d(tr_val_idx, val_idx)

        print(f"  Training: {len(tr_idx)}  Validation: {len(val_idx)}  Test: {len(te_idx)}")

        tr_loader  = DataLoader([data_list[i] for i in tr_idx],  batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader([data_list[i] for i in val_idx], batch_size=BATCH_SIZE, shuffle=False)
        te_loader  = DataLoader([data_list[i] for i in te_idx],  batch_size=BATCH_SIZE, shuffle=False)

        torch.manual_seed(RANDOM_SEED + fold)
        model     = BrainGCN().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=10, factor=0.5, min_lr=1e-5
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_auc   = 0.0
        best_weights   = None
        patience_count = 0

        for epoch in range(1, EPOCHS + 1):
            loss          = train_epoch(model, tr_loader, optimizer, criterion)
            val_auc, _, _ = evaluate(model, val_loader)
            scheduler.step(val_auc)

            if epoch % 20 == 0:
                print(f"  Epoch {epoch:3d}  loss={loss:.4f}  val_AUC={val_auc:.3f}")

            if val_auc > best_val_auc:
                best_val_auc   = val_auc
                best_weights   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}  (best val AUC={best_val_auc:.3f})")
                break

        model.load_state_dict({k: v.to(DEVICE) for k, v in best_weights.items()})
        te_auc, te_probs, te_trues = evaluate(model, te_loader)

        te_preds = [1 if p >= 0.5 else 0 for p in te_probs]
        acc  = accuracy_score(te_trues, te_preds)
        cm   = confusion_matrix(te_trues, te_preds, labels=[0, 1])
        sens = cm[1, 1] / max(cm[1].sum(), 1)
        spec = cm[0, 0] / max(cm[0].sum(), 1)

        print(f"  Result: Acc={acc:.3f}  AUC={te_auc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}")

        for idx, prob in zip(te_idx, te_probs):
            all_probs[idx] = prob

        fold_results.append({
            "fold": fold + 1,
            "acc":  acc,
            "auc":  te_auc,
            "sens": sens,
            "spec": spec,
            "cm":   cm,
            "te_idx": te_idx,
        })

    return fold_results, all_probs


def compute_node_importance(connectomes, labels, fold_results, threshold):
    """
    Compute node importance as mean |ASD - Control| node feature difference
    averaged across held-out test subjects and folds.
    """
    R        = connectomes.shape[1]
    node_imp = np.zeros((R, 5))

    for fold_res in fold_results:
        te = fold_res["te_idx"]
        feats = []
        for i in te:
            _, _, x, _ = build_sparse_graph(connectomes[i], threshold)
            feats.append((x, labels[i]))
        asd_f  = np.stack([f for f, l in feats if l == 1])
        ctrl_f = np.stack([f for f, l in feats if l == 0])
        node_imp += np.nan_to_num(np.abs(asd_f.mean(0) - ctrl_f.mean(0)))

    return node_imp / N_FOLDS


def print_summary(fold_results):
    """Print mean +/- SD for each metric across folds."""
    print("\n=== GCN Summary ===")
    rows = []
    for key, label in {"acc": "Accuracy", "auc": "AUC-ROC",
                        "sens": "Sensitivity (ASD)", "spec": "Specificity (CTRL)"}.items():
        vals   = [r[key] for r in fold_results]
        mu, sd = np.mean(vals), np.std(vals)
        print(f"  {label:25s}: {mu:.3f} +/- {sd:.3f}")
        rows.append({"Metric": label, "Mean": round(mu, 3), "SD": round(sd, 3)})
    return rows


def save(fold_results, all_probs, labels, node_imp, summary_rows):
    """Save GCN cross-validation results."""
    with open(RESULTS_DIR / "gnn_cv_results.pkl", "wb") as f:
        pickle.dump({"fold_results": fold_results, "all_probs": all_probs,
                     "labels": labels, "node_imp": node_imp}, f)
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "gnn_metrics.csv", index=False)
    print("\nSaved results/gnn_cv_results.pkl and gnn_metrics.csv")
    print("Next: python scripts/06_figures.py")


def main():
    print("=" * 60)
    print("  BrainGCN -- PyTorch Geometric")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    connectomes = np.load(DATA_DIR / "connectomes_harmonized.npy")
    labels      = np.load(DATA_DIR / "labels.npy")

    print(f"\nSubjects : {len(labels)}  ASD={labels.sum()}  CTRL={(labels == 0).sum()}")
    print(f"GCN edge threshold : |z| > {GCN_THRESHOLD}  (sparser than GB pipeline threshold of 0.20)")
    print(f"Architecture : GCNConv(5->64)+BN -> GCNConv(64->64)+BN -> MeanPool -> FC(64->32) -> FC(32->1)")
    print(f"Epochs: {EPOCHS}  Batch: {BATCH_SIZE}  LR: {LR}  Dropout: {DROPOUT}")
    print(f"Early stopping: patience={PATIENCE} epochs | Val split: {int(VAL_FRAC*100)}% of training set")

    print(f"\nBuilding sparse graphs at |z| > {GCN_THRESHOLD} ...")
    data_list, edge_counts = connectomes_to_pyg(connectomes, labels, GCN_THRESHOLD)
    ec = np.array(edge_counts)
    print(f"  Edges per subject : {ec.mean():.0f} +/- {ec.std():.0f}")
    print(f"  Graph density     : {ec.mean() / (200 * 199):.3f}  (was 0.602 at threshold 0.20)")

    print(f"\n5-Fold Cross-Validation (GCN on ComBat-harmonized ABIDE connectomes)\n")
    fold_results, all_probs = run_cv(data_list, labels)

    node_imp     = compute_node_importance(connectomes, labels, fold_results, GCN_THRESHOLD)
    summary_rows = print_summary(fold_results)
    save(fold_results, all_probs, labels, node_imp, summary_rows)


if __name__ == "__main__":
    main()