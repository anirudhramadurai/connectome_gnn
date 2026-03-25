"""
02_build_graphs.py
------------------
Converts each subject's 200×200 Fisher z-transformed correlation matrix
into a sparse graph suitable for the GCN.

Graph construction decisions:
  - Edge threshold : |z| > 0.20  (standard for fMRI functional connectivity graphs;
                                   retains ~15–30% of edges depending on subject)
  - Edge weights   : the raw Fisher z value (preserves connectivity strength)
  - Node features  : 5 per ROI — see below

Node features (all computed from the thresholded graph):
  0  mean_fc    — average Fisher z to all other ROIs (overall connectivity)
  1  degree     — number of edges above threshold (hubness)
  2  clustering — local clustering coefficient (local integration)
  3  pos_fc     — mean positive FC (excitatory connectivity profile)
  4  neg_fc     — mean negative / anti-correlated FC (inhibitory profile)
"""

import numpy as np
import pickle
from pathlib import Path

# ── Load ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
connectomes = np.load(DATA_DIR / "connectomes.npy")   # (N, 200, 200)
labels      = np.load(DATA_DIR / "labels.npy")
with open(DATA_DIR / "roi_meta.pkl", "rb") as f:
    meta = pickle.load(f)

N, R, _ = connectomes.shape
THRESHOLD = 0.20
print(f"Building graphs for {N} subjects | {R} ROIs | threshold |z| > {THRESHOLD}")


def build_edge_list(mat, threshold=THRESHOLD):
    """
    Returns:
      edge_index  — (2, E) int array of (source, dest) node indices
      edge_weight — (E,)   float array of Fisher z edge weights
    """
    mask = np.abs(mat) > threshold
    np.fill_diagonal(mask, False)
    src, dst    = np.where(mask)
    edge_weight = mat[src, dst]
    edge_index  = np.stack([src, dst], axis=0)
    return edge_index.astype(np.int32), edge_weight.astype(np.float32)


def compute_node_features(mat, threshold=THRESHOLD):
    """
    Returns (R, 5) float32 feature matrix.
    """
    adj = (np.abs(mat) > threshold).astype(np.float32)
    np.fill_diagonal(adj, 0)

    mean_fc = mat.mean(axis=1)
    degree  = adj.sum(axis=1)

    # Clustering coefficient: fraction of a node's neighbour pairs that are
    # also connected to each other (measures local network cliquishness)
    adj2     = adj @ adj
    triangles = (adj2 * adj).sum(axis=1)
    possible  = degree * (degree - 1)
    clust     = np.where(possible > 0, triangles / possible, 0.0)

    pos_fc = np.where(mat > threshold,  mat, 0.0).mean(axis=1)
    neg_fc = np.where(mat < -threshold, mat, 0.0).mean(axis=1)

    return np.stack([mean_fc, degree, clust, pos_fc, neg_fc], axis=1).astype(np.float32)


# ── Process all subjects ──────────────────────────────────────────────────────
graphs     = []
edge_counts = []

for i, mat in enumerate(connectomes):
    ei, ew = build_edge_list(mat)
    x      = compute_node_features(mat)
    graphs.append({
        "edge_index":  ei,
        "edge_weight": ew,
        "x":           x,
        "y":           int(labels[i]),
        "n_edges":     ei.shape[1],
    })
    edge_counts.append(ei.shape[1])

# ── Summary ───────────────────────────────────────────────────────────────────
edge_arr = np.array(edge_counts)
print(f"\nGraph statistics:")
print(f"  Edges per subject : {edge_arr.mean():.0f} ± {edge_arr.std():.0f}")
print(f"  Min / Max edges   : {edge_arr.min()} / {edge_arr.max()}")
print(f"  Graph density     : {edge_arr.mean() / (R * (R-1)):.3f}")
print(f"  Node feature dim  : {graphs[0]['x'].shape[1]}")

asd_e  = edge_arr[labels == 1].mean()
ctrl_e = edge_arr[labels == 0].mean()
print(f"\n  Mean edges — ASD: {asd_e:.0f}  |  CTRL: {ctrl_e:.0f}")
print(f"  (Lower in ASD = reduced connectivity, consistent with literature)")

# ── Save ──────────────────────────────────────────────────────────────────────
with open(DATA_DIR / "graphs.pkl", "wb") as f:
    pickle.dump(graphs, f)

print(f"\nSaved {len(graphs)} graph objects → data/graphs.pkl")
print("Ready for 03_train_evaluate.py")
