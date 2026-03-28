"""
02_build_graphs.py
------------------
Converts each subject's 200x200 Fisher z-transformed correlation matrix
into a graph representation with node-level features for classification.

Graph construction decisions
-----------------------------
Edge threshold : |z| > 0.20 -- standard threshold for fMRI functional
                 connectivity graphs (Bullmore & Sporns 2009). Retains
                 the strongest 15-30% of connections depending on subject.
Edge weights   : raw Fisher z values, preserving connectivity strength
Node features  : 5 per ROI, computed from the thresholded adjacency matrix

Node features
-------------
  0  mean_fc    -- average Fisher z to all other ROIs (overall connectivity)
  1  degree     -- number of edges above threshold (hubness)
  2  clustering -- local clustering coefficient (local network cliquishness)
  3  pos_fc     -- mean positive FC (excitatory connectivity profile)
  4  neg_fc     -- mean negative / anti-correlated FC (inhibitory profile)

Usage
-----
  python scripts/02_build_graphs.py

Inputs
------
  data/connectomes.npy   from 01_fetch_and_prepare.py
  data/labels.npy        from 01_fetch_and_prepare.py
  data/roi_meta.pkl      from 01_fetch_and_prepare.py

Outputs
-------
  data/graphs.pkl        list of graph dicts, one per subject

References
----------
Bullmore E, Sporns O. (2009). Complex brain networks: graph theoretical
  analysis of structural and functional systems. Nat Rev Neurosci,
  10(3):186-198. doi:10.1038/nrn2575.
"""

import pickle
import numpy as np
from pathlib import Path

# Configuration

THRESHOLD = 0.20   # absolute Fisher z threshold for edge inclusion

DATA_DIR = Path(__file__).parent.parent / "data"


# Load

def load_data():
    """Load connectomes, labels, and ROI metadata from data/."""
    connectomes = np.load(DATA_DIR / "connectomes.npy")   # (N, 200, 200)
    labels      = np.load(DATA_DIR / "labels.npy")        # (N,)

    with open(DATA_DIR / "roi_meta.pkl", "rb") as f:
        roi_meta = pickle.load(f)

    return connectomes, labels, roi_meta


# Graph construction

def build_edge_list(mat, threshold=THRESHOLD):
    """
    Threshold the connectivity matrix and return a sparse edge representation.

    Edges are included for all pairs (i, j) where |z_ij| > threshold.
    Self-connections (diagonal) are always excluded.

    Parameters
    ----------
    mat       : (R, R) float array of Fisher z-transformed correlations
    threshold : float, absolute value cutoff for edge inclusion

    Returns
    -------
    edge_index  : (2, E) int32 array of (source, destination) node indices
    edge_weight : (E,) float32 array of Fisher z edge weights
    """
    mask = np.abs(mat) > threshold
    np.fill_diagonal(mask, False)

    src, dst    = np.where(mask)
    edge_weight = mat[src, dst]
    edge_index  = np.stack([src, dst], axis=0)

    return edge_index.astype(np.int32), edge_weight.astype(np.float32)


def compute_node_features(mat, threshold=THRESHOLD):
    """
    Compute a 5-dimensional feature vector for each ROI.

    Features are derived from the thresholded adjacency matrix and the
    original Fisher z correlation values.

    Parameters
    ----------
    mat       : (R, R) float array of Fisher z-transformed correlations
    threshold : float, edge inclusion threshold

    Returns
    -------
    features : (R, 5) float32 array
    """
    adj = (np.abs(mat) > threshold).astype(np.float32)
    np.fill_diagonal(adj, 0)

    mean_fc = mat.mean(axis=1)
    degree  = adj.sum(axis=1)

    # Clustering coefficient: fraction of a node's neighbour pairs that are
    # also connected to each other. Measures local network cliquishness.
    adj2      = adj @ adj
    triangles = (adj2 * adj).sum(axis=1)
    possible  = degree * (degree - 1)
    clust     = np.where(possible > 0, triangles / possible, 0.0)

    pos_fc = np.where(mat >  threshold, mat, 0.0).mean(axis=1)
    neg_fc = np.where(mat < -threshold, mat, 0.0).mean(axis=1)

    return np.stack([mean_fc, degree, clust, pos_fc, neg_fc], axis=1).astype(np.float32)


# Process all subjects

def build_all_graphs(connectomes, labels):
    """
    Build graph objects for all subjects.

    Each graph is stored as a dict with keys:
      edge_index  : (2, E) int32
      edge_weight : (E,) float32
      x           : (200, 5) float32 node feature matrix
      y           : int, diagnostic label (1 = ASD, 0 = Control)
      n_edges     : int, number of edges above threshold

    Parameters
    ----------
    connectomes : (N, 200, 200) float64 array
    labels      : (N,) int array

    Returns
    -------
    graphs      : list of N graph dicts
    edge_counts : (N,) int array
    """
    graphs      = []
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

    return graphs, np.array(edge_counts)


# Summary

def print_summary(graphs, edge_counts, labels):
    """Print graph construction statistics."""
    R = graphs[0]["x"].shape[0]

    print("\nGraph statistics:")
    print(f"  Edges per subject : {edge_counts.mean():.0f} +/- {edge_counts.std():.0f}")
    print(f"  Min / Max edges   : {edge_counts.min()} / {edge_counts.max()}")
    print(f"  Graph density     : {edge_counts.mean() / (R * (R - 1)):.3f}")
    print(f"  Node feature dim  : {graphs[0]['x'].shape[1]}")

    asd_edges  = edge_counts[labels == 1].mean()
    ctrl_edges = edge_counts[labels == 0].mean()
    print(f"\n  Mean edges -- ASD: {asd_edges:.0f}  |  CTRL: {ctrl_edges:.0f}")


# Save

def save(graphs):
    """Save graph objects to data/graphs.pkl."""
    with open(DATA_DIR / "graphs.pkl", "wb") as f:
        pickle.dump(graphs, f)

    print(f"\nSaved {len(graphs)} graph objects to data/graphs.pkl")
    print("Next: python scripts/03_train_evaluate.py")


# Main

def main():
    connectomes, labels, roi_meta = load_data()
    N, R, _ = connectomes.shape

    print(f"Building graphs for {N} subjects | {R} ROIs | threshold |z| > {THRESHOLD}")

    graphs, edge_counts = build_all_graphs(connectomes, labels)
    print_summary(graphs, edge_counts, labels)
    save(graphs)


if __name__ == "__main__":
    main()
