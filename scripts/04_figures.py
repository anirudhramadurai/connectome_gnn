"""
04_figures.py
-------------
Generates 6 publication-quality figures from the ABIDE connectome pipeline.

Figure 1 -- Functional connectivity matrices (ASD vs. Control)
  Representative 200x200 Fisher z-transformed correlation matrices for one
  subject per group. Warmer colours = stronger positive correlation.
  Network boundary lines are overlaid. Colorbar anchored to the right of
  the ASD panel.

Figure 2 -- Brain graph visualisation (circular layout)
  Each of the 200 CC200 ROIs arranged as a node in a circle, coloured by
  functional network. The top 300 edges by absolute weight are drawn;
  red = positive FC, blue = anti-correlation.

Figure 3 -- Node feature distributions by group
  Violin plots comparing ASD vs. Control distributions for each of the 5
  node features across all subjects and ROIs. Reveals the degree of
  overlap between groups at the node level.

Figure 4 -- Per-fold classification performance
  Bar chart showing AUC-ROC and accuracy for each of the 5 CV folds,
  with a dashed chance line at 0.50.

Figure 5 -- ROC curves (5-fold CV)
  One ROC curve per fold plus the interpolated mean +/- 1 SD band.
  The mean AUC summarises overall classification performance.

Figure 6 -- Node importance heatmap (brain network x feature)
  Mean absolute difference between ASD and Control node features,
  aggregated by functional network and averaged across 5 held-out test
  folds. Highlights which networks and features differ most between groups.

Usage
-----
  python scripts/04_figures.py

Inputs
------
  data/connectomes.npy     from 01_fetch_and_prepare.py
  data/labels.npy          from 01_fetch_and_prepare.py
  data/metadata.csv        from 01_fetch_and_prepare.py
  data/graphs.pkl          from 02_build_graphs.py
  data/roi_meta.pkl        from 01_fetch_and_prepare.py
  results/cv_results.pkl   from 03_train_evaluate.py

Outputs
-------
  figures/fig1_connectivity_matrices.png
  figures/fig2_graph_structure.png
  figures/fig3_feature_distributions.png
  figures/fig4_training_loss.png
  figures/fig5_roc_curves.png
  figures/fig6_node_importance.png
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.model_selection import StratifiedKFold

# Style

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        150,
})

# Colour palette consistent across all figures
GROUP_COLORS = {"ASD": "#D95F3B", "CTRL": "#3A78B5"}
CMAP_FC      = LinearSegmentedColormap.from_list("fc", ["#2166AC", "#F7F7F7", "#D6604D"])
NET_COLORS   = {
    "DMN":         "#D95F3B",
    "Visual":      "#3A78B5",
    "SMN":         "#5BAD72",
    "DAN":         "#8B6BBD",
    "VAN":         "#D4A843",
    "FPN":         "#C0456A",
    "Limbic":      "#44AAAA",
    "Subcortical": "#888888",
}
FEAT_LABELS = ["Mean FC", "Degree", "Clustering\nCoeff", "Positive FC", "Negative FC"]

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTDIR      = Path(__file__).parent.parent / "figures"
OUTDIR.mkdir(exist_ok=True)


# Load

def load_data():
    """Load all pipeline outputs needed for figure generation."""
    connectomes = np.load(DATA_DIR / "connectomes.npy")
    labels      = np.load(DATA_DIR / "labels.npy")
    metadata    = pd.read_csv(DATA_DIR / "metadata.csv")

    with open(DATA_DIR / "graphs.pkl",   "rb") as f:
        graphs = pickle.load(f)
    with open(DATA_DIR / "roi_meta.pkl", "rb") as f:
        roi_meta = pickle.load(f)
    with open(RESULTS_DIR / "cv_results.pkl", "rb") as f:
        res = pickle.load(f)

    networks = np.array(roi_meta["networks"])
    nets     = list(dict.fromkeys(roi_meta["networks"]))   # unique, order-preserved

    return connectomes, labels, metadata, graphs, networks, nets, res


# Figure 1: Connectivity matrices

def fig1_matrices(connectomes, labels, networks, nets):
    """
    Plot representative 200x200 FC matrices for one CTRL and one ASD subject.
    """
    ctrl_idx = np.where(labels == 0)[0][0]
    asd_idx  = np.where(labels == 1)[0][0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, idx, group in zip(axes, [ctrl_idx, asd_idx], ["CTRL", "ASD"]):
        mat = connectomes[idx]
        im  = ax.imshow(mat, cmap=CMAP_FC, vmin=-1.5, vmax=1.5, aspect="auto")

        # Network boundary lines
        cumsum     = 0
        boundaries = []
        for net in nets:
            cumsum += (networks == net).sum()
            boundaries.append(cumsum - 0.5)
        for b in boundaries[:-1]:
            ax.axhline(b, color="white", lw=0.5, alpha=0.6)
            ax.axvline(b, color="white", lw=0.5, alpha=0.6)

        ax.set_title(group, fontsize=14, fontweight="bold",
                     color=GROUP_COLORS[group], pad=6)
        ax.set_xlabel("ROI index (CC200 parcellation)")
        ax.set_ylabel("ROI index")

        # Network labels on left edge
        cumsum = 0
        for net in nets:
            n = (networks == net).sum()
            ax.text(-5, cumsum + n / 2, net, ha="right", va="center",
                    fontsize=6, color=NET_COLORS[net], fontweight="bold")
            cumsum += n

    cb = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cb.set_label("Fisher z (functional connectivity)", fontsize=10)

    fig.suptitle(
        "Resting-State Functional Connectivity Matrices\n"
        "ABIDE dataset, CC200 parcellation (200 ROIs, 8 networks)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig1_connectivity_matrices.png", bbox_inches="tight")
    plt.close()
    print("Fig 1 saved")


# Figure 2: Brain graph visualisation

def fig2_graph_viz(graphs, networks, nets):
    """
    Circular graph layout showing the top 300 edges for one subject per group.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, group_val, group_name in zip(axes, [0, 1], ["CTRL", "ASD"]):
        idx = next(i for i, g in enumerate(graphs) if g["y"] == group_val)
        g   = graphs[idx]
        N   = g["x"].shape[0]
        ei  = g["edge_index"]
        ew  = g["edge_weight"]

        # Circular layout
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pos   = np.stack([np.cos(theta), np.sin(theta)], axis=1)

        # Top 300 edges by absolute connectivity strength
        top_e = np.argsort(np.abs(ew))[::-1][:300]
        for e in top_e:
            s, d = ei[0, e], ei[1, e]
            w    = float(ew[e])
            col  = "#D6604D" if w > 0 else "#2166AC"
            ax.plot([pos[s, 0], pos[d, 0]], [pos[s, 1], pos[d, 1]],
                    color=col, alpha=0.10, lw=0.5, zorder=1)

        for i in range(N):
            ax.scatter(pos[i, 0], pos[i, 1], s=20,
                       color=NET_COLORS[networks[i]],
                       zorder=3, edgecolors="none")

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            f"{group_name}\n{g['n_edges']} edges above threshold",
            fontsize=12, fontweight="bold", color=GROUP_COLORS[group_name],
        )

    # Legend
    net_handles = [mpatches.Patch(color=NET_COLORS[n], label=n) for n in nets]
    edge_handles = [
        plt.Line2D([0], [0], color="#D6604D", lw=1.5, label="Positive FC"),
        plt.Line2D([0], [0], color="#2166AC", lw=1.5, label="Anti-correlation"),
    ]
    axes[1].legend(
        handles=net_handles + edge_handles,
        title="Network / Edge type",
        fontsize=7, title_fontsize=8,
        loc="lower right", bbox_to_anchor=(1.32, 0),
        frameon=False,
    )

    fig.suptitle(
        "Brain Graph Structure, Circular Layout\n"
        "Nodes = CC200 ROIs, coloured by network | "
        "Top 300 edges shown | Threshold |z| > 0.20",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(OUTDIR / "fig2_graph_structure.png", bbox_inches="tight")
    plt.close()
    print("Fig 2 saved")


# Figure 3: Node feature distributions

def fig3_features(graphs):
    """
    Violin plots comparing ASD and Control distributions for each node feature.
    """
    asd_feats  = np.stack([g["x"] for g in graphs if g["y"] == 1])   # (N_asd, 200, 5)
    ctrl_feats = np.stack([g["x"] for g in graphs if g["y"] == 0])

    asd_flat  = asd_feats.reshape(-1, 5)
    ctrl_flat = ctrl_feats.reshape(-1, 5)

    fig, axes = plt.subplots(1, 5, figsize=(14, 4.5))

    for fi, (ax, fname) in enumerate(zip(axes, FEAT_LABELS)):
        for group_name, data, col in [
            ("CTRL", ctrl_flat[:, fi], GROUP_COLORS["CTRL"]),
            ("ASD",  asd_flat[:, fi],  GROUP_COLORS["ASD"]),
        ]:
            rng    = np.random.default_rng(fi)
            sample = data[rng.choice(len(data), min(3000, len(data)), replace=False)]
            pos    = 0 if group_name == "CTRL" else 1

            parts = ax.violinplot([sample], positions=[pos],
                                  widths=0.65, showmedians=True,
                                  showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(col)
                pc.set_alpha(0.75)
            parts["cmedians"].set_color("white")
            parts["cmedians"].set_linewidth(2.5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Control", "ASD"], fontsize=9)
        ax.set_title(fname, fontsize=10, fontweight="bold")
        if fi == 0:
            ax.set_ylabel("Feature value")
        ax.axhline(0, color="#cccccc", lw=0.8, ls="--")

    fig.suptitle(
        "Node Feature Distributions by Diagnostic Group\n"
        "Each observation = one ROI in one subject | "
        "All 200 CC200 regions x all ABIDE subjects",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(OUTDIR / "fig3_feature_distributions.png", bbox_inches="tight")
    plt.close()
    print("Fig 3 saved")


# Figure 4: Per-fold classification performance

def fig4_performance(res):
    """
    Bar chart showing AUC-ROC and accuracy for each cross-validation fold.
    """
    fold_aucs = [r["auc"]  for r in res["fold_results"]]
    fold_accs = [r["acc"]  for r in res["fold_results"]]
    folds     = [f"Fold {r['fold']}" for r in res["fold_results"]]

    x     = np.arange(len(folds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars_auc = ax.bar(x - width / 2, fold_aucs, width,
                      label="AUC-ROC", color="#3A78B5", alpha=0.85)
    bars_acc = ax.bar(x + width / 2, fold_accs, width,
                      label="Accuracy", color="#D95F3B", alpha=0.85)

    ax.axhline(0.5, color="#aaaaaa", lw=1.2, ls="--", label="Chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Score")
    ax.set_title(
        "Per-Fold Classification Performance\n"
        "ABIDE -- Gradient Boosting on connectome features",
        fontweight="bold",
    )
    ax.legend(fontsize=9, frameon=False)

    for bar in list(bars_auc) + list(bars_acc):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig4_training_loss.png", bbox_inches="tight")
    plt.close()
    print("Fig 4 saved")


# Figure 5: ROC curves

def fig5_roc(res):
    """
    One ROC curve per fold plus the interpolated mean +/- 1 SD band.
    """
    probs  = res["all_probs"]
    labels = res["labels"]
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fig, ax    = plt.subplots(figsize=(6, 5.5))
    mean_fpr   = np.linspace(0, 1, 300)
    tprs, aucs = [], []
    colors     = ["#3A78B5", "#D95F3B", "#5BAD72", "#8B6BBD", "#D4A843"]

    for fi, (_, te_idx) in enumerate(skf.split(np.arange(len(labels)), labels)):
        fpr, tpr, _ = roc_curve(labels[te_idx], probs[te_idx])
        auc_score   = sk_auc(fpr, tpr)
        aucs.append(auc_score)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        ax.plot(fpr, tpr, alpha=0.40, lw=1.3, color=colors[fi],
                label=f"Fold {fi + 1}  ({auc_score:.2f})")

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr  = np.std(tprs,  axis=0)
    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color="#111111", lw=2.5,
            label=f"Mean ({mean_auc:.2f} +/- {std_auc:.2f})", zorder=5)
    ax.fill_between(mean_fpr,
                    np.clip(mean_tpr - std_tpr, 0, 1),
                    np.clip(mean_tpr + std_tpr, 0, 1),
                    alpha=0.15, color="#888888", label="+/- 1 SD")
    ax.plot([0, 1], [0, 1], "--", color="#aaaaaa", lw=1.2, label="Chance")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        "ROC Curves -- ASD vs Control\n"
        "5-Fold CV on ABIDE",
        fontweight="bold",
    )
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig5_roc_curves.png", bbox_inches="tight")
    plt.close()
    print("Fig 5 saved")


# Figure 6: Node importance heatmap

def fig6_node_importance(res, networks, nets):
    """
    Heatmap of mean |ASD - Control| node feature difference, aggregated
    by functional network across all 5 held-out folds.
    """
    node_imp = res["node_imp"]   # (200, 5)

    # Aggregate by network
    net_imp = np.zeros((len(nets), 5))
    for ni, net in enumerate(nets):
        idx        = np.where(networks == net)[0]
        net_imp[ni] = node_imp[idx].mean(axis=0)

    # Normalise each feature column to [0, 1] for visual comparability
    col_min      = net_imp.min(axis=0)
    col_max      = net_imp.max(axis=0)
    net_imp_norm = (net_imp - col_min) / np.maximum(col_max - col_min, 1e-9)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(net_imp_norm.T, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(nets)))
    ax.set_xticklabels(nets, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(5))
    ax.set_yticklabels(FEAT_LABELS, fontsize=10)

    for j in range(len(nets)):
        for fi in range(5):
            val = net_imp_norm[j, fi]
            ax.text(j, fi, f"{val:.2f}",
                    ha="center", va="center", fontsize=8.5,
                    color="white" if val > 0.55 else "#2a2a2a",
                    fontweight="bold" if val > 0.75 else "normal")

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Normalised importance\n(0 = least different, 1 = most different)",
                 fontsize=9)

    ax.set_title(
        "Node Importance by Network and Feature\n"
        "Mean |ASD - Control| difference across 5 held-out folds",
        fontweight="bold", pad=12, fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(OUTDIR / "fig6_node_importance.png", bbox_inches="tight")
    plt.close()
    print("Fig 6 saved")


# Main

def main():
    connectomes, labels, metadata, graphs, networks, nets, res = load_data()

    fig1_matrices(connectomes, labels, networks, nets)
    fig2_graph_viz(graphs, networks, nets)
    fig3_features(graphs)
    fig4_performance(res)
    fig5_roc(res)
    fig6_node_importance(res, networks, nets)

    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
