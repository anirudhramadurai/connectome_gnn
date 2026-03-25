"""
04_figures.py
-------------
Generates 6 publication-quality figures from real ABIDE connectome data.

Figure 1 — Functional connectivity matrices: ASD vs. Control
           Shows the raw 200×200 Fisher z-transformed correlation matrices
           for a representative subject from each group. Warmer colours =
           stronger positive correlation. Network boundary lines are overlaid.

Figure 2 — Brain graph visualisation (circular layout)
           Each of the 200 ROIs is a node arranged in a circle, coloured
           by functional network. The top 300 edges by absolute weight are
           drawn, coloured by sign (red = positive FC, blue = anti-correlation).
           ASD subjects visually show sparser long-range connections.

Figure 3 — Node feature distributions by group
           Violin plots comparing ASD vs. Control distributions for each of
           the 5 node features across all subjects and ROIs. Reveals the
           feature-level signal the GCN is exploiting.

Figure 4 — Training loss curves
           BCE loss per epoch across all 5 CV folds. Shows convergence
           behaviour and consistency across folds.

Figure 5 — ROC curves (5-fold CV)
           One ROC curve per fold plus the interpolated mean ± SD band.
           The mean AUC summarises classification performance.

Figure 6 — Node importance heatmap (brain network × feature)
           Visualises which brain networks and node features show the
           largest mean absolute difference between ASD and Control groups,
           averaged across all 5 held-out test sets. Highlights where in
           the brain and what aspect of connectivity the GCN is sensitive to.
"""

import pickle, sys, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from sklearn.metrics import roc_curve, auc as sk_auc
from sklearn.model_selection import StratifiedKFold

# ── Style ─────────────────────────────────────────────────────────────────────
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
GROUP_COLORS  = {"ASD": "#D95F3B", "CTRL": "#3A78B5"}
CMAP_FC       = LinearSegmentedColormap.from_list("fc", ["#2166AC","#F7F7F7","#D6604D"])
NET_COLORS    = {
    "DMN":         "#D95F3B",
    "Visual":      "#3A78B5",
    "SMN":         "#5BAD72",
    "DAN":         "#8B6BBD",
    "VAN":         "#D4A843",
    "FPN":         "#C0456A",
    "Limbic":      "#44AAAA",
    "Subcortical": "#888888",
}
FEAT_LABELS   = ["Mean FC", "Degree", "Clustering\nCoeff", "Positive FC", "Negative FC"]
OUTDIR        = Path("figures")
OUTDIR.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
connectomes = np.load("data/connectomes.npy")
labels      = np.load("data/labels.npy")
meta        = pd.read_csv("data/metadata.csv")

with open("data/graphs.pkl", "rb") as f:
    graphs = pickle.load(f)
with open("data/roi_meta.pkl", "rb") as f:
    roi_meta = pickle.load(f)
with open("results/cv_results.pkl", "rb") as f:
    res = pickle.load(f)

networks = np.array(roi_meta["networks"])
NETS     = list(dict.fromkeys(roi_meta["networks"]))   # unique, order-preserved

N_ROIS = connectomes.shape[1]


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Connectivity matrices
# ─────────────────────────────────────────────────────────────────────────────
def fig1_matrices():
    ctrl_idx = np.where(labels == 0)[0][0]
    asd_idx  = np.where(labels == 1)[0][0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, idx, group in zip(axes, [ctrl_idx, asd_idx], ["CTRL", "ASD"]):
        mat = connectomes[idx]
        im  = ax.imshow(mat, cmap=CMAP_FC, vmin=-1.5, vmax=1.5, aspect="auto")

        # Network boundary lines
        cumsum = 0
        boundaries = []
        for net in NETS:
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
        for net in NETS:
            n = (networks == net).sum()
            ax.text(-5, cumsum + n/2, net, ha="right", va="center",
                    fontsize=6, color=NET_COLORS[net], fontweight="bold")
            cumsum += n

    cb = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cb.set_label("Fisher z (functional connectivity)", fontsize=10)
    fig.suptitle(
        "Resting-State Functional Connectivity Matrices\n"
        "ABIDE dataset — CC200 parcellation (200 ROIs, 8 networks)",
        fontsize=13, fontweight="bold", y=1.02
    )
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig1_connectivity_matrices.png", bbox_inches="tight")
    plt.close()
    print("Fig 1 saved")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Graph visualisation
# ─────────────────────────────────────────────────────────────────────────────
def fig2_graph_viz():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, group_val, group_name in zip(axes, [0, 1], ["CTRL", "ASD"]):
        idx = next(i for i, g in enumerate(graphs) if g["y"] == group_val)
        g   = graphs[idx]
        N   = g["x"].shape[0]
        ei  = g["edge_index"]
        ew  = g["edge_weight"]

        # Circular layout: ROIs arranged in network order around the circle
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pos   = np.stack([np.cos(theta), np.sin(theta)], axis=1)

        # Draw top 300 edges by absolute connectivity strength
        top_e = np.argsort(np.abs(ew))[::-1][:300]
        for e in top_e:
            s, d = ei[0, e], ei[1, e]
            w    = float(ew[e])
            # Red = positive FC, Blue = anti-correlated
            col  = "#D6604D" if w > 0 else "#2166AC"
            ax.plot([pos[s, 0], pos[d, 0]], [pos[s, 1], pos[d, 1]],
                    color=col, alpha=0.10, lw=0.5, zorder=1)

        # Draw nodes coloured by functional network
        for i in range(N):
            ax.scatter(pos[i, 0], pos[i, 1], s=20,
                       color=NET_COLORS[networks[i]],
                       zorder=3, edgecolors="none")

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{group_name}\n{g['n_edges']} edges above threshold",
                     fontsize=12, fontweight="bold",
                     color=GROUP_COLORS[group_name])

    # Network legend
    net_handles = [mpatches.Patch(color=NET_COLORS[n], label=n) for n in NETS]
    # Edge colour legend
    edge_handles = [
        plt.Line2D([0],[0], color="#D6604D", lw=1.5, label="Positive FC"),
        plt.Line2D([0],[0], color="#2166AC", lw=1.5, label="Anti-correlation"),
    ]
    axes[1].legend(handles=net_handles + edge_handles,
                   title="Network / Edge type",
                   fontsize=7, title_fontsize=8,
                   loc="lower right", bbox_to_anchor=(1.32, 0),
                   frameon=False)

    fig.suptitle(
        "Brain Graph Structure — Circular Layout\n"
        "Nodes = CC200 ROIs, coloured by network | "
        "Top 300 edges shown | Threshold |z| > 0.20",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(OUTDIR / "fig2_graph_structure.png", bbox_inches="tight")
    plt.close()
    print("Fig 2 saved")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Node feature distributions
# ─────────────────────────────────────────────────────────────────────────────
def fig3_features():
    asd_feats  = np.stack([g["x"] for g in graphs if g["y"] == 1])   # (N_asd, R, 5)
    ctrl_feats = np.stack([g["x"] for g in graphs if g["y"] == 0])

    # Flatten across subjects and ROIs: (N_asd * R, 5)
    asd_flat  = asd_feats.reshape(-1, 5)
    ctrl_flat = ctrl_feats.reshape(-1, 5)

    fig, axes = plt.subplots(1, 5, figsize=(14, 4.5))

    for fi, (ax, fname) in enumerate(zip(axes, FEAT_LABELS)):
        for group_name, data, col in [
            ("CTRL", ctrl_flat[:, fi], GROUP_COLORS["CTRL"]),
            ("ASD",  asd_flat[:, fi],  GROUP_COLORS["ASD"]),
        ]:
            # Subsample 3000 values for violin (all data is equivalent, just faster)
            rng    = np.random.default_rng(fi)
            sample = data[rng.choice(len(data), min(3000, len(data)), replace=False)]
            pos    = 0 if group_name == "CTRL" else 1
            parts  = ax.violinplot([sample], positions=[pos],
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
        "All 200 CC200 regions × all ABIDE subjects",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(OUTDIR / "fig3_feature_distributions.png", bbox_inches="tight")
    plt.close()
    print("Fig 3 saved")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Training loss
# ─────────────────────────────────────────────────────────────────────────────
def fig4_loss():
    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = ["#3A78B5","#D95F3B","#5BAD72","#8B6BBD","#D4A843"]

    all_losses = []
    for fi, fold_res in enumerate(res["fold_results"]):
        losses = fold_res["tr_losses"]
        epochs = np.arange(1, len(losses) + 1)
        ax.plot(epochs, losses, alpha=0.45, lw=1.5,
                color=colors[fi], label=f"Fold {fi+1}")
        all_losses.append(losses)

    mean_loss = np.mean(all_losses, axis=0)
    ax.plot(epochs, mean_loss, color="#111111", lw=2.5,
            label=f"Mean (5-fold)", zorder=5)

    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Binary cross-entropy loss")
    ax.set_title(
        "GCN Training Loss Curves — 5-Fold Cross-Validation\n"
        "One curve per fold; black = mean across folds",
        fontweight="bold"
    )
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    fig.savefig(OUTDIR / "fig4_training_loss.png", bbox_inches="tight")
    plt.close()
    print("Fig 4 saved")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — ROC curves
# ─────────────────────────────────────────────────────────────────────────────
def fig5_roc():
    probs  = res["all_probs"]
    labs   = res["labels"]
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    mean_fpr = np.linspace(0, 1, 300)
    tprs, aucs = [], []

    colors = ["#3A78B5","#D95F3B","#5BAD72","#8B6BBD","#D4A843"]
    for fi, (_, te_idx) in enumerate(skf.split(np.arange(len(labs)), labs)):
        fpr, tpr, _ = roc_curve(labs[te_idx], probs[te_idx])
        auc_score   = sk_auc(fpr, tpr)
        aucs.append(auc_score)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        ax.plot(fpr, tpr, alpha=0.40, lw=1.3, color=colors[fi],
                label=f"Fold {fi+1}  (AUC = {auc_score:.2f})")

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr  = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color="#111111", lw=2.5,
            label=f"Mean  (AUC = {mean_auc:.2f} ± {std_auc:.2f})", zorder=5)
    ax.fill_between(mean_fpr,
                    np.clip(mean_tpr - std_tpr, 0, 1),
                    np.clip(mean_tpr + std_tpr, 0, 1),
                    alpha=0.15, color="#888888", label="± 1 SD")
    ax.plot([0, 1], [0, 1], "--", color="#aaaaaa", lw=1.2, label="Random chance")

    ax.set_xlabel("False Positive Rate\n(1 − Specificity)")
    ax.set_ylabel("True Positive Rate\n(Sensitivity / ASD recall)")
    ax.set_title(
        "ROC Curves — ASD vs. Neurotypical Classification\n"
        "5-Fold Stratified Cross-Validation on ABIDE",
        fontweight="bold"
    )
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    plt.tight_layout()
    fig.savefig(OUTDIR / "fig5_roc_curves.png", bbox_inches="tight")
    plt.close()
    print("Fig 5 saved")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Node importance heatmap
# ─────────────────────────────────────────────────────────────────────────────
def fig6_node_importance():
    node_imp = res["node_imp"]    # (R, 5) — mean |ASD - CTRL| per ROI per feature

    # Aggregate by functional network (mean across ROIs within each network)
    net_imp = np.zeros((len(NETS), 5))
    for ni, net in enumerate(NETS):
        idx = np.where(networks == net)[0]
        net_imp[ni] = node_imp[idx].mean(axis=0)

    # Normalise each feature column to [0, 1] for visual comparability
    col_min = net_imp.min(axis=0)
    col_max = net_imp.max(axis=0)
    net_imp_norm = (net_imp - col_min) / np.maximum(col_max - col_min, 1e-9)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(net_imp_norm.T, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=1)

    ax.set_xticks(range(len(NETS)))
    ax.set_xticklabels(NETS, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(5))
    ax.set_yticklabels(FEAT_LABELS, fontsize=10)

    # Annotate each cell with its value
    for j in range(len(NETS)):
        for fi in range(5):
            val = net_imp_norm[j, fi]
            ax.text(j, fi, f"{val:.2f}", ha="center", va="center",
                    fontsize=8.5,
                    color="white" if val > 0.55 else "#2a2a2a",
                    fontweight="bold" if val > 0.75 else "normal")

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Normalised importance\n(0 = least different, 1 = most different)",
                 fontsize=9)

    ax.set_title(
        "GCN Node Importance — Which Brain Regions Drive ASD Classification?\n"
        "Mean |ASD − Control| node feature difference per network, "
        "averaged across 5 held-out test folds",
        fontweight="bold", pad=12, fontsize=11
    )
    plt.tight_layout()
    fig.savefig(OUTDIR / "fig6_node_importance.png", bbox_inches="tight")
    plt.close()
    print("Fig 6 saved")


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig1_matrices()
    fig2_graph_viz()
    fig3_features()
    fig4_loss()
    fig5_roc()
    fig6_node_importance()
    print("\nAll figures saved to figures/")
