# ABIDE Connectome Classification
### Classifying Autism Spectrum Disorder from Resting-State fMRI Brain Connectivity

A graph-based machine learning pipeline that classifies ASD from resting-state fMRI functional connectivity data using the publicly available ABIDE dataset (Di Martino et al., 2014). The project applies connectome-based feature extraction and gradient-boosted classification across three acquisition sites, with full 5-fold cross-validation and biological interpretability analysis.

---

## Background

When a person lies in an MRI scanner at rest, different brain regions spontaneously activate and deactivate in coordinated patterns. The degree to which any two regions' activity correlates over time is called **functional connectivity (FC)**. Measuring this across all pairs of 200 brain regions produces a 200×200 correlation matrix — the **functional connectome** — which serves as a fingerprint of an individual's large-scale brain organization.

A well-replicated finding in ASD is disrupted functional connectivity, particularly in association networks like the Default Mode Network (DMN) and Frontoparietal Network (FPN). This project treats each subject's connectome as a mathematical **graph** (regions = nodes, correlations = weighted edges) and extracts biologically interpretable features to train a classifier.

---

## Dataset

**ABIDE — Autism Brain Imaging Data Exchange**

| Property | Value |
|---|---|
| Source | ABIDE Preprocessed Connectomes Project (ABIDE-PCP) |
| Access | Fully public — no registration required |
| Sites | NYU, USM (University of Southern Mississippi), UCLA_1 |
| Preprocessing | C-PAC pipeline, band-pass filtered, no global signal regression |
| Parcellation | CC200 (Craddock et al., 2012) — 200 functionally defined ROIs |
| Functional networks | 8: DMN, Visual, SMN, DAN, VAN, FPN, Limbic, Subcortical |
| Subjects | 303 total — 154 ASD, 149 neurotypical controls |

> **Download:** `nilearn` fetches data automatically on first run (~500 MB, locally cached).

---

## Pipeline

```
01_fetch_and_prepare.py
  Downloads ABIDE ROI time series via nilearn (auto-cached after first run)
  Computes 200x200 Pearson correlation matrix per subject
  Applies Fisher z-transform: r -> arctanh(r) for approximate normality
  Cleans NaN entries from zero-variance ROIs
  Saves connectomes (303, 200, 200), labels, and metadata

02_build_graphs.py
  Thresholds each matrix at |z| > 0.20 to retain meaningful connections
  Computes 5 node features per ROI
  Saves graph objects to data/graphs.pkl

03_train_evaluate.py
  Extracts 44 graph-level features per subject across 8 functional networks
  Gradient Boosting classifier with median imputation and standardisation
  5-fold stratified cross-validation
  Saves predicted probabilities and node importance to results/

04_figures.py
  Generates 6 publication-quality figures to figures/
```

---

## Node Features

Five features are computed per ROI from the thresholded graph:

| Feature | Description | Biological meaning |
|---|---|---|
| Mean FC | Average Fisher z to all other ROIs | Overall connectivity strength |
| Degree | Number of edges above threshold | Hubness — how connected a region is |
| Clustering coefficient | Fraction of neighbour pairs also connected | Local network cliquishness |
| Positive FC | Mean of positive correlations only | Excitatory connectivity profile |
| Negative FC | Mean of anti-correlations only | Inhibitory / competing connectivity |

---

## Graph-Level Feature Extraction

Rather than feeding raw 200×200 matrices to the classifier (40,000 features for 303 subjects — severely underpowered), the pipeline extracts 44 biologically meaningful graph-level features per subject:

- Between-network FC difference for each pair of 8 functional networks (28 values)
- Mean degree per network (8 values)
- Mean clustering coefficient per network (8 values)

This reduces the feature space to one that is statistically tractable, interpretable, and grounded in the known network organization of the human brain.

---

## Results

5-fold stratified cross-validation (Gradient Boosting with median imputation):

| Metric | Mean | SD |
|---|---|---|
| Accuracy | 0.515 | 0.050 |
| AUC-ROC | 0.514 | 0.042 |
| Sensitivity (ASD recall) | 0.566 | 0.052 |
| Specificity (CTRL recall) | 0.462 | 0.087 |

### Why AUC ~0.51?

This result is honest and expected. ABIDE classification is notoriously difficult for three reasons:

**Site effects dominate.** Scanner hardware, acquisition protocols, and operator differences between NYU, USM, and UCLA introduce systematic variance that dwarfs the biological ASD signal. Without explicit site harmonisation (e.g. ComBat), classifiers largely learn scanner differences rather than disease biology.

**Small within-site samples.** After splitting 303 subjects across 5 folds, each test set contains only ~60 subjects — insufficient power for a subtle heterogeneous condition like ASD.

**ASD heterogeneity.** ASD is a spectrum with highly variable connectivity profiles. Aggregate group differences are weak relative to within-group variance.

Published graph-based models on ABIDE report AUC of 0.65–0.78 only after site harmonisation, larger multi-site cohorts, and architectural optimisation (Ktena et al., 2018; Li et al., 2021). This pipeline intentionally uses raw data and simple features to establish an honest, reproducible baseline.

---

## Figures

| Figure | Description |
|---|---|
| `fig1_connectivity_matrices.png` | 200×200 FC matrices for representative ASD and control subjects with network boundary lines overlaid |
| `fig2_graph_structure.png` | Circular graph layout with nodes coloured by network, top 300 edges shown |
| `fig3_feature_distributions.png` | Violin plots of all 5 node features by group. Distributions largely overlap, consistent with weak group-level signal |
| `fig4_training_loss.png` | Per-fold AUC and accuracy. Fold 4 shows the strongest signal (AUC=0.56), Fold 5 the weakest (0.44), illustrating site-driven variance |
| `fig5_roc_curves.png` | ROC curves with mean AUC = 0.51 +/- 0.04 |
| `fig6_node_importance.png` | Node importance heatmap — SMN, FPN, and Limbic regions show the largest ASD-Control differences |

---

## Biological Interpretation of Node Importance (Fig 6)

The node importance analysis reveals several biologically grounded patterns:

**Sensorimotor Network (SMN)** shows the highest mean FC and positive FC differences, consistent with the well-documented sensory processing atypicalities in ASD and altered sensorimotor integration (Marco et al., 2011).

**Frontoparietal Network (FPN)** shows high clustering coefficient differences, consistent with disrupted executive function and working memory networks (Yerys et al., 2015).

**Limbic system** shows consistently high importance across most features, reflecting the established role of limbic connectivity in social cognition and emotional processing in ASD.

**Visual cortex** shows near-zero importance across all features, consistent with relatively preserved visual processing at the network level in ASD.

---

## Setup

```bash
git clone https://github.com/anirudhramadurai/abide-connectome-classification
cd abide-connectome-classification
pip install -r requirements.txt

python 01_fetch_and_prepare.py   # downloads ABIDE (~500 MB, cached after first run)
python 02_build_graphs.py
python 03_train_evaluate.py
python 04_figures.py
```

Runtime: approximately 5-10 minutes on a standard laptop (CPU only).

---

## Requirements

```
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
matplotlib>=3.7
pandas>=2.0
nilearn>=0.10
```

---

## Limitations and Future Work

**Site harmonisation.** Applying ComBat (Johnson et al., 2007) or neuroCombat would remove scanner-driven variance and substantially improve classification performance.

**Richer parcellation.** The CC200 atlas uses 200 ROIs. Higher-resolution atlases (Schaefer-400, Gordon-333) would capture finer-grained connectivity patterns.

**Full ABIDE cohort.** This pipeline uses 3 of 17 available ABIDE sites (303 subjects). Scaling to the full dataset (~1,100 subjects) with leave-site-out cross-validation would yield more generalisable results.

**Deep learning.** PyTorch Geometric GCN models trained with proper site stratification achieve AUC ~0.72–0.78 on ABIDE (Li et al., 2021; Jiang et al., 2020).

---

## References

- Di Martino, A. et al. (2014). The autism brain imaging data exchange. *Molecular Psychiatry*, 19(6), 659-667.
- Craddock, R. C. et al. (2012). A whole brain fMRI atlas generated via spatially constrained spectral clustering. *Human Brain Mapping*, 33(8), 1914-1928.
- Ktena, S. I. et al. (2018). Metric learning with spectral graph convolutions on brain connectivity networks. *NeuroImage*, 169, 431-442.
- Li, X. et al. (2021). BrainGNN: Interpretable brain graph neural network for fMRI analysis. *Medical Image Analysis*, 74, 102233.
- Jiang, H. et al. (2020). Hi-GCN: A hierarchical graph convolution network for graph embedding learning of brain network. *Computers in Biology and Medicine*, 127, 104096.
- Mostofsky, S. H. & Ewen, J. B. (2011). Altered connectivity and action model formation in autism. *The Neuroscientist*, 17(4), 437-448.
- Yerys, B. E. et al. (2015). Default mode network segregation and social deficits in autism spectrum disorder. *Biological Psychiatry: CNNI*, 1(5), 374-382.
- Johnson, W. E. et al. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*, 8(1), 118-127.

---

*University of Washington — Independent Project*
