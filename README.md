# abide-connectome-asd

**Graph-based classification of Autism Spectrum Disorder from resting-state fMRI functional connectivity data**

---

## Overview

Using publicly available resting-state fMRI data from 303 participants across three ABIDE sites (NYU, USM, UCLA), I built a six-script Python pipeline to classify ASD from whole-brain functional connectivity patterns. Each participant's brain is represented as a graph of 200 regions connected by functional correlations. The pipeline applies ComBat site harmonization to remove scanner-driven variance, then evaluates two classifiers: a gradient-boosted ensemble on PCA-compressed connectivity features, and a PyTorch Geometric Graph Convolutional Network.

Without harmonization, classification performance is at chance (AUC = 0.51), confirming that scanner differences between sites dominate the connectivity signal. After ComBat harmonization, PCA + gradient boosting achieves AUC = 0.723 +/- 0.036 in 5-fold cross-validation, consistent with published results on harmonized ABIDE data [5, 6]. The GCN (script 05) learns signal during training but does not generalize reliably at this dataset size, illustrating a known limitation of graph neural networks on dense brain connectivity graphs with fewer than ~500 subjects; this is documented as a limitation and future direction.

The node importance analysis reveals that limbic system connectivity shows the largest ASD-Control differences across almost all features, followed by the frontoparietal network (FPN) for clustering coefficient and anti-correlations, and the sensorimotor network (SMN) for overall mean FC. These patterns are consistent with published findings on social cognition, executive function, and sensorimotor integration differences in ASD [8, 9].

---

## What is this project?

This project applies a reproducible Python pipeline to publicly available neuroimaging data to investigate whether ASD can be detected from the pattern of functional connections between brain regions, when those connections are modeled as a graph.

The full pipeline runs in approximately 30 minutes on a standard laptop (CPU only), from automatic data download to six figures.

---

## The Biology (plain language first)

### What is ASD and why study brain connectivity?

Autism Spectrum Disorder (ASD) is a neurodevelopmental condition affecting approximately 1-2% of the global population [1], characterized by differences in social communication and restricted or repetitive behavior patterns. It is a spectrum, meaning the condition presents very differently across individuals.

Unlike conditions such as Alzheimer's disease, where specific brain regions are visibly lost, ASD does not produce obvious structural damage detectable by standard brain imaging. This has led researchers to focus instead on how brain regions **communicate with each other**: the hypothesis is that ASD arises not from failure of any single region, but from disrupted coordination across the brain's large-scale networks [1].

### What is resting-state fMRI and functional connectivity?

**Functional MRI (fMRI)** measures brain activity indirectly via the **BOLD signal** (Blood Oxygen Level Dependent) -- the ratio of oxygenated to deoxygenated hemoglobin, which changes when neurons fire. In **resting-state fMRI**, there is no task; participants lie still in the scanner. Even at rest, certain brain regions fluctuate in activity together, activating and deactivating in synchrony.

**Functional connectivity (FC)** is the statistical correlation between the BOLD time series of two brain regions. A correlation near +1 means the regions consistently co-activate (positive FC); near -1 means they consistently suppress each other (anti-correlation). By computing this for every pair of 200 brain regions, we produce a 200x200 **functional connectome** capturing the brain's large-scale coordination structure for each subject.

### What does the ASD connectivity literature say?

The most replicated finding in ASD neuroimaging is **long-range underconnectivity**: regions in the Default Mode Network and association cortices show weaker-than-typical synchronization in ASD [3, 4]. There is also evidence of **local overconnectivity** in sensorimotor regions [8]. These differences are real but subtle and highly variable across individuals, making ASD classification from fMRI a genuinely hard problem.

### Why model the connectome as a graph?

A 200x200 correlation matrix contains 19,900 unique pairwise connections. Using all of them as features for 303 subjects would give a model 66 times more features than samples -- a recipe for overfitting. A **graph** offers a structured alternative: 200 ROIs as nodes, connections above a threshold as edges, and compact node features summarizing each region's local connectivity properties. This reduces the raw feature space to something a classifier can actually learn from.

---

## Research Questions

**Primary:** Does ComBat site harmonization recover biological ASD classification signal that is masked by scanner-site variance in the unharmonized baseline, and what level of performance is achievable with harmonized data on this 3-site, 303-subject cohort?

**Secondary:** Which brain networks and connectivity properties drive the classification, and are the patterns consistent with the published ASD connectivity literature?

---

## Data

Resting-state fMRI data were obtained from the **ABIDE Preprocessed Connectomes Project (ABIDE-PCP)** [1].

| Field | Value |
|---|---|
| Source | ABIDE Preprocessed Connectomes Project (ABIDE-PCP) |
| Access | Fully public, no registration required |
| Sites | NYU, USM (University of Southern Mississippi), UCLA_1 |
| Preprocessing | C-PAC pipeline, band-pass filtered, no global signal regression |
| Parcellation | CC200 (Craddock et al., 2012) [2], 200 functionally defined ROIs |
| Functional networks | 8: DMN, Visual, SMN, DAN, VAN, FPN, Limbic, Subcortical |
| Subjects | 303 total: 154 ASD, 149 neurotypical controls |

> **Download:** `nilearn` fetches data automatically on first run (~500 MB, locally cached thereafter).

**What is a parcellation?** Rather than analyzing millions of individual brain voxels, we divide the brain into 200 regions called parcels. The **CC200 parcellation** [2] defines these regions by clustering voxels with similar temporal activity patterns in resting-state fMRI from healthy adults. Each parcel is treated as a single node in the graph.

**What are the 8 functional networks?** Large-scale networks are groups of regions that consistently co-activate. The 8 used here are: Default Mode Network (DMN), Visual, Sensorimotor (SMN), Dorsal Attention (DAN), Ventral Attention (VAN), Frontoparietal (FPN), Limbic, and Subcortical.

---

## Pipeline Overview

Six Python scripts run in sequence. Every output is fully reproducible from the automatic ABIDE download.

```
01_fetch_and_prepare.py
  Download ABIDE fMRI time series, compute 200x200 connectomes, apply Fisher z-transform

02_harmonize.py
  ComBat site harmonization -- remove scanner-site variance while preserving ASD biology

03_build_graphs.py
  Threshold harmonized matrices at |z| > 0.20, compute 5 node features per ROI

04_train_evaluate.py  [main result]
  PCA (50 components) + Gradient Boosting on full upper triangle -- 5-fold CV

05_gnn_train_evaluate.py
  PyTorch Geometric GCN on sparser graphs (|z| > 0.50) -- 5-fold CV

06_figures.py
  6 publication-quality figures from the harmonized pipeline
```

### Step-by-step

**Step 1: Data retrieval and connectome construction.** Downloads pre-processed ROI time series via `nilearn`, computes 200x200 Pearson correlation matrices, and applies the Fisher z-transform (r -> arctanh(r)) to produce approximately normally distributed connectivity values. Zero-variance ROIs are set to 0.

**Step 2: ComBat site harmonization.** Applies ComBat [11, 12] to remove systematic scanner-site variance before any classification. The upper triangle of each matrix (19,900 values) is treated as a feature vector. ComBat fits additive and multiplicative site effects using an empirical Bayes framework and subtracts them, leaving biological ASD-vs-Control variance intact. Critically, the diagnostic label is passed as a protected covariate; without this, ComBat removes biological signal along with site effects.

**What is ComBat?** ComBat was developed to correct batch effects in genomics microarray data -- the same problem occurs in multi-site neuroimaging, where scanners at different hospitals produce systematically different connectivity values unrelated to biology. In this dataset, USM had inflated mean FC (0.32 vs ~0.25 at NYU and UCLA) before harmonization. After ComBat, all three sites converge to ~0.265 and between-site variance drops to zero.

**Step 3: Graph construction.** Applies threshold |z| > 0.20 to harmonized matrices, retaining the strongest connections. Five features are computed per ROI from the resulting graph (see Node Features below).

**Step 4: PCA + Gradient Boosting.** Extracts the full upper triangle (19,900 values), applies PCA within each fold (50 components, ~62% variance explained), and trains a gradient-boosted classifier. PCA is fit on the training set only to prevent leakage. This is the primary result.

**Step 5: GCN.** Builds sparser graphs at |z| > 0.50 (~17% density vs 60% at 0.20) to give message passing meaningful local structure, then trains a two-layer GCNConv model with batch normalization and global mean pooling. See Results for an honest account of performance.

**Step 6: Figures.** Generates 6 figures from the harmonized pipeline outputs.

---

## Node Features

Five features are computed per ROI from the thresholded, harmonized graph:

| Feature | Description | Biological meaning |
|---|---|---|
| Mean FC | Average Fisher z to all other ROIs | Overall connectivity strength |
| Degree | Number of edges above threshold | Hubness; number of significant connections |
| Clustering coefficient | Fraction of a node's neighbors also connected to each other | Local cliquishness |
| Positive FC | Mean of positive correlations only | Co-activation profile |
| Negative FC | Mean of anti-correlations only | Competing connectivity |

---

## Results

### Gradient Boosting baseline (unharmonized)

5-fold stratified CV on raw connectomes without site correction:

| Metric | Mean | SD |
|---|---|---|
| Accuracy | 0.515 | 0.050 |
| AUC-ROC | 0.514 | 0.042 |
| Sensitivity (ASD) | 0.566 | 0.052 |
| Specificity (CTRL) | 0.462 | 0.087 |

AUC at chance. Site effects between NYU, USM, and UCLA dominate the connectivity signal, masking the biological ASD effect. The USM scanner produced inflated mean FC relative to the other sites, and without correction a classifier learns scanner fingerprints rather than disease biology.

### PCA + Gradient Boosting on ComBat-harmonized connectomes (primary result)

5-fold stratified CV after ComBat harmonization, PCA (50 components, ~62% variance explained), and gradient boosting:

| Metric | Mean | SD |
|---|---|---|
| Accuracy | 0.666 | 0.036 |
| AUC-ROC | 0.723 | 0.036 |
| Sensitivity (ASD) | 0.747 | 0.054 |
| Specificity (CTRL) | 0.583 | 0.121 |

ComBat removes scanner-site variance; PCA efficiently captures the global structure of the harmonized 19,900-dimensional connectivity space in 50 components. AUC 0.723 is consistent with published results on harmonized ABIDE data [5, 6].

### GCN (PyTorch Geometric)

5-fold CV with a 2-layer GCNConv model on sparser graphs (|z| > 0.50, ~17% density):

| Metric | Mean | SD |
|---|---|---|
| AUC-ROC | ~0.49-0.54 | high variance |

The GCN shows meaningful validation AUC during training (up to 0.72-0.82 in some folds) but does not generalize to the test set reliably. This is a known limitation of GCNs applied to dense brain connectivity graphs at this dataset scale: with only ~194 training subjects per fold and 5 node features, the model does not have enough signal to learn generalizable graph-level representations. At 60% graph density, message passing averages over ~120 neighbors per node, collapsing to a global mean after 2 layers. Even at 17% density (~6,900 edges), the 5-dimensional node features are insufficient to discriminate ASD from Control through neighborhood aggregation alone. The gradient boosting + PCA approach works because PCA directly captures global connectivity structure from all 19,900 pairwise connections simultaneously.

Published GCN results achieving AUC 0.72-0.78 on ABIDE use either much larger multi-site cohorts [6] or more informative per-node features derived from the full time series rather than graph-derived statistics [7]. This is the primary future direction for this project.

---

## Figures

### Figure 1: Resting-State Functional Connectivity Matrices

[![Functional connectivity matrices](https://github.com/anirudhramadurai/abide-connectome-asd/raw/main/figures/fig1_connectivity_matrices.png)](https://github.com/anirudhramadurai/abide-connectome-asd/raw/main/figures/fig1_connectivity_matrices.png)

ComBat-harmonized Fisher z-transformed correlation matrices for a representative Control (left) and ASD subject (right). Warm red = positive FC; cool blue = anti-correlation; white/neutral = near-zero. The white grid lines divide the 200 ROIs into 8 functional networks. The ASD subject shows more uniformly warm coloring -- a consequence of higher overall mean FC in this particular subject, visible across most network blocks.

---

### Figure 2: Brain Graph Structure

[![Brain graph structure](https://github.com/anirudhramadurai/abide-connectome-asd/raw/main/figures/fig2_graph_structure.png)](https://github.com/anirudhramadurai/abide-connectome-asd/raw/main/figures/fig2_graph_structure.png)

All 200 CC200 ROIs arranged in a circle, colored by functional network. The top 300 edges by absolute FC strength are drawn: red = positive FC, blue = anti-correlation. The ASD subject (right, 33,946 edges) has more than twice the edges above threshold as the Control (left, 15,558 edges). After harmonization this difference reflects individual biological variation -- not the scanner-site artifact that drove the 0.51 unharmonized AUC.

---

### Figure 3: Node Feature Distributions by Group

[![Node feature distributions](https://github.com/anirudhramadurai/abide-connectome-asd/raw/main/figures/fig3_feature_distributions.png)](https://github.com/anirudhramadurai/abide-connectome-asd/raw/main/figures/fig3_feature_distributions.png)

Violin plots comparing the 5 node features between Control (blue) and ASD (orange) across all subjects and ROIs (149 x 200 = 29,800 control observations; 154 x 200 = 30,800 ASD observations). The white bar marks the median. All five features show substantial overlap between groups, explaining why individual ROI-level features are insufficient for classification -- the discriminative signal is in how features vary across the network structure, not in any single feature's marginal distribution.

---

### Figure 4: Per-Fold Classification Performance

[![Per-fold performance](https://github.com/anirudhramadurai/abide-connectome-asd/raw/main/figures/fig4_performance.png)](https://github.com/anirudhramadurai/abide-connectome-asd/raw/main/figures/fig4_performance.png)

AUC-ROC (blue) and Accuracy (orange) for each of the 5 cross-validation folds after ComBat harmonization. All five folds are well above chance. Fold 1 achieves the best AUC (0.77) and Fold 5 the lowest (0.67), a range of 0.10 -- substantially tighter than the unharmonized baseline, where fold-to-fold variance reflected which sites happened to fall in the test set. The consistent performance across folds is the main indicator that harmonization succeeded in removing site confounding.

---

### Figure 5: ROC Curves

[![ROC curves](https://github.com/anirudhramadurai/abide-connectome-asd/raw/main/figures/fig5_roc_curves.png)](https://github.com/anirudhramadurai/abide-connectome-asd/raw/main/figures/fig5_roc_curves.png)

ROC curves for each of the 5 folds plus the interpolated mean with +/- 1 SD band. The mean AUC of 0.723 +/- 0.036 summarizes overall classification performance. All 5 fold curves track clearly above the diagonal chance line, confirming that the classifier is learning biological signal rather than site confounds. The individual fold spread (0.67-0.77) reflects genuine within-ABIDE heterogeneity in ASD presentation across subjects.

---

### Figure 6: Node Importance by Network and Feature

[![Node importance heatmap](https://github.com/anirudhramadurai/abide-connectome-asd/raw/main/figures/fig6_node_importance.png)](https://github.com/anirudhramadurai/abide-connectome-asd/raw/main/figures/fig6_node_importance.png)

Mean absolute difference between ASD and Control node features, aggregated by functional network and averaged across 5 held-out folds. Values are normalized 0-1 per feature row. Darker red = larger ASD-Control difference.

**Limbic system** shows the highest importance for Mean FC (1.00), Degree (1.00), and Positive FC (1.00) -- the strongest pattern in the heatmap. The limbic system encompasses the amygdala, hippocampus, and cingulate cortex, which are central to emotional processing and social cognition -- two domains characteristically affected in ASD.

**Frontoparietal Network (FPN)** shows the highest importance for Clustering Coefficient (1.00) and Negative FC (1.00), and high Degree (0.44). The FPN is a key network for executive function, working memory, and flexible reasoning. The FPN anti-correlation pattern is particularly interesting: negative FC (anti-correlated connectivity) between FPN regions shows the largest ASD-Control difference of any feature-network combination, consistent with disrupted executive network dynamics [9].

**Sensorimotor Network (SMN)** shows high Mean FC (0.82) and Positive FC (0.62), consistent with sensory processing atypicalities and altered sensorimotor integration in ASD [8].

**Default Mode Network (DMN)** and **Visual cortex** show near-zero importance across most features. Near-zero visual cortex importance is consistent with the literature: visual processing is largely preserved at the network level in ASD [1].

**Dorsal Attention Network (DAN)** shows high Negative FC importance (0.98), second only to FPN. Anti-correlations between attention and default mode regions are a fundamental feature of healthy brain organization; disruption of this pattern in ASD has been reported previously.

---

## Summary of Findings

| Question | Finding |
|---|---|
| Does unharmonized multi-site classification work? | No. AUC = 0.51. Scanner variance at USM inflates FC and dominates the signal [11]. |
| Does ComBat harmonization recover signal? | Yes. AUC improves from 0.514 to 0.723 after removing site effects [11, 12]. |
| Does a GCN outperform gradient boosting here? | No. GCN learns during training but does not generalize at 303 subjects with 5 node features. |
| Which networks show the largest ASD-Control differences? | Limbic (social cognition), FPN (executive function), and SMN (sensorimotor) [8, 9]. |
| Are node-level features sufficient alone? | No. Individual ROI distributions overlap substantially; network-level structure is necessary. |

---

## Setup and Usage

**Requirements:** Python 3.9+

```bash
# 1. Clone the repository
git clone https://github.com/anirudhramadurai/abide-connectome-asd.git
cd abide-connectome-asd

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
chmod +x run_all.sh
./run_all.sh
```

Or run each step individually:

```bash
python scripts/01_fetch_and_prepare.py   # downloads ABIDE (~500 MB, cached after first run)
python scripts/02_harmonize.py           # ComBat site harmonization (~1 min)
python scripts/03_build_graphs.py        # graph construction
python scripts/04_train_evaluate.py      # PCA + gradient boosting (~2 min)
python scripts/05_gnn_train_evaluate.py  # GCN training (~15 min, CPU)
python scripts/06_figures.py             # generate all figures
```

Figures are saved to `figures/`. Data files to `data/`. Results to `results/`.

---

## A note on `models/gcn_numpy.py`

This file contains a from-scratch NumPy implementation of the graph convolution mathematics from Kipf & Welling (2017) [13]. It is **not** the production classifier used in the pipeline.

It exists as an educational resource to make the GCN forward pass transparent: symmetric normalized adjacency (D^-1/2 A D^-1/2), two-layer propagation, global mean pooling. The backward pass is analytically derived and numerically unstable on real fMRI data at this graph density without automatic differentiation. The actual GCN in `05_gnn_train_evaluate.py` uses PyTorch Geometric with autograd.

---

## Limitations and Future Directions

**Current limitations:**

- Sample size (n = 303 across 3 sites) is on the small end for deep learning approaches; gradient boosting + PCA outperforms GCN at this scale
- The 5-dimensional node feature set (mean FC, degree, clustering, pos FC, neg FC) is derived entirely from thresholded graph statistics and does not capture the rich temporal structure of the original BOLD time series
- Network assignments in the CC200 parcellation are approximate; ROI-to-network mappings are based on Power et al. (2011) applied to CC200 ordering
- Results are correlational; no causal inference is possible from cross-sectional observational data
- The gradient boosting pipeline uses the full connectome upper triangle as input to PCA; this does not leverage the graph structure that a GCN is designed to exploit

**Future directions:**

- Scale to the full ABIDE cohort (~1,100 subjects across 17 sites) with leave-site-out cross-validation
- Implement a PyTorch Geometric GCN with time-series-derived node features (regional variance, autocorrelation, spectral power) which provide richer input than graph statistics alone [6, 7]
- Apply graph attention networks (GAT) to learn which edges are most informative rather than using a fixed threshold
- Test higher-resolution parcellations (Schaefer-400, Gordon-333) to capture finer-grained connectivity patterns
- Incorporate demographic covariates (age, sex, IQ) to account for biological confounds

---

## References

1. Di Martino A, Yan C-G, Li Q, et al. The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism. *Molecular Psychiatry*. 2014;19(6):659-667. doi:10.1038/mp.2013.78. PMID: 24514918.

2. Craddock RC, James GA, Holtzheimer PE, Hu XP, Mayberg HS. A whole brain fMRI atlas generated via spatially constrained spectral clustering. *Human Brain Mapping*. 2012;33(8):1914-1928. doi:10.1002/hbm.21333.

3. Just MA, Cherkassky VL, Keller TA, Minshew NJ. Cortical activation and synchronization during sentence comprehension in high-functioning autism: evidence of underconnectivity. *Brain*. 2004;127(8):1811-1821. doi:10.1093/brain/awh199.

4. Assaf M, Jagannathan K, Calhoun VD, et al. Abnormal functional connectivity of default mode sub-networks in autism spectrum disorder patients. *NeuroImage*. 2010;53(1):247-256. doi:10.1016/j.neuroimage.2010.05.067.

5. Ktena SI, Parisot S, Ferrante E, et al. Metric learning with spectral graph convolutions on brain connectivity networks. *NeuroImage*. 2018;169:431-442. doi:10.1016/j.neuroimage.2017.12.052.

6. Li X, Zhou Y, Dvornek N, et al. BrainGNN: Interpretable brain graph neural network for fMRI analysis. *Medical Image Analysis*. 2021;74:102233. doi:10.1016/j.media.2021.102233.

7. Jiang H, Cao P, Xu M, Yang J, Zaiane O. Hi-GCN: A hierarchical graph convolution network for graph embedding learning of brain network. *Computers in Biology and Medicine*. 2020;127:104096. doi:10.1016/j.compbiomed.2020.104096.

8. Marco EJ, Hinkley LBN, Hill SS, Nagarajan SS. Sensory processing in autism: a review of neurophysiologic findings. *Pediatric Research*. 2011;69(5 Pt 2):48R-54R. doi:10.1203/PDR.0b013e3182130c54.

9. Yerys BE, Gordon EM, Abrams DN, et al. Default mode network segregation and social deficits in autism spectrum disorder. *NeuroImage: Clinical*. 2015;9:223-232. doi:10.1016/j.nicl.2015.07.018.

10. Bullmore E, Sporns O. Complex brain networks: graph theoretical analysis of structural and functional systems. *Nature Reviews Neuroscience*. 2009;10(3):186-198. doi:10.1038/nrn2575.

11. Johnson WE, Li C, Rabinovic A. Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*. 2007;8(1):118-127. doi:10.1093/biostatistics/kxj037.

12. Fortin JP, Parker D, Tunc B, et al. Harmonization of multi-site diffusion tensor imaging data. *NeuroImage*. 2017;161:149-170. doi:10.1016/j.neuroimage.2017.08.047.

13. Kipf TN, Welling M. Semi-supervised classification with graph convolutional networks. *ICLR 2017*. arXiv:1609.02907.

---

## Acknowledgements

Developed as an independent computational neuroscience project. Data from the ABIDE Preprocessed Connectomes Project, accessed via nilearn. ABIDE was supported by grants from the Autism Speaks Foundation and the National Institute of Mental Health.
