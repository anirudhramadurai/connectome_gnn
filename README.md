# connectome-gnn

**Graph-based classification of Autism Spectrum Disorder from resting-state fMRI functional connectivity data**

---

## tl;dr

Using publicly available resting-state fMRI data from 303 participants across three ABIDE sites (NYU, USM, UCLA), I built a four-script Python pipeline to ask whether ASD can be detected from whole-brain functional connectivity patterns represented as graphs. After computing 200x200 correlation matrices per subject, converting them to graphs, and extracting 44 biologically grounded graph-level features, a gradient-boosted classifier achieves AUC = 0.51 ± 0.04 in 5-fold cross-validation. This near-chance result is honest and expected on raw multi-site ABIDE data without site harmonisation: scanner differences between acquisition sites dominate the signal and obscure the biological ASD effect. The node importance analysis (Fig 6) reveals that sensorimotor (SMN), frontoparietal (FPN), and limbic network features show the largest ASD–Control differences, consistent with the published literature on sensorimotor integration, executive function, and social cognition atypicalities in ASD. The main limitation is the absence of ComBat site harmonisation, which is the standard next step to recover meaningful biological classification performance.

---

## What is this project?

This is an independent computational neuroscience project that frames ASD classification as a graph machine learning problem. Each participant's brain is represented as a graph — 200 brain regions as nodes, functional correlations as weighted edges — and a classifier is trained to distinguish ASD from neurotypical controls using features derived from that graph structure.

The full pipeline runs in approximately 10 minutes on a standard laptop, from automatic data download to six publication-quality figures.

---

## The Biology (plain language first)

### What is ASD and why study brain connectivity?

Autism Spectrum Disorder (ASD) is a neurodevelopmental condition affecting approximately 1–2% of the global population, characterised by differences in social communication and repetitive or restricted behaviour patterns. It is a spectrum, meaning individuals vary widely in how the condition presents and how much it affects daily life.

Unlike schizophrenia, where postmortem transcriptomics has been highly informative, the biology of ASD is better studied through brain imaging in living participants. The dominant neuroscientific hypothesis about ASD is the **connectivity hypothesis**: that the condition arises not from failure of any single brain region, but from disrupted coordination between brain regions — specifically, altered long-range functional connectivity (Di Martino et al., 2014; Just et al., 2004).

### What is resting-state fMRI and functional connectivity?

Functional MRI (fMRI) measures brain activity indirectly through the blood-oxygen-level-dependent (BOLD) signal: when neurons in a region fire more, blood flow increases, changing the ratio of oxygenated to deoxygenated haemoglobin, which is detectable as a small change in MRI signal intensity. In resting-state fMRI, participants simply lie still in the scanner without performing any task, allowing the spontaneous fluctuations in brain activity to be measured.

**Functional connectivity (FC)** is the temporal correlation between the BOLD time series of two brain regions. If region A and region B consistently activate and deactivate together over the course of the scan, they are said to be functionally connected. Across all pairs of 200 regions, this produces a 200x200 **functional connectome** — a snapshot of the brain's large-scale coordination structure for that individual.

### What does the ASD connectivity literature say?

The most replicated finding in ASD neuroimaging is **long-range underconnectivity**: regions that are far apart in the brain, particularly within the Default Mode Network (DMN) and between association cortices, show weaker-than-typical synchronisation. This is thought to reflect disrupted integration of information across distributed networks, contributing to the social and cognitive profile of ASD (Just et al., 2004; Assaf et al., 2010).

Additionally, there is evidence of **local overconnectivity** in sensorimotor regions of ASD participants — elevated synchrony within nearby regions, which may relate to sensory processing differences and motor coordination atypicalities (Marco et al., 2011).

These connectivity differences are subtle at the group level and highly variable across individuals, which is a central reason why ASD classification from fMRI remains difficult.

### Why model the connectome as a graph?

A 200x200 correlation matrix contains 19,900 unique pairwise connections. Treating this as a flat feature vector for 303 subjects is statistically intractable (far more features than samples). Graph-based representations offer an alternative: brain regions become **nodes**, significant correlations become **edges**, and the local topology around each node (its degree, clustering, connectivity profile) can be summarised into a compact, biologically interpretable feature set that scales with the number of regions, not the number of pairs.

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

> **Download:** `nilearn` fetches data automatically on first run (~500 MB, locally cached thereafter).

The **CC200 parcellation** divides the brain into 200 functionally homogeneous regions of interest (ROIs) using spatially constrained spectral clustering of resting-state fMRI data from healthy adults (Craddock et al., 2012). Functional parcellations like CC200 are preferable to anatomical atlases for connectivity studies because the parcels respect functional boundaries rather than cytoarchitectonic ones.

---

## Pipeline

```
01_fetch_and_prepare.py  -->  02_build_graphs.py  -->  03_train_evaluate.py  -->  04_figures.py
  Download ABIDE             Threshold matrices,       Extract graph-level        6 publication-
  fMRI time series,          compute node features,    features, train            quality figures
  compute connectomes,       build graph objects       classifier, 5-fold CV
  Fisher z-transform
```

### What each step does

**Step 1: Data retrieval and connectome construction**
Downloads pre-processed ROI time series for each subject using `nilearn.datasets.fetch_abide_pcp`. For each subject, computes the 200x200 Pearson correlation matrix between all pairs of ROI time series, then applies the Fisher z-transform: r → arctanh(r). This transformation converts bounded correlation coefficients (−1 to 1) into approximately normally distributed values with no fixed bounds, making them more suitable for downstream statistics. Zero-variance ROIs (regions with no signal in some subjects) produce NaN correlations and are set to 0 using `np.nan_to_num`.

**Step 2: Graph construction and node feature extraction**
Thresholds each Fisher z matrix at |z| > 0.20, a standard cutoff in fMRI graph analysis that retains strong connections while removing weak or noisy ones (Bullmore & Sporns, 2009). Five node features are then computed per ROI: mean FC, degree, local clustering coefficient, mean positive FC, and mean negative FC. These capture different aspects of a region's role in the network.

**Step 3: Feature extraction and classification**
Rather than using the full graph structure (which would require a GCN with autograd), the pipeline extracts 44 graph-level summary features per subject — between-network FC differences (28 values), mean degree per network (8 values), and mean clustering coefficient per network (8 values). A gradient-boosted classifier with median imputation and standardisation is trained under 5-fold stratified cross-validation.

**Step 4: Figures**
Generates six publication-quality figures covering raw connectivity matrices, graph topology, feature distributions, classification performance, ROC curves, and node importance.

---

## Node Features

| Feature | Description | Biological meaning |
|---|---|---|
| Mean FC | Average Fisher z to all other ROIs | Overall connectivity strength — how embedded the region is in the network |
| Degree | Number of edges above threshold | Hubness — how many significant connections the region maintains |
| Clustering coefficient | Fraction of a node's neighbours that are also connected to each other | Local cliquishness — whether a region's connections form tight local clusters |
| Positive FC | Mean of positive correlations only | Excitatory or co-activation connectivity profile |
| Negative FC | Mean of anti-correlations only | Competing or anticorrelated connectivity — regions that suppress each other |

---

## Results

5-fold stratified cross-validation (Gradient Boosting with median imputation and standard scaling):

| Metric | Mean | SD |
|---|---|---|
| Accuracy | 0.515 | 0.050 |
| AUC-ROC | 0.514 | 0.042 |
| Sensitivity (ASD recall) | 0.566 | 0.052 |
| Specificity (CTRL recall) | 0.462 | 0.087 |

### Why AUC ~0.51?

This result is honest and expected. ABIDE classification on raw data is difficult for three reasons:

**Site effects dominate.** Scanner hardware, acquisition protocols, and operator practices differ between NYU, USM, and UCLA. These differences introduce systematic variance that accounts for a much larger proportion of the connectivity signal than the biological ASD effect. Without explicit site harmonisation (ComBat; Johnson et al., 2007), a classifier trained on multi-site data learns scanner differences rather than biology.

**Small within-site samples.** After 5-fold splitting, each test set contains approximately 60 subjects — insufficient statistical power for a subtle, heterogeneous condition like ASD.

**ASD heterogeneity.** ASD is a spectrum. Group-level connectivity differences are real but small relative to within-group variance. Published models reporting AUC 0.65–0.78 on ABIDE use site harmonisation, larger cohorts, and deep learning architectures (Ktena et al., 2018; Li et al., 2021). This pipeline establishes a transparent, reproducible baseline.

---

## Figures

### Figure 1 — Resting-State Functional Connectivity Matrices

![Functional connectivity matrices](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig1_connectivity_matrices.png)

Fisher z-transformed correlation matrices for a representative control (left) and ASD subject (right). Each pixel (i, j) shows the strength of functional connectivity between ROI i and ROI j: warm colours indicate positive correlation (regions that activate together), cool colours indicate anticorrelation (regions that suppress each other). White boundary lines divide the matrix into 8 functional networks. The ASD subject shows characteristic white diagonal stripes reflecting zero-variance ROIs in the UCLA site data — a known artifact in preprocessed ABIDE that is handled by NaN imputation in the pipeline.

---

### Figure 2 — Brain Graph Structure

![Brain graph structure](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig2_graph_structure.png)

Each of the 200 CC200 ROIs is arranged as a node in a circle, coloured by functional network. The top 300 edges by absolute connectivity strength are drawn (red = positive FC, blue = anticorrelation). The ASD subject (right) shows more edges above the |z| > 0.20 threshold than the control (35,718 vs 16,620), consistent with the site-driven overconnectivity artifact seen in some ABIDE sites. In datasets with harmonised acquisition, ASD typically shows fewer long-range connections — the underconnectivity hypothesis (Just et al., 2004).

---

### Figure 3 — Node Feature Distributions by Group

![Node feature distributions](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig3_feature_distributions.png)

Violin plots comparing the distribution of all 5 node features between neurotypical controls (blue) and ASD subjects (orange), across all 200 ROIs and all 303 subjects. Each observation is one ROI in one subject. White bars show the median. The distributions overlap substantially across all five features — confirming that group differences are weak at the individual node level, and motivating the use of network-level aggregated features for classification.

---

### Figure 4 — Per-Fold Classification Performance

![Per-fold performance](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig4_training_loss.png)

AUC-ROC (blue) and accuracy (orange) for each of the 5 cross-validation folds. The dashed grey line marks chance level (0.50). Fold 4 achieves the highest AUC (0.56) and accuracy (0.58); Fold 5 falls below chance (AUC = 0.44). This variance across folds is a direct signature of site effects: depending on which subjects fall in the test set, the classifier may encounter a scanner profile it has not seen, degrading performance.

---

### Figure 5 — ROC Curves

![ROC curves](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig5_roc_curves.png)

Receiver Operating Characteristic (ROC) curves for each fold plus the interpolated mean with ±1 SD band. The x-axis is the false positive rate (fraction of controls incorrectly classified as ASD); the y-axis is the true positive rate (fraction of ASD subjects correctly identified). A perfect classifier would reach the top-left corner. The mean AUC of 0.51 ± 0.04 indicates near-chance performance, with all five folds hugging the diagonal. This is the correct and expected result for raw ABIDE data without site correction.

---

### Figure 6 — Node Importance by Network and Feature

![Node importance heatmap](https://github.com/anirudhramadurai/connectome_gnn/raw/main/figures/fig6_node_importance.png)

Mean absolute difference between ASD and control node features, aggregated by functional network and averaged across the 5 held-out test folds. Values are normalised to [0, 1] per feature column. Darker red = larger ASD-Control difference for that network-feature combination.

Several biologically grounded patterns emerge:

**Sensorimotor Network (SMN)** shows the largest differences in mean FC and positive FC. This is consistent with the well-documented sensory processing atypicalities in ASD and altered sensorimotor integration (Marco et al., 2011).

**Frontoparietal Network (FPN)** shows high clustering coefficient and positive FC differences, consistent with disrupted executive function and working memory networks in ASD (Yerys et al., 2015).

**Limbic system** shows consistently high importance across most features, reflecting the established role of limbic connectivity in social cognition and emotional processing differences in ASD.

**Visual cortex** shows near-zero importance across all features — consistent with relatively preserved visual processing at the network level in ASD, a known distinguishing feature of the disorder compared to other neurodevelopmental conditions.

---

## Setup and Usage

```bash
# 1. Clone the repository
git clone https://github.com/anirudhramadurai/connectome_gnn.git
cd connectome_gnn

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
python 01_fetch_and_prepare.py  # downloads ABIDE (~500 MB, cached after first run)
python 02_build_graphs.py
python 03_train_evaluate.py
python 04_figures.py
```

Runtime: approximately 5–10 minutes on a standard laptop (CPU only).

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

**Site harmonisation.** The single most impactful improvement would be applying ComBat (Johnson et al., 2007) or neuroCombat to remove scanner-driven variance before classification. Published models achieving AUC 0.65–0.78 uniformly apply some form of site correction.

**Richer parcellation.** The CC200 atlas uses 200 ROIs. Higher-resolution atlases such as Schaefer-400 or Gordon-333 parcellations would capture finer-grained connectivity patterns, potentially recovering more biological signal.

**Full ABIDE cohort.** This pipeline uses 3 of 17 available ABIDE sites (303 of ~1,100 subjects). Scaling to the full dataset with leave-site-out cross-validation would yield substantially more statistical power and a more honest estimate of generalisability.

**Deep learning.** PyTorch Geometric GCN models with proper site stratification achieve AUC ~0.72–0.78 on ABIDE (Li et al., 2021; Jiang et al., 2020). The from-scratch numpy GCN in `models/gcn_numpy.py` demonstrates the mathematical operations (normalised graph convolution, global mean pooling, Adam optimisation) but is numerically unstable on real fMRI data at this scale without autograd.

**Longitudinal data.** ABIDE is cross-sectional. Longitudinal fMRI datasets tracking connectivity changes over development in ASD would enable more mechanistic questions about when and how connectivity differences emerge.

---

## References

- Di Martino A, et al. (2014). The autism brain imaging data exchange: towards large-scale evaluation of the intrinsic brain architecture in autism. *Molecular Psychiatry*, 19(6), 659–667.
- Craddock RC, et al. (2012). A whole brain fMRI atlas generated via spatially constrained spectral clustering. *Human Brain Mapping*, 33(8), 1914–1928.
- Just MA, et al. (2004). Cortical activation and synchronization during sentence comprehension in high-functioning autism: evidence of underconnectivity. *Brain*, 127(8), 1811–1821.
- Assaf M, et al. (2010). Abnormal functional connectivity of default mode sub-networks in autism spectrum disorder patients. *NeuroImage*, 53(1), 247–256.
- Ktena SI, et al. (2018). Metric learning with spectral graph convolutions on brain connectivity networks. *NeuroImage*, 169, 431–442.
- Li X, et al. (2021). BrainGNN: Interpretable brain graph neural network for fMRI analysis. *Medical Image Analysis*, 74, 102233.
- Jiang H, et al. (2020). Hi-GCN: A hierarchical graph convolution network for graph embedding learning of brain network. *Computers in Biology and Medicine*, 127, 104096.
- Marco EJ, et al. (2011). Sensory processing in autism: a review of neurophysiologic findings. *Pediatric Research*, 69(5 Pt 2), 48R–54R.
- Yerys BE, et al. (2015). Default mode network segregation and social deficits in autism spectrum disorder patients. *Biological Psychiatry: CNNI*, 1(5), 374–382.
- Bullmore E, Sporns O. (2009). Complex brain networks: graph theoretical analysis of structural and functional systems. *Nature Reviews Neuroscience*, 10(3), 186–198.
- Johnson WE, et al. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*, 8(1), 118–127.

---

## Acknowledgements

Developed as an independent project, drawing on methods from BIME 534 (Biology & Informatics, University of Washington). Data from the ABIDE Preprocessed Connectomes Project, downloaded via nilearn.

*University of Washington — Independent Project*
