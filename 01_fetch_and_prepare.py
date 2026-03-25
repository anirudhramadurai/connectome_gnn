"""
01_fetch_and_prepare.py
-----------------------
Downloads real resting-state fMRI connectomes from the ABIDE dataset
(Di Martino et al., Mol Psychiatry 2014) using the nilearn library.

Data source : ABIDE Preprocessed Connectomes Project (ABIDE-PCP)
Parcellation: CC200 (Craddock et al., 2012) — 200 functionally-defined ROIs
Pipeline    : C-PAC, band-pass filtered, global signal not regressed (filt_noglobal)
Access      : Fully public, no registration required

What this script does:
  1. Downloads pre-computed ROI time series for ASD and control subjects
     from 3 well-characterised acquisition sites (NYU, USM, UCLA).
  2. Computes a 200×200 Pearson correlation matrix (functional connectome)
     for every subject from their ROI time series.
  3. Fisher z-transforms all correlations for normality.
  4. Saves the connectomes, labels, and subject metadata to data/.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from nilearn import datasets

# ── Configuration ────────────────────────────────────────────────────────────
# Three large, well-characterised ABIDE sites
SITES       = ["NYU", "USM", "UCLA_1"]
# Maximum subjects per site (set None for all available)
MAX_PER_SITE = None
DATA_DIR    = Path("data")
DATA_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("  ABIDE Connectome Download")
print("  Sites:", ", ".join(SITES))
print("=" * 60)

# ── Download ─────────────────────────────────────────────────────────────────
print("\nFetching ABIDE phenotypic data and ROI time series …")
print("(First run will download ~500 MB — subsequent runs use cache)\n")

abide = datasets.fetch_abide_pcp(
    derivatives   = ["rois_cc200"],
    SITE_ID       = SITES,
    n_subjects    = MAX_PER_SITE,
    pipeline      = "cpac",
    band_pass_filtering = True,
    global_signal_regression = False,
    verbose       = 1,
)

pheno    = abide.phenotypic
ts_files = abide.rois_cc200   # list of paths to .1D time-series files

print(f"\nDownloaded {len(ts_files)} subjects")
print(f"Diagnosis breakdown:\n{pd.Series(pheno['DX_GROUP']).value_counts().to_string()}")
# DX_GROUP: 1 = ASD, 2 = Control

# ── Compute connectomes ───────────────────────────────────────────────────────
print("\nComputing functional connectivity matrices …")

connectomes, labels, subject_ids, sites = [], [], [], []
skipped = 0

for i, ts_path in enumerate(ts_files):
    try:
        if isinstance(ts_path, np.ndarray):
            ts = ts_path
        else:
            ts = np.loadtxt(ts_path)   # shape: (timepoints, 200)
    except Exception as e:
        print(f"  Skipping subject {i}: {e}")
        skipped += 1
        continue

    # Need at least 50 timepoints for a reliable correlation estimate
    if ts.shape[0] < 50:
        print(f"  Skipping subject {i}: only {ts.shape[0]} timepoints")
        skipped += 1
        continue

    # 200×200 Pearson correlation matrix
    corr = np.corrcoef(ts.T)

    # Fisher z-transform: r → atanh(r) for approximate normality
    # Clip slightly inside [-1,1] to avoid atanh(±1) = ±inf
    corr_z = np.arctanh(np.clip(corr, -0.999, 0.999))
    np.fill_diagonal(corr_z, 0.0)   # zero diagonal (self-correlation undefined)
    corr_z = np.nan_to_num(corr_z, nan=0.0, posinf=0.0, neginf=0.0)

    connectomes.append(corr_z)
    # ABIDE DX_GROUP: 1 = ASD → map to 1; 2 = Control → map to 0
    labels.append(1 if pheno["DX_GROUP"].iloc[i] == 1 else 0)
    subject_ids.append(str(pheno["SUB_ID"].iloc[i]))
    sites.append(str(pheno["SITE_ID"].iloc[i]))

print(f"  Processed : {len(connectomes)} subjects")
print(f"  Skipped   : {skipped} (bad files / short scans)")

connectomes = np.array(connectomes)   # (N, 200, 200)
labels      = np.array(labels)        # (N,)  1=ASD, 0=CTRL

n_asd  = labels.sum()
n_ctrl = (labels == 0).sum()
print(f"\n  ASD      : {n_asd}")
print(f"  Controls : {n_ctrl}")
print(f"  Total    : {len(labels)}")

# ── ROI metadata from CC200 atlas ────────────────────────────────────────────
# Broad network assignments for CC200 ROIs (approximate, based on Power et al. 2011)
# We assign each ROI a functional network label for downstream visualisation
NETWORK_MAP = {
    "DMN":         range(0,   28),
    "Visual":      range(28,  52),
    "SMN":         range(52,  74),
    "DAN":         range(74,  94),
    "VAN":         range(94, 112),
    "FPN":         range(112, 134),
    "Limbic":      range(134, 150),
    "Subcortical": range(150, 200),
}
networks = [""] * 200
roi_names = []
for net, rng in NETWORK_MAP.items():
    for r in rng:
        networks[r] = net
        roi_names.append(f"{net}_{r:03d}")

# ── Save ─────────────────────────────────────────────────────────────────────
np.save(DATA_DIR / "connectomes.npy", connectomes)
np.save(DATA_DIR / "labels.npy",      labels)

metadata = pd.DataFrame({
    "subject_id": subject_ids,
    "site":       sites,
    "label":      labels,
    "group":      ["ASD" if l == 1 else "CTRL" for l in labels],
})
metadata.to_csv(DATA_DIR / "metadata.csv", index=False)

with open(DATA_DIR / "roi_meta.pkl", "wb") as f:
    pickle.dump({"roi_names": roi_names, "networks": networks}, f)

print(f"\nSaved to data/")
print(f"  connectomes.npy  — shape {connectomes.shape}, dtype {connectomes.dtype}")
print(f"  labels.npy       — {len(labels)} subjects")
print(f"  metadata.csv     — subject IDs, sites, diagnoses")
print(f"  roi_meta.pkl     — ROI names and network assignments")
print("\nReady for 02_build_graphs.py")
