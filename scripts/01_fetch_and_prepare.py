"""
01_fetch_and_prepare.py
-----------------------
Downloads resting-state fMRI connectomes from the ABIDE dataset and
computes functional connectivity matrices for each subject.

Dataset   : ABIDE Preprocessed Connectomes Project (ABIDE-PCP)
            Di Martino A, et al. Mol Psychiatry (2014). PMID: 24514918
Parcellation: CC200 -- 200 functionally-defined ROIs
              Craddock RC, et al. Hum Brain Mapp (2012). PMID: 21769991
Pipeline  : C-PAC, band-pass filtered, no global signal regression
Access    : Fully public, no registration required

What this script does
---------------------
1. Downloads pre-computed ROI time series for ASD and control subjects
   from three well-characterised acquisition sites (NYU, USM, UCLA_1)
   using nilearn. Data is cached locally after the first run.
2. Computes a 200x200 Pearson correlation matrix (functional connectome)
   for each subject from their ROI time series.
3. Applies the Fisher z-transform (r -> arctanh(r)) to convert bounded
   correlation coefficients into approximately normally distributed values
   suitable for downstream statistical analysis.
4. Saves connectomes, diagnostic labels, subject metadata, and ROI
   network assignments to data/.

Note on nilearn API
-------------------
nilearn >= 0.10 returns numpy arrays directly from fetch_abide_pcp rather
than file paths. This script checks for both cases and handles them
transparently.

Usage
-----
  python scripts/01_fetch_and_prepare.py

Outputs
-------
  data/connectomes.npy   (N, 200, 200) float64 Fisher z matrices
  data/labels.npy        (N,) int array -- 1 = ASD, 0 = Control
  data/metadata.csv      Subject IDs, sites, diagnoses
  data/roi_meta.pkl      ROI names and network assignments

References
----------
Di Martino A, et al. (2014). The autism brain imaging data exchange:
  towards a large-scale evaluation of the intrinsic brain architecture
  in autism. Mol Psychiatry, 19(6):659-667. doi:10.1038/mp.2013.78.

Craddock RC, et al. (2012). A whole brain fMRI atlas generated via
  spatially constrained spectral clustering. Hum Brain Mapp,
  33(8):1914-1928. doi:10.1002/hbm.21333.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from nilearn import datasets

# -- Configuration ------------------------------------------------------------

SITES        = ["NYU", "USM", "UCLA_1"]   # three large, well-characterised sites
MAX_PER_SITE = None                        # None = all available subjects
MIN_TIMEPOINTS = 50                        # minimum scan length for reliable FC

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Network assignments for CC200 ROIs (approximate, based on Power et al. 2011)
# Each ROI is assigned to one of 8 canonical functional networks
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


# -- Download -----------------------------------------------------------------

def download_abide():
    """
    Download ABIDE ROI time series and phenotypic data via nilearn.
    Data is cached locally and reused on subsequent runs.

    Returns
    -------
    pheno    : pandas DataFrame of phenotypic/demographic information
    ts_files : list of time-series arrays or file paths, one per subject
    """
    print("Fetching ABIDE phenotypic data and ROI time series ...")
    print("(First run will download ~500 MB -- subsequent runs use cache)\n")

    abide = datasets.fetch_abide_pcp(
        derivatives              = ["rois_cc200"],
        SITE_ID                  = SITES,
        n_subjects               = MAX_PER_SITE,
        pipeline                 = "cpac",
        band_pass_filtering      = True,
        global_signal_regression = False,
        verbose                  = 1,
    )

    pheno    = abide.phenotypic
    ts_files = abide.rois_cc200

    print(f"\nDownloaded {len(ts_files)} subjects")
    print("Diagnosis breakdown:")
    print(pd.Series(pheno["DX_GROUP"]).value_counts().to_string())
    # DX_GROUP coding: 1 = ASD, 2 = Control

    return pheno, ts_files


# -- Compute connectomes ------------------------------------------------------

def compute_connectomes(pheno, ts_files):
    """
    Compute Fisher z-transformed functional connectivity matrices.

    For each subject, computes the 200x200 Pearson correlation matrix
    between all pairs of ROI time series, then applies the Fisher
    z-transform: r -> arctanh(r). This converts bounded correlation
    values (-1 to 1) into approximately normally distributed Fisher z
    scores with no fixed range, which are more appropriate for
    statistical modelling.

    Zero-variance ROIs (regions with no signal in some subjects, typically
    due to scanner field-of-view gaps) produce undefined correlations and
    are replaced with 0.

    Parameters
    ----------
    pheno    : phenotypic DataFrame from download_abide()
    ts_files : time-series list from download_abide()

    Returns
    -------
    connectomes : list of (200, 200) float64 arrays
    labels      : list of int -- 1 = ASD, 0 = Control
    subject_ids : list of str
    sites       : list of str
    """
    print("\nComputing functional connectivity matrices ...")

    connectomes, labels, subject_ids, sites = [], [], [], []
    skipped = 0

    for i, ts_path in enumerate(ts_files):
        try:
            # nilearn >= 0.10 returns numpy arrays directly
            if isinstance(ts_path, np.ndarray):
                ts = ts_path
            else:
                ts = np.loadtxt(ts_path)   # shape: (timepoints, 200)
        except Exception as e:
            print(f"  Skipping subject {i}: {e}")
            skipped += 1
            continue

        if ts.shape[0] < MIN_TIMEPOINTS:
            print(f"  Skipping subject {i}: only {ts.shape[0]} timepoints")
            skipped += 1
            continue

        # 200x200 Pearson correlation matrix
        corr = np.corrcoef(ts.T)

        # Fisher z-transform: clip slightly inside (-1, 1) to avoid atanh(+-1) = +-inf
        corr_z = np.arctanh(np.clip(corr, -0.999, 0.999))
        np.fill_diagonal(corr_z, 0.0)
        corr_z = np.nan_to_num(corr_z, nan=0.0, posinf=0.0, neginf=0.0)

        connectomes.append(corr_z)
        labels.append(1 if pheno["DX_GROUP"].iloc[i] == 1 else 0)
        subject_ids.append(str(pheno["SUB_ID"].iloc[i]))
        sites.append(str(pheno["SITE_ID"].iloc[i]))

    print(f"  Processed : {len(connectomes)} subjects")
    print(f"  Skipped   : {skipped} (bad files / short scans)")

    return connectomes, labels, subject_ids, sites


# -- Build ROI metadata -------------------------------------------------------

def build_roi_meta():
    """
    Build ROI name and network assignment lists for the CC200 parcellation.

    Network boundaries are approximate, based on the functional network
    assignments of Power et al. (2011) mapped onto CC200 ROI ordering.

    Returns
    -------
    roi_names : list of 200 strings in the format "NETWORK_index"
    networks  : list of 200 network name strings
    """
    networks  = [""] * 200
    roi_names = []

    for net, rng in NETWORK_MAP.items():
        for r in rng:
            networks[r] = net
            roi_names.append(f"{net}_{r:03d}")

    return roi_names, networks


# -- Save outputs -------------------------------------------------------------

def save(connectomes, labels, subject_ids, sites, roi_names, networks):
    """
    Save all outputs to data/.

    Parameters
    ----------
    connectomes : list of (200, 200) arrays
    labels      : list of int
    subject_ids : list of str
    sites       : list of str
    roi_names   : list of str from build_roi_meta()
    networks    : list of str from build_roi_meta()
    """
    conn_arr   = np.array(connectomes)
    labels_arr = np.array(labels)

    np.save(DATA_DIR / "connectomes.npy", conn_arr)
    np.save(DATA_DIR / "labels.npy",      labels_arr)

    metadata = pd.DataFrame({
        "subject_id": subject_ids,
        "site":       sites,
        "label":      labels,
        "group":      ["ASD" if l == 1 else "CTRL" for l in labels],
    })
    metadata.to_csv(DATA_DIR / "metadata.csv", index=False)

    with open(DATA_DIR / "roi_meta.pkl", "wb") as f:
        pickle.dump({"roi_names": roi_names, "networks": networks}, f)

    n_asd  = labels_arr.sum()
    n_ctrl = (labels_arr == 0).sum()

    print(f"\n  ASD      : {n_asd}")
    print(f"  Controls : {n_ctrl}")
    print(f"  Total    : {len(labels_arr)}")
    print(f"\nSaved to data/")
    print(f"  connectomes.npy  -- shape {conn_arr.shape}, dtype {conn_arr.dtype}")
    print(f"  labels.npy       -- {len(labels_arr)} subjects")
    print(f"  metadata.csv     -- subject IDs, sites, diagnoses")
    print(f"  roi_meta.pkl     -- ROI names and network assignments")
    print("\nNext: python scripts/02_build_graphs.py")


# -- Main ---------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  ABIDE Connectome Download")
    print(f"  Sites: {', '.join(SITES)}")
    print("=" * 60)

    pheno, ts_files = download_abide()
    connectomes, labels, subject_ids, sites = compute_connectomes(pheno, ts_files)

    if len(connectomes) == 0:
        raise RuntimeError("No connectomes computed. Check data download.")

    roi_names, networks = build_roi_meta()
    save(connectomes, labels, subject_ids, sites, roi_names, networks)


if __name__ == "__main__":
    main()
