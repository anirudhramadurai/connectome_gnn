"""
02_harmonize.py
---------------
Applies ComBat site harmonization to the functional connectivity matrices,
removing systematic scanner-driven variance introduced by differences between
MRI acquisition sites.

Why this step is necessary
--------------------------
ABIDE data is collected across multiple sites using different MRI scanners,
different acquisition parameters, and different operator practices. These
technical differences introduce systematic variance in the connectivity matrices
that is far larger than the biological ASD signal. Without correction, a
classifier trained on mixed-site data learns to recognize scanner fingerprints
rather than disease biology, which is why the unharmonized baseline achieves
AUC near chance despite real biological signal being present.

ComBat was originally developed to correct for batch effects in genomics
microarray data (Johnson et al., 2007) and has since been widely adopted in
neuroimaging for multi-site harmonization (Fortin et al., 2017). It fits a
linear model of the form:

    Y_ij = alpha_i + X_ij * beta + gamma_i + delta_i * epsilon_ij

where gamma_i and delta_i capture the additive and multiplicative batch effects
for site i, estimated using an empirical Bayes framework that borrows strength
across features. After estimating these parameters, the batch effects are
subtracted to produce harmonized data while preserving biological covariate
effects.

Here we treat the upper triangle of each 200x200 connectivity matrix as a
feature vector (19,900 features per subject), apply ComBat across the three
sites, then reconstruct the full symmetric matrix.

Usage
-----
  python scripts/02_harmonize.py

Inputs
------
  data/connectomes.npy   from 01_fetch_and_prepare.py
  data/labels.npy        from 01_fetch_and_prepare.py
  data/metadata.csv      from 01_fetch_and_prepare.py

Outputs
-------
  data/connectomes_harmonized.npy   (N, 200, 200) site-harmonized connectomes

References
----------
Johnson WE, Li C, Rabinovic A. (2007). Adjusting batch effects in microarray
  expression data using empirical Bayes methods. Biostatistics, 8(1):118-127.
  doi:10.1093/biostatistics/kxj037.

Fortin JP, Parker D, Tunc B, et al. (2017). Harmonization of multi-site
  diffusion tensor imaging data. NeuroImage, 161:149-170.
  doi:10.1016/j.neuroimage.2017.08.047.

Fortin JP, Cullen N, Sheline YI, et al. (2018). Harmonization of cortical
  thickness measurements across scanners and sites. NeuroImage, 167:104-120.
  doi:10.1016/j.neuroimage.2017.11.024.
"""

import numpy as np
import pandas as pd
from pathlib import Path

try:
    from neuroCombat import neuroCombat
except ImportError:
    raise ImportError(
        "neuroCombat is required for site harmonization.\n"
        "Install with: pip install neuroCombat"
    )

DATA_DIR = Path(__file__).parent.parent / "data"


def load_data():
    """Load raw connectomes, labels, and site metadata."""
    connectomes = np.load(DATA_DIR / "connectomes.npy")   # (N, 200, 200)
    labels      = np.load(DATA_DIR / "labels.npy")
    metadata    = pd.read_csv(DATA_DIR / "metadata.csv")
    sites       = metadata["site"].values

    return connectomes, labels, sites, metadata


def extract_upper_triangle(connectomes):
    """
    Extract upper triangle values from each connectivity matrix.

    For a symmetric 200x200 matrix, the upper triangle contains
    200 * 199 / 2 = 19,900 unique pairwise connectivity values.
    ComBat operates on this feature representation rather than the
    full matrix to avoid redundancy.

    Parameters
    ----------
    connectomes : (N, 200, 200) array

    Returns
    -------
    tri_data : (19900, N) array suitable for neuroCombat input
    tri_idx  : tuple of row and column indices for reconstruction
    """
    N, R, _ = connectomes.shape
    tri_idx  = np.triu_indices(R, k=1)   # k=1 excludes diagonal
    tri_data = np.stack([c[tri_idx] for c in connectomes], axis=1)  # (19900, N)
    return tri_data, tri_idx


def reconstruct_matrices(tri_data_harmonized, tri_idx, N, R):
    """
    Reconstruct full symmetric connectivity matrices from harmonized
    upper triangle values.

    Parameters
    ----------
    tri_data_harmonized : (19900, N) array
    tri_idx             : tuple from extract_upper_triangle()
    N                   : number of subjects
    R                   : number of ROIs (200)

    Returns
    -------
    connectomes_harmonized : (N, 200, 200) float64 array
    """
    connectomes_harmonized = np.zeros((N, R, R), dtype=np.float64)

    for i in range(N):
        vals = tri_data_harmonized[:, i]
        connectomes_harmonized[i][tri_idx] = vals
        connectomes_harmonized[i].T[tri_idx] = vals   # enforce symmetry

    return connectomes_harmonized


def run_combat(tri_data, sites, labels):
    """
    Apply ComBat site harmonization to the connectivity feature matrix.

    ComBat expects a (features x samples) matrix and a covariate DataFrame
    with a column identifying the batch (site) for each sample. Crucially,
    the diagnostic label (ASD vs Control) must be passed as a protected
    biological covariate. Without this, ComBat cannot distinguish biological
    variance from site variance and will remove both -- which is why AUC
    degrades when diagnosis is omitted.

    Parameters
    ----------
    tri_data : (19900, N) feature matrix
    sites    : (N,) array of site strings
    labels   : (N,) int array -- 1 = ASD, 0 = Control

    Returns
    -------
    harmonized : (19900, N) array with site effects removed,
                 biological ASD-vs-Control signal preserved
    """
    covars = pd.DataFrame({
        "site":      sites,
        "diagnosis": labels.astype(int),
    })

    print(f"  Running ComBat on {tri_data.shape[0]} features x {tri_data.shape[1]} subjects")
    print(f"  Sites: {dict(pd.Series(sites).value_counts())}")
    print(f"  Protecting diagnosis covariate (ASD=1, CTRL=0)")

    result = neuroCombat(
        dat              = tri_data,
        covars           = covars,
        batch_col        = "site",
        categorical_cols = ["diagnosis"],
    )

    return result["data"]


def print_summary(connectomes_raw, connectomes_harmonized, sites):
    """
    Print before/after statistics to verify harmonization reduced
    between-site variance without removing within-site biological signal.
    """
    unique_sites = np.unique(sites)

    print("\nSite mean FC before vs after harmonization:")
    print(f"  {'Site':<12}  {'Before':>8}  {'After':>8}  {'Change':>8}")

    for site in unique_sites:
        idx    = sites == site
        before = connectomes_raw[idx].mean()
        after  = connectomes_harmonized[idx].mean()
        print(f"  {site:<12}  {before:>8.4f}  {after:>8.4f}  {after - before:>+8.4f}")

    print(f"\nOverall mean FC:")
    print(f"  Before: {connectomes_raw.mean():.4f}  (SD: {connectomes_raw.std():.4f})")
    print(f"  After : {connectomes_harmonized.mean():.4f}  (SD: {connectomes_harmonized.std():.4f})")

    # Between-site variance reduction
    site_means_before = np.array([connectomes_raw[sites == s].mean() for s in unique_sites])
    site_means_after  = np.array([connectomes_harmonized[sites == s].mean() for s in unique_sites])
    print(f"\nBetween-site variance:")
    print(f"  Before: {site_means_before.var():.6f}")
    print(f"  After : {site_means_after.var():.6f}")


def save(connectomes_harmonized):
    """Save harmonized connectomes to data/."""
    np.save(DATA_DIR / "connectomes_harmonized.npy", connectomes_harmonized)
    print(f"\nSaved to data/")
    print(f"  connectomes_harmonized.npy  -- shape {connectomes_harmonized.shape}")
    print("\nNext: python scripts/03_build_graphs.py")


def main():
    print("=" * 60)
    print("  ComBat Site Harmonization")
    print("  Removing scanner-site variance from connectivity matrices")
    print("=" * 60)

    connectomes, labels, sites, metadata = load_data()
    N, R, _ = connectomes.shape

    print(f"\nLoaded {N} subjects | {R} ROIs | {len(np.unique(sites))} sites")

    print("\nExtracting upper triangle features ...")
    tri_data, tri_idx = extract_upper_triangle(connectomes)

    print("\nApplying ComBat ...")
    tri_harmonized = run_combat(tri_data, sites, labels)

    print("\nReconstructing connectivity matrices ...")
    connectomes_harmonized = reconstruct_matrices(tri_harmonized, tri_idx, N, R)

    print_summary(connectomes, connectomes_harmonized, sites)
    save(connectomes_harmonized)


if __name__ == "__main__":
    main()