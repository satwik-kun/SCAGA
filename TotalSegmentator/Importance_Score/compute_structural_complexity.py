import os
import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree

# ==========================================================
# PATHS & CONFIGURATION (resolved via machine_config.yaml if available)
# ==========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOTALSEG_DIR = os.path.dirname(SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(TOTALSEG_DIR)
_MACHINE_CFG = os.path.join(_PROJECT_ROOT, "machine_config.yaml")

EDGE_ATLAS_PATH = os.path.join(TOTALSEG_DIR, "edge_probability_atlas.nii.gz")
OUTPUT_PATH = os.path.join(TOTALSEG_DIR, "structural_complexity_atlas.nii.gz")

if os.path.exists(_MACHINE_CFG):
    try:
        import yaml
        with open(_MACHINE_CFG, "r") as f:
            _mc = yaml.safe_load(f) or {}
            _gen_dir = os.path.abspath(os.path.join(_PROJECT_ROOT, _mc.get("generated_dir", "./generated")))
            _atlas_dir = os.path.join(_gen_dir, "atlases")
            os.makedirs(_atlas_dir, exist_ok=True)
            EDGE_ATLAS_PATH = os.path.join(_atlas_dir, "edge_probability_atlas.nii.gz")
            OUTPUT_PATH = os.path.join(_atlas_dir, "structural_complexity_atlas.nii.gz")
    except Exception:
        pass

K_NEIGHBORS = 20  # Number of KNN neighbors for local 3D neighborhood covariance
METRIC_SELECTED = "anisotropy"  # Chosen metric for SCAGA

# ==========================================================

def compute_3d_eigenvalues_knn(coords, k_neighbors=K_NEIGHBORS):
    """
    Construct cKDTree and KNN neighborhood for 3D point cloud coordinates.
    Compute 3D covariance matrix for each neighborhood and extract sorted eigenvalues.
    Returns:
      evals: ndarray of shape (N, 3) where eval[:, 0] >= eval[:, 1] >= eval[:, 2] >= 0
    """
    print(f"[INFO] Constructing 3D cKDTree for {len(coords)} points...")
    tree = cKDTree(coords)
    
    print(f"[INFO] Querying {k_neighbors}-nearest neighbors...")
    dists, idxs = tree.query(coords, k=k_neighbors)
    
    print(f"[INFO] Computing 3D local covariance matrices and eigenvalue decomposition...")
    # Gather neighbors: shape (N, k, 3)
    neighbors = coords[idxs]
    
    # Center neighbors around local mean: shape (N, k, 3)
    means = np.mean(neighbors, axis=1, keepdims=True)
    centered = neighbors - means
    
    # Covariance matrices per point: (N, 3, 3)
    # C = (1/k) * sum_i (x_i - mean)(x_i - mean)^T
    covs = np.matmul(centered.transpose(0, 2, 1), centered) / float(k_neighbors)
    
    # Eigenvalue decomposition for symmetric matrices
    evals = np.linalg.eigvalsh(covs)  # Ascending order: lambda_3 <= lambda_2 <= lambda_1
    
    # Sort descending: lambda_1 >= lambda_2 >= lambda_3 >= 0
    evals = evals[:, ::-1]
    evals = np.clip(evals, a_min=0.0, a_max=None)
    return evals

def compute_all_complexity_metrics(evals):
    """
    Compute 3D geometric structural complexity metrics from sorted eigenvalues (l1 >= l2 >= l3 >= 0):
      1. Anisotropy:        A = (l1 - l3) / l1
      2. Planarity:         P = (l2 - l3) / l1
      3. Linearity:         L = (l1 - l2) / l1
      4. Surface Variation: E_v = l3 / (l1 + l2 + l3)
      5. Omnivariance:      O = (l1 * l2 * l3)^(1/3)
      6. Eigenentropy:      H = - sum(e_i * ln(e_i)) where e_i = l_i / sum(l_j)
    """
    l1, l2, l3 = evals[:, 0], evals[:, 1], evals[:, 2]
    sum_l = l1 + l2 + l3 + 1e-10

    # 1. Anisotropy
    anisotropy = (l1 - l3) / (l1 + 1e-10)
    
    # 2. Planarity
    planarity = (l2 - l3) / (l1 + 1e-10)
    
    # 3. Linearity
    linearity = (l1 - l2) / (l1 + 1e-10)
    
    # 4. Surface Variation
    surface_variation = l3 / sum_l
    
    # 5. Omnivariance
    omnivariance = np.cbrt(np.maximum(l1 * l2 * l3, 0.0))
    
    # 6. Eigenentropy
    e1, e2, e3 = l1 / sum_l, l2 / sum_l, l3 / sum_l
    p = np.column_stack([e1, e2, e3])
    p_safe = np.maximum(p, 1e-10)
    eigenentropy = -np.sum(p_safe * np.log(p_safe), axis=1)

    return {
        "anisotropy": anisotropy,
        "planarity": planarity,
        "linearity": linearity,
        "surface_variation": surface_variation,
        "omnivariance": omnivariance,
        "eigenentropy": eigenentropy,
    }

def compute_structural_complexity_map(edge_atlas_path=EDGE_ATLAS_PATH, output_path=OUTPUT_PATH, metric=METRIC_SELECTED, k_neighbors=K_NEIGHBORS):
    """
    Compute Structural Complexity Atlas (Anisotropy) using KDTree + KNN Covariance Eigenvalue Decomposition.
    """
    if not os.path.exists(edge_atlas_path):
        raise FileNotFoundError(f"Edge atlas not found at: {edge_atlas_path}")

    nii = nib.load(edge_atlas_path)
    edge_data = nii.get_fdata().astype(np.float32)
    shape = edge_data.shape

    # Extract non-zero voxel coordinates for point cloud neighborhood construction
    non_zero_mask = edge_data > 1e-4
    coords = np.column_stack(np.where(non_zero_mask)).astype(np.float32)

    if len(coords) < k_neighbors:
        raise ValueError(f"Insufficient non-zero points ({len(coords)}) for k={k_neighbors} neighbors.")

    print(f"[OK] Extracted {len(coords)} non-zero voxels for KDTree + KNN 3D Covariance analysis.")

    # 1. KDTree + KNN Covariance Eigenvalue Decomposition
    evals = compute_3d_eigenvalues_knn(coords, k_neighbors=k_neighbors)

    # 2. Compute all 6 geometric structural complexity metrics
    metrics = compute_all_complexity_metrics(evals)

    if metric not in metrics:
        raise ValueError(f"Selected metric '{metric}' invalid. Options: {list(metrics.keys())}")

    chosen_values = metrics[metric]

    # Normalize chosen metric values to [0, 1]
    m_min, m_max = chosen_values.min(), chosen_values.max()
    norm_values = (chosen_values - m_min) / (m_max - m_min + 1e-8) if m_max > m_min else chosen_values

    # Map computed 3D point cloud metrics back into 3D NIfTI volume
    complexity_volume = np.zeros(shape, dtype=np.float32)
    nz_indices = np.where(non_zero_mask)
    complexity_volume[nz_indices] = norm_values

    comp_nii = nib.Nifti1Image(complexity_volume, nii.affine, nii.header)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(comp_nii, output_path)

    print(f"\n[OK] Saved Structural Complexity Atlas ({metric.upper()}) to: {output_path}")
    print(f"     Shape: {complexity_volume.shape}, Range: [{complexity_volume.min():.4f}, {complexity_volume.max():.4f}], Mean: {complexity_volume.mean():.6f}")
    return complexity_volume

if __name__ == "__main__":
    compute_structural_complexity_map()
