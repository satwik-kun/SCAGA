import os
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter

# ==========================================================
# PATHS & CONFIGURATION (resolved via machine_config.yaml if available)
# ==========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOTALSEG_DIR = os.path.dirname(SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(TOTALSEG_DIR)
_MACHINE_CFG = os.path.join(_PROJECT_ROOT, "machine_config.yaml")

EDGE_ATLAS_PATH = os.path.join(TOTALSEG_DIR, "edge_probability_atlas.nii.gz")
COMPLEXITY_ATLAS_PATH = os.path.join(TOTALSEG_DIR, "structural_complexity_atlas.nii.gz")
OUTPUT_ATLAS_PATH = os.path.join(TOTALSEG_DIR, "importance_score_weighted_sum.nii.gz")

if os.path.exists(_MACHINE_CFG):
    try:
        import yaml
        with open(_MACHINE_CFG, "r") as f:
            _mc = yaml.safe_load(f) or {}
            _gen_dir = os.path.abspath(os.path.join(_PROJECT_ROOT, _mc.get("generated_dir", "./generated")))
            _atlas_dir = os.path.join(_gen_dir, "atlases")
            os.makedirs(_atlas_dir, exist_ok=True)
            EDGE_ATLAS_PATH = os.path.join(_atlas_dir, "edge_probability_atlas.nii.gz")
            COMPLEXITY_ATLAS_PATH = os.path.join(_atlas_dir, "structural_complexity_atlas.nii.gz")
            OUTPUT_ATLAS_PATH = os.path.join(_atlas_dir, "importance_score_weighted_sum.nii.gz")
    except Exception:
        pass

FORMULATION = "weighted_sum"  # Options: 'weighted_sum', 'product', 'hybrid'
ALPHA = 0.7

# ==========================================================

def min_max_normalize(data):
    """Normalize array to range [0, 1]."""
    d_min = np.min(data)
    d_max = np.max(data)
    if d_max > d_min:
        return (data - d_min) / (d_max - d_min)
    return np.zeros_like(data, dtype=np.float32)

def generate_default_complexity_atlas(edge_data):
    """
    Generate structural complexity (anisotropy / local variance) map from edge map
    if an explicit structural complexity atlas file is not present.
    """
    print("[INFO] Computing structural complexity map from edge data...")
    gx, gy, gz = np.gradient(edge_data)
    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    smooth_grad = gaussian_filter(grad_mag, sigma=1.0)
    complexity = smooth_grad + 0.3 * grad_mag
    return min_max_normalize(complexity).astype(np.float32)

def compute_importance(edge_norm, complexity_norm, formulation=FORMULATION, alpha=ALPHA):
    """
    Compute Importance Score I based on formulation.
    Formulations:
      - Product:      I = E_norm * A_norm
      - Weighted Sum: I = alpha * E_norm + (1 - alpha) * A_norm
      - Hybrid:       I = E_norm * (alpha + (1 - alpha) * A_norm)
    
    CRITICAL: Raw importance values are retained (no post-min-max normalization).
    """
    if formulation == "product":
        importance = edge_norm * complexity_norm
    elif formulation == "weighted_sum":
        importance = alpha * edge_norm + (1.0 - alpha) * complexity_norm
    elif formulation == "hybrid":
        importance = edge_norm * (alpha + (1.0 - alpha) * complexity_norm)
    else:
        raise ValueError(f"Unknown formulation: {formulation}")
    
    return importance.astype(np.float32)

def main():
    print("=" * 60)
    print("SCAGA: Importance Score Atlas Computation")
    print("=" * 60)
    print(f"Edge Atlas         : {EDGE_ATLAS_PATH}")
    print(f"Complexity Atlas   : {COMPLEXITY_ATLAS_PATH}")
    print(f"Formulation        : {FORMULATION}")
    print(f"Alpha              : {ALPHA}")
    print(f"Output Path        : {OUTPUT_ATLAS_PATH}")

    # Load Edge Atlas
    if not os.path.exists(EDGE_ATLAS_PATH):
        raise FileNotFoundError(f"Edge Atlas not found at: {EDGE_ATLAS_PATH}")
    
    edge_nii = nib.load(EDGE_ATLAS_PATH)
    edge_raw = edge_nii.get_fdata().astype(np.float32)
    edge_norm = min_max_normalize(edge_raw)
    print(f"[OK] Loaded Edge Atlas: shape={edge_raw.shape}, raw range=[{edge_raw.min():.4f}, {edge_raw.max():.4f}]")

    # Load or generate Structural Complexity Atlas
    if os.path.exists(COMPLEXITY_ATLAS_PATH):
        comp_nii = nib.load(COMPLEXITY_ATLAS_PATH)
        comp_raw = comp_nii.get_fdata().astype(np.float32)
        comp_norm = min_max_normalize(comp_raw)
        print(f"[OK] Loaded Structural Complexity Atlas: raw range=[{comp_raw.min():.4f}, {comp_raw.max():.4f}]")
    else:
        print(f"[NOTE] Complexity Atlas not found at {COMPLEXITY_ATLAS_PATH}. Generating default anisotropy/complexity map...")
        comp_norm = generate_default_complexity_atlas(edge_norm)
        comp_nii_save = nib.Nifti1Image(comp_norm, edge_nii.affine, edge_nii.header)
        nib.save(comp_nii_save, COMPLEXITY_ATLAS_PATH)
        print(f"[OK] Saved generated Complexity Atlas to: {COMPLEXITY_ATLAS_PATH}")

    imp_map = compute_importance(edge_norm, comp_norm, formulation=FORMULATION, alpha=ALPHA)
    
    imp_nii = nib.Nifti1Image(imp_map, edge_nii.affine, edge_nii.header)
    os.makedirs(os.path.dirname(OUTPUT_ATLAS_PATH), exist_ok=True)
    nib.save(imp_nii, OUTPUT_ATLAS_PATH)
    
    print(f"\n[OK] Computed Importance ({FORMULATION}):")
    print(f"     Range: [{imp_map.min():.6f}, {imp_map.max():.6f}], Mean: {imp_map.mean():.6f}, Std: {imp_map.std():.6f}")
    print(f"     Saved to: {OUTPUT_ATLAS_PATH}")

    print("\n" + "=" * 60)
    print("Importance Score Computation Finished Successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()
