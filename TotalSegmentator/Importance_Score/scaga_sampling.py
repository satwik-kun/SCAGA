import os
import numpy as np
import nibabel as nib

# ==========================================================
# PATHS & CONFIGURATION (resolved via machine_config.yaml if present)
# ==========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOTALSEG_DIR = os.path.dirname(SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(TOTALSEG_DIR)
_MACHINE_CFG = os.path.join(_PROJECT_ROOT, "machine_config.yaml")

ATLAS_PATH = os.path.join(TOTALSEG_DIR, "importance_score_weighted_sum.nii.gz")
if not os.path.exists(ATLAS_PATH):
    ATLAS_PATH = os.path.join(TOTALSEG_DIR, "edge_probability_atlas.nii.gz")

OUTPUT_PATH = os.path.join(TOTALSEG_DIR, "sampled_points_weighted_sum.npy")

if os.path.exists(_MACHINE_CFG):
    try:
        import yaml
        with open(_MACHINE_CFG, "r") as f:
            _mc = yaml.safe_load(f) or {}
            _gen_dir = os.path.abspath(os.path.join(_PROJECT_ROOT, _mc.get("generated_dir", "./generated")))
            _atlas_dir = os.path.join(_gen_dir, "atlases")
            _pts_dir = os.path.join(_gen_dir, "points")
            os.makedirs(_atlas_dir, exist_ok=True)
            os.makedirs(_pts_dir, exist_ok=True)
            _candidate = os.path.join(_atlas_dir, "importance_score_weighted_sum.nii.gz")
            ATLAS_PATH = _candidate if os.path.exists(_candidate) else os.path.join(_atlas_dir, "edge_probability_atlas.nii.gz")
            OUTPUT_PATH = os.path.join(_pts_dir, "sampled_points_weighted_sum.npy")
    except Exception:
        pass

GS_RES = 12
NUM_POINTS = GS_RES ** 3          # 1728
EDGE_RATIO = 0.80                 # 80% Guided / 20% Uniform
SEED = 42

# ==========================================================

def sample_gaussian_centers(atlas_path=ATLAS_PATH, output_path=OUTPUT_PATH, num_points=NUM_POINTS, edge_ratio=EDGE_RATIO, seed=SEED):
    """
    Sample Gaussian centers using 80% guided probability sampling and 20% uniform sampling.
    Random seed is fixed (default=42) so uniform components are identical across formulations.
    """
    if not os.path.exists(atlas_path):
        raise FileNotFoundError(f"Atlas file not found: {atlas_path}")

    # Load atlas
    nii = nib.load(atlas_path)
    atlas = nii.get_fdata().astype(np.float32)

    edge_points_count = int(num_points * edge_ratio)
    uniform_points_count = num_points - edge_points_count

    print("=" * 60)
    print("SCAGA: Gaussian Center Sampling")
    print("=" * 60)
    print(f"Atlas Path     : {atlas_path}")
    print(f"Output Path    : {output_path}")
    print(f"Total Points   : {num_points}")
    print(f"Guided ({round(edge_ratio*100)}%)  : {edge_points_count}")
    print(f"Uniform ({round((1-edge_ratio)*100)}%) : {uniform_points_count}")
    print(f"Random Seed    : {seed}")

    # Flatten and normalize atlas to probability distribution
    prob = atlas.flatten()
    prob_sum = prob.sum()
    if prob_sum == 0 or np.isnan(prob_sum):
        raise RuntimeError("Atlas contains zero or invalid total probability.")
    
    prob = prob / prob_sum

    # Set seed for reproducible sampling
    np.random.seed(seed)

    # 1. Guided sampling
    indices = np.random.choice(
        len(prob),
        size=edge_points_count,
        replace=False,
        p=prob
    )
    coords = np.column_stack(np.unravel_index(indices, atlas.shape))

    coords_norm = coords.astype(np.float32)
    coords_norm[:, 0] /= (atlas.shape[0] - 1)
    coords_norm[:, 1] /= (atlas.shape[1] - 1)
    coords_norm[:, 2] /= (atlas.shape[2] - 1)

    # 2. Uniform sampling
    uniform_points = np.random.uniform(
        0.0, 1.0, size=(uniform_points_count, 3)
    ).astype(np.float32)

    # Combine guided + uniform
    points = np.vstack([coords_norm, uniform_points])

    # Shuffle combined points
    np.random.shuffle(points)

    # Validation
    assert points.shape == (num_points, 3), f"Expected {(num_points, 3)}, got {points.shape}"
    assert np.all(points >= 0.0) and np.all(points <= 1.0), "Point coordinates out of [0, 1] range!"

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.save(output_path, points)
    print(f"\n[OK] Saved sampled points to: {output_path} (Shape: {points.shape})")
    print("=" * 60)
    return points

if __name__ == "__main__":
    sample_gaussian_centers()
