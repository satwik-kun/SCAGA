import os
import numpy as np
import nibabel as nib

# ============================================================
# SETTINGS & PATHS (resolved via machine_config.yaml if present)
# ============================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_MACHINE_CFG = os.path.join(_PROJECT_ROOT, "machine_config.yaml")

ATLAS = "edge_probability_atlas.nii.gz"

GS_RES = 12
NUM_POINTS = GS_RES ** 3          # 1728
EDGE_RATIO = 0.80                 # 80% Edge / 20% Uniform
EDGE_POINTS = int(NUM_POINTS * EDGE_RATIO)
UNIFORM_POINTS = NUM_POINTS - EDGE_POINTS
SEED = 42

ratio_name = f"{round(EDGE_RATIO*100)}edge_{round((1-EDGE_RATIO)*100)}uniform"
POINTS_FILE = f"sampled_points_{ratio_name}.npy"
VIS_FILE = f"mixed_sampling_visualization_{ratio_name}.nii.gz"

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
            ATLAS = os.path.join(_atlas_dir, "edge_probability_atlas.nii.gz")
            POINTS_FILE = os.path.join(_pts_dir, f"sampled_points_{ratio_name}.npy")
            VIS_FILE = os.path.join(_atlas_dir, f"mixed_sampling_visualization_{ratio_name}.nii.gz")
    except Exception:
        pass

# ============================================================

np.random.seed(SEED)

print("=" * 70)
print("Mixed Edge-Guided Gaussian Initialization")
print("=" * 70)

print(f"Atlas           : {ATLAS}")
print(f"GS Resolution   : {GS_RES}")
print(f"Total Points    : {NUM_POINTS}")
print(f"Edge Ratio      : {EDGE_RATIO:.2f}")
print(f"Uniform Ratio   : {1-EDGE_RATIO:.2f}")
print(f"Random Seed     : {SEED}")

# ============================================================
# Load Atlas
# ============================================================

nii = nib.load(ATLAS)
atlas = nii.get_fdata().astype(np.float32)

print("\nAtlas Statistics")
print("-" * 70)

print(f"Shape           : {atlas.shape}")
print(f"Min             : {atlas.min():.6f}")
print(f"Max             : {atlas.max():.6f}")
print(f"Mean            : {atlas.mean():.6f}")
print(f"Std             : {atlas.std():.6f}")

# ============================================================
# Convert Atlas into Probability Distribution
# ============================================================

prob = atlas.flatten()

if prob.sum() == 0:
    raise RuntimeError("Atlas contains no probability values.")

prob /= prob.sum()

# ============================================================
# Sample Edge Points
# ============================================================

print("\nSampling Edge Points...")

indices = np.random.choice(
    len(prob),
    size=EDGE_POINTS,
    replace=False,
    p=prob
)

coords = np.column_stack(
    np.unravel_index(indices, atlas.shape)
)

coords_norm = coords.astype(np.float32)

coords_norm[:, 0] /= (atlas.shape[0] - 1)
coords_norm[:, 1] /= (atlas.shape[1] - 1)
coords_norm[:, 2] /= (atlas.shape[2] - 1)

# ============================================================
# Generate Uniform Points
# ============================================================

print("Generating Uniform Points...")

uniform_points = np.random.uniform(
    0.0,
    1.0,
    size=(UNIFORM_POINTS, 3)
).astype(np.float32)

# ============================================================
# Combine
# ============================================================

points = np.vstack([
    coords_norm,
    uniform_points
])

# Shuffle so edge and uniform points are mixed
np.random.shuffle(points)

# ============================================================
# Validation
# ============================================================

print("\nValidating Point Cloud...")

assert points.shape == (NUM_POINTS, 3), \
    f"Expected {(NUM_POINTS,3)}, got {points.shape}"

assert np.all(points >= 0.0), \
    "Coordinates below 0 detected."

assert np.all(points <= 1.0), \
    "Coordinates above 1 detected."

unique_points = np.unique(points, axis=0)

assert len(unique_points) == NUM_POINTS, \
    "Duplicate Gaussian centers detected."

# ============================================================
# Save Gaussian Initialization
# ============================================================

np.save(
    POINTS_FILE,
    points
)

print(f"Saved: {POINTS_FILE}")

# ============================================================
# Visualization Volume
#
# 0   = Background
# 100 = Edge-guided
# 255 = Uniform
# ============================================================

vis = np.zeros(
    atlas.shape,
    dtype=np.uint8
)

# Edge voxels

vis[
    coords[:, 0],
    coords[:, 1],
    coords[:, 2]
] = 100

# Uniform voxels

uniform_voxels = np.round(
    uniform_points * (np.array(atlas.shape) - 1)
).astype(np.int32)

vis[
    uniform_voxels[:, 0],
    uniform_voxels[:, 1],
    uniform_voxels[:, 2]
] = 255

vis_img = nib.Nifti1Image(
    vis,
    affine=nii.affine,
    header=nii.header
)

nib.save(
    vis_img,
    VIS_FILE
)

print(f"Saved: {VIS_FILE}")

# ============================================================
# Statistics
# ============================================================

sampled_probs = atlas[
    coords[:, 0],
    coords[:, 1],
    coords[:, 2]
]

print()
print("=" * 70)
print("Sampling Statistics")
print("=" * 70)

print(f"Edge Points         : {EDGE_POINTS}")
print(f"Uniform Points      : {UNIFORM_POINTS}")
print(f"Total Points        : {NUM_POINTS}")

print()

print(f"Unique Edge Points  : {len(np.unique(indices))}")

print()

print("Coordinate Range")
print(f"Min : {points.min(axis=0)}")
print(f"Max : {points.max(axis=0)}")

print()

print("Uniform Distribution Mean")
print(f"X : {uniform_points[:,0].mean():.4f}")
print(f"Y : {uniform_points[:,1].mean():.4f}")
print(f"Z : {uniform_points[:,2].mean():.4f}")

print()

print("Edge Probability Statistics")
print(f"Mean : {sampled_probs.mean():.6f}")
print(f"Std  : {sampled_probs.std():.6f}")
print(f"Min  : {sampled_probs.min():.6f}")
print(f"Max  : {sampled_probs.max():.6f}")

print()

print("Edge Probability Percentiles")

for p in [5, 25, 50, 75, 90, 95, 99]:
    print(f"{p:>2}% : {np.percentile(sampled_probs, p):.6f}")

print()
print("=" * 70)
print("Verification Summary")
print("=" * 70)

print(f"[OK] {POINTS_FILE} created")
print(f"[OK] {VIS_FILE} created")
print("[OK] Shape correct")
print("[OK] Coordinates normalized to [0,1]")
print("[OK] No duplicate Gaussian centers")
print("[OK] Mixed sampling completed")
print("[OK] Ready for DIF-Gaussian")

print("=" * 70)