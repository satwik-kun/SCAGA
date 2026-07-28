import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

# ==========================================================
# PATHS (resolved via machine_config.yaml if available)
# ==========================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_MACHINE_CFG = os.path.join(_PROJECT_ROOT, "machine_config.yaml")

EDGE_FOLDER = os.path.join(_SCRIPT_DIR, "edges")
SUM_OUTPUT = os.path.join(_SCRIPT_DIR, "edge_sum.nii.gz")
ATLAS_OUTPUT = os.path.join(_SCRIPT_DIR, "edge_probability_atlas.nii.gz")

if os.path.exists(_MACHINE_CFG):
    try:
        import yaml
        with open(_MACHINE_CFG, "r") as f:
            _mc = yaml.safe_load(f) or {}
            _gen_dir = os.path.abspath(os.path.join(_PROJECT_ROOT, _mc.get("generated_dir", "./generated")))
            EDGE_FOLDER = os.path.join(_gen_dir, "edges")
            _atlas_dir = os.path.join(_gen_dir, "atlases")
            os.makedirs(_atlas_dir, exist_ok=True)
            SUM_OUTPUT = os.path.join(_atlas_dir, "edge_sum.nii.gz")
            ATLAS_OUTPUT = os.path.join(_atlas_dir, "edge_probability_atlas.nii.gz")
    except Exception:
        pass

# ----------------------------------------------------------

edge_files = sorted([
    f for f in os.listdir(EDGE_FOLDER)
    if f.endswith("_edge.nii.gz")
])

if len(edge_files) == 0:
    raise RuntimeError("No edge files found.")

print("=" * 60)
print(f"Found {len(edge_files)} edge maps")
print("=" * 60)

# ----------------------------------------------------------
# Load first scan to determine shape
# ----------------------------------------------------------

first = nib.load(os.path.join(EDGE_FOLDER, edge_files[0]))

shape = first.shape

print("Volume Shape:", shape)

edge_sum = np.zeros(shape, dtype=np.float32)

# ----------------------------------------------------------
# Sum all scans
# ----------------------------------------------------------

for fname in tqdm(edge_files):

    nii = nib.load(os.path.join(EDGE_FOLDER, fname))

    edge = nii.get_fdata()

    edge_sum += edge

# ----------------------------------------------------------
# Save summed atlas
# ----------------------------------------------------------

sum_img = nib.Nifti1Image(
    edge_sum,
    affine=first.affine,
    header=first.header
)

nib.save(sum_img, SUM_OUTPUT)

print("\nSaved summed atlas:")
print(SUM_OUTPUT)

# ----------------------------------------------------------
# Compute probability atlas
# ----------------------------------------------------------

edge_probability = edge_sum / len(edge_files)

atlas_img = nib.Nifti1Image(
    edge_probability.astype(np.float32),
    affine=first.affine,
    header=first.header
)

nib.save(atlas_img, ATLAS_OUTPUT)

print("\nSaved probability atlas:")
print(ATLAS_OUTPUT)

print("\nDone.")
print(f"Maximum count : {edge_sum.max():.0f}")
print(f"Maximum probability : {edge_probability.max():.4f}")