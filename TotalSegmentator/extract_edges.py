import os
import subprocess
import nibabel as nib
import numpy as np
from skimage.segmentation import find_boundaries
from tqdm import tqdm

# =====================================================
# PATHS (resolved via machine_config.yaml if available)
# =====================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_MACHINE_CFG = os.path.join(_PROJECT_ROOT, "machine_config.yaml")

_dataset_root = os.path.join(_PROJECT_ROOT, "Preprocessing", "data")
_gen_dir = _SCRIPT_DIR
if os.path.exists(_MACHINE_CFG):
    try:
        import yaml
        with open(_MACHINE_CFG, "r") as f:
            _mc = yaml.safe_load(f) or {}
            _dataset_root = os.path.abspath(os.path.join(_PROJECT_ROOT, _mc.get("dataset_root", "./Preprocessing/data")))
            _gen_dir = os.path.abspath(os.path.join(_PROJECT_ROOT, _mc.get("generated_dir", "./generated")))
    except Exception:
        pass

INPUT_FOLDER = os.path.join(_dataset_root, "LUNA16", "processed", "images")
OUTPUT_FOLDER = os.path.join(_gen_dir, "edges") if os.path.exists(_MACHINE_CFG) else os.path.join(_SCRIPT_DIR, "edges")

# =====================================================

TASK = "total"

# Find TotalSegmentator path on Windows if not in PATH
import shutil
totalseg_bin = "TotalSegmentator"
if not shutil.which(totalseg_bin):
    appdata_path = os.path.expandvars(r'%APPDATA%\Python\Python314\Scripts\TotalSegmentator.exe')
    if os.path.exists(appdata_path):
        totalseg_bin = appdata_path
    else:
        python314_path = r"C:\Python314\Scripts\TotalSegmentator.exe"
        if os.path.exists(python314_path):
            totalseg_bin = python314_path

# =====================================================

if not os.path.exists(INPUT_FOLDER):
    raise FileNotFoundError(
        f"\nInput folder does not exist:\n{INPUT_FOLDER}"
    )

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

nii_files = sorted(
    f for f in os.listdir(INPUT_FOLDER)
    if f.endswith(".nii.gz")
)

print("=" * 60)
print(f"Found {len(nii_files)} scans.")
print("=" * 60)

success = 0
failed = 0
skipped = 0

for fname in tqdm(nii_files):

    input_ct = os.path.join(INPUT_FOLDER, fname)

    temp_seg = os.path.join(
        OUTPUT_FOLDER,
        "__temp_segmentation.nii.gz"
    )

    edge_name = fname.replace(".nii.gz", "_edge.nii.gz")

    output_edge = os.path.join(
        OUTPUT_FOLDER,
        edge_name
    )

    # --------------------------------------------------
    # Skip if already processed
    # --------------------------------------------------

    if os.path.exists(output_edge):
        skipped += 1
        continue

    print(f"\nProcessing: {fname}")

    try:

        # --------------------------------------------------
        # Run TotalSegmentator
        # --------------------------------------------------

        cmd = [
            totalseg_bin,
            "-i", input_ct,
            "-o", temp_seg,
            "-ta", TASK,
            "-f",
            "-ml",
            "-q"
        ]

        subprocess.run(cmd, check=True)

        # --------------------------------------------------
        # Load segmentation
        # --------------------------------------------------

        nii = nib.load(temp_seg)
        labels = nii.get_fdata().astype(np.int16)

        # --------------------------------------------------
        # Extract 3D anatomical boundaries
        # --------------------------------------------------

        edges = find_boundaries(
            labels,
            connectivity=3,
            mode="outer"
        )

        edge_img = nib.Nifti1Image(
            edges.astype(np.uint8),
            affine=nii.affine,
            header=nii.header
        )

        nib.save(edge_img, output_edge)

        success += 1

    except Exception as e:

        failed += 1

        print("\n" + "=" * 60)
        print(f"FAILED: {fname}")
        print(e)
        print("=" * 60)

    finally:

        # --------------------------------------------------
        # Remove temporary segmentation
        # --------------------------------------------------

        if os.path.exists(temp_seg):
            os.remove(temp_seg)

print("\n")
print("=" * 60)
print("Processing Complete")
print("=" * 60)
print(f"Successful : {success}")
print(f"Skipped    : {skipped}")
print(f"Failed     : {failed}")
print("=" * 60)