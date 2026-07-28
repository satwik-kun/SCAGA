import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# =====================================================
# SETTINGS
# =====================================================

ATLAS_FILE = "edge_probability_atlas.nii.gz"
POINTS_FILE = "sampled_points.npy"

GS_RES = 12

# =====================================================
# LOAD
# =====================================================

atlas = nib.load(ATLAS_FILE).get_fdata().astype(np.float32)
sampled = np.load(POINTS_FILE)

shape = np.array(atlas.shape)

print("="*70)
print("Evaluation of Gaussian Initialization")
print("="*70)

print("Atlas Shape :", atlas.shape)
print("Loaded Points :", sampled.shape)

# =====================================================
# BASIC CHECKS
# =====================================================

print("\nBasic Statistics")
print("-"*70)

print("Min :", sampled.min())
print("Max :", sampled.max())

print("Mean XYZ :", sampled.mean(axis=0))
print("Std XYZ  :", sampled.std(axis=0))

print("Unique :", len(np.unique(sampled, axis=0)))

# =====================================================
# SAMPLE PROBABILITIES
# =====================================================

vox = np.round(
    sampled * (shape-1)
).astype(int)

probs = atlas[
    vox[:,0],
    vox[:,1],
    vox[:,2]
]

print("\nSampled Point Probabilities")
print("-"*70)

print("Mean :", probs.mean())
print("Median :", np.median(probs))
print("Std :", probs.std())
print("Min :", probs.min())
print("Max :", probs.max())

# =====================================================
# ORIGINAL DIF GRID
# =====================================================

grid = np.mgrid[:GS_RES,:GS_RES,:GS_RES] / GS_RES
grid = grid.reshape(3,-1).T

grid_vox = np.round(
    grid * (shape-1)
).astype(int)

grid_probs = atlas[
    grid_vox[:,0],
    grid_vox[:,1],
    grid_vox[:,2]
]

print("\nOriginal Uniform Grid")
print("-"*70)

print("Mean :", grid_probs.mean())
print("Median :", np.median(grid_probs))
print("Std :", grid_probs.std())
print("Min :", grid_probs.min())
print("Max :", grid_probs.max())

# =====================================================
# RANDOM POINTS
# =====================================================

rand = np.random.rand(len(sampled),3)

rand_vox = np.round(
    rand * (shape-1)
).astype(int)

rand_probs = atlas[
    rand_vox[:,0],
    rand_vox[:,1],
    rand_vox[:,2]
]

print("\nRandom Sampling")
print("-"*70)

print("Mean :", rand_probs.mean())
print("Median :", np.median(rand_probs))
print("Std :", rand_probs.std())

# =====================================================
# IMPROVEMENT
# =====================================================

print("\nImprovement")
print("-"*70)

print("Atlas Mean                :", atlas.mean())

print("Random Mean               :", rand_probs.mean())
print("Uniform Grid Mean         :", grid_probs.mean())
print("Mixed Sampling Mean       :", probs.mean())

print()

print("Gain over Uniform Grid    :",
      probs.mean()/grid_probs.mean())

print("Gain over Random          :",
      probs.mean()/rand_probs.mean())

# =====================================================
# HISTOGRAM
# =====================================================

plt.figure(figsize=(8,5))

plt.hist(
    grid_probs,
    bins=40,
    alpha=0.5,
    label="Uniform Grid"
)

plt.hist(
    probs,
    bins=40,
    alpha=0.5,
    label="Mixed Sampling"
)

plt.xlabel("Atlas Probability")
plt.ylabel("Count")
plt.title("Probability Distribution of Gaussian Centers")

plt.legend()

plt.tight_layout()

plt.show()

# =====================================================
# PASS / FAIL
# =====================================================

print("\nVerification")
print("-"*70)

checks = []

checks.append(sampled.shape==(1728,3))
checks.append(sampled.min()>=0)
checks.append(sampled.max()<=1)
checks.append(len(np.unique(sampled,axis=0))==1728)
checks.append(probs.mean()>grid_probs.mean())

labels = [
    "Correct Shape",
    "Coordinates >=0",
    "Coordinates <=1",
    "No Duplicate Points",
    "Higher Atlas Probability than Uniform Grid"
]

for l,c in zip(labels,checks):
    print(("PASS" if c else "FAIL"),"-",l)

print("="*70)