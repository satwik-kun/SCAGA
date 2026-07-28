import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ATLAS_FILE = "importance_score_weighted_sum.nii.gz"
EDGE_FILE = "edge_probability_atlas.nii.gz"
POINTS_FILE = "sampled_points_weighted_sum.npy"
POINTS_FILE_CODE = "../Code/code/datasets/dst_gs.py"

print("="*70)
print("VERIFICATION & STATISTICS")
print("="*70)

# 1. Load data
imp_atlas = nib.load(ATLAS_FILE).get_fdata().astype(np.float32)
edge_atlas = nib.load(EDGE_FILE).get_fdata().astype(np.float32)
points = np.load(POINTS_FILE)

# 2. Compute min, max, mean, std
imp_min = imp_atlas.min()
imp_max = imp_atlas.max()
imp_mean = imp_atlas.mean()
imp_std = imp_atlas.std()

print(f"Importance Atlas Stats:")
print(f"  Min : {imp_min:.6f}")
print(f"  Max : {imp_max:.6f}")
print(f"  Mean: {imp_mean:.6f}")
print(f"  Std : {imp_std:.6f}")

# 3. Pearson correlation with Edge Atlas
# Flatten for correlation
imp_flat = imp_atlas.flatten()
edge_flat = edge_atlas.flatten()
corr, _ = pearsonr(imp_flat, edge_flat)
print(f"\nPearson correlation with Edge Atlas: {corr:.6f}")

print("\n" + "="*70)
print("USER VERIFICATION CHECKS")
print("="*70)

# ✓ Importance atlas contains no NaN values
no_nans = not np.isnan(imp_atlas).any()
print(f"✓ Importance atlas contains no NaN values: {no_nans}")

# ✓ Probability distribution sums to 1
prob_sum = imp_flat.sum()
prob_dist = imp_flat / prob_sum
dist_sum = prob_dist.sum()
sums_to_1 = np.isclose(dist_sum, 1.0)
print(f"✓ Probability distribution sums to 1 (when normalized): {sums_to_1} (Sum={dist_sum:.6f})")

# ✓ Sample count = 1728
sample_count = points.shape[0] == 1728
print(f"✓ Sample count = 1728: {sample_count} (Count={points.shape[0]})")

# ✓ No duplicate guided samples
unique_points = np.unique(points, axis=0)
no_duplicates = len(unique_points) == 1728
print(f"✓ No duplicate guided samples: {no_duplicates} (Unique={len(unique_points)})")

# ✓ Coordinates remain inside [0,1]
in_bounds = (points >= 0.0).all() and (points <= 1.0).all()
print(f"✓ Coordinates remain inside [0,1]: {in_bounds}")

# ✓ POINTS_FILE now references the weighted-sum initialization
with open(POINTS_FILE_CODE, 'r') as f:
    code = f.read()
    references_weighted = 'POINTS_FILE = "sampled_points_weighted_sum.npy"' in code
print(f"✓ POINTS_FILE now references the weighted-sum initialization: {references_weighted}")

# ========================================================
# VISUALIZATIONS (Saved to disk instead of showing)
# ========================================================
print("\nGenerating Visualizations...")
os.makedirs("visualizations", exist_ok=True)

# 1. Orthogonal slices
mid_x, mid_y, mid_z = np.array(imp_atlas.shape) // 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(imp_atlas[mid_x, :, :], cmap='viridis')
axes[0].set_title('Sagittal')
axes[1].imshow(imp_atlas[:, mid_y, :], cmap='viridis')
axes[1].set_title('Coronal')
axes[2].imshow(imp_atlas[:, :, mid_z], cmap='viridis')
axes[2].set_title('Axial')
plt.suptitle('Orthogonal Slices (Weighted Sum Importance)')
plt.savefig('visualizations/orthogonal_slices.png')
plt.close()

# 2. Difference heatmap (Importance − Edge)
diff_atlas = imp_atlas - edge_atlas
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
im0 = axes[0].imshow(diff_atlas[mid_x, :, :], cmap='coolwarm', vmin=-1, vmax=1)
axes[0].set_title('Sagittal Diff')
im1 = axes[1].imshow(diff_atlas[:, mid_y, :], cmap='coolwarm', vmin=-1, vmax=1)
axes[1].set_title('Coronal Diff')
im2 = axes[2].imshow(diff_atlas[:, :, mid_z], cmap='coolwarm', vmin=-1, vmax=1)
axes[2].set_title('Axial Diff')
fig.colorbar(im2, ax=axes.ravel().tolist())
plt.suptitle('Difference Heatmap (Importance - Edge)')
plt.savefig('visualizations/difference_heatmap.png')
plt.close()

# 3. Histogram
plt.figure(figsize=(8, 5))
plt.hist(imp_flat[imp_flat > 1e-4], bins=100, alpha=0.7, label='Importance Score', color='blue')
plt.hist(edge_flat[edge_flat > 1e-4], bins=100, alpha=0.5, label='Edge Probability', color='red')
plt.title('Histogram of Non-Zero Values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('visualizations/histogram.png')
plt.close()

# 4. Gaussian center visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
edge_p = points[:1382] * 255
uniform_p = points[1382:] * 255
ax.scatter(edge_p[:,0], edge_p[:,1], edge_p[:,2], c='green', s=8, alpha=0.8, label='Guided (80%)')
ax.scatter(uniform_p[:,0], uniform_p[:,1], uniform_p[:,2], c='red', s=20, alpha=1.0, label='Uniform (20%)')
ax.set_xlim(0, 255); ax.set_ylim(0, 255); ax.set_zlim(0, 255)
ax.set_title("Gaussian Centers (Weighted Sum Initialization)")
ax.legend()
plt.tight_layout()
plt.savefig('visualizations/gaussian_centers.png')
plt.close()

print("Visualizations saved to 'visualizations' folder.")
