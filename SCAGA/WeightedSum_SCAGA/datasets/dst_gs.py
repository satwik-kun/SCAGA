import os
import numpy as np
from copy import deepcopy

from datasets.base import CBCT_dataset


# ============================================================
# INITIALIZATION FOR FAST BASELINE EXPERIMENT (SCAGA-aniso)
# ============================================================

POINTS_FILE = "sampled_points_weighted_sum.npy"

# ============================================================


class CBCT_dataset_gs(CBCT_dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ----------------------------------------------------
        # Load pre-generated Gaussian initialization
        # ----------------------------------------------------

        points_path = os.path.join(
            os.path.dirname(__file__),
            POINTS_FILE
        )

        if not os.path.isfile(points_path):
            # Fallback to standard sampled_points.npy if anisotropy file is not found
            points_path = os.path.join(
                os.path.dirname(__file__),
                "sampled_points.npy"
            )

        if not os.path.isfile(points_path):
            raise FileNotFoundError(
                f"\nGaussian initialization not found:\n{points_path}"
            )

        self.points_gs = np.load(points_path).astype(np.float32)

        expected = self.cfg.gs_res ** 3

        # ----------------------------------------------------
        # Load Weighted Sum Importance Atlas for Loss Supervision
        # ----------------------------------------------------

        atlas_path = os.path.join(
            os.path.dirname(__file__),
            "importance_score_weighted_sum.nii.gz"
        )

        if os.path.isfile(atlas_path):
            import nibabel as nib
            self.importance_atlas = nib.load(atlas_path).get_fdata().astype(np.float32)
        else:
            self.importance_atlas = None

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        assert self.points_gs.shape == (expected, 3), \
            f"Expected ({expected}, 3), got {self.points_gs.shape}"

        assert np.all(self.points_gs >= 0.0), \
            "Gaussian coordinates contain values below 0."

        assert np.all(self.points_gs <= 1.0), \
            "Gaussian coordinates contain values above 1."

        unique = np.unique(self.points_gs, axis=0)

        assert len(unique) == expected, \
            "Duplicate Gaussian centers detected."

        # ----------------------------------------------------
        # Information
        # ----------------------------------------------------

        print("\n" + "=" * 70)
        print("Gaussian Initialization")
        print("=" * 70)

        print(f"Initialization File : {POINTS_FILE}")
        print(f"Loaded From         : {points_path}")

        print()

        print(f"Gaussian Centers    : {len(self.points_gs)}")
        print(f"Shape               : {self.points_gs.shape}")

        print()

        print("Coordinate Range")
        print(f"Min : {self.points_gs.min(axis=0)}")
        print(f"Max : {self.points_gs.max(axis=0)}")

        print("=" * 70)

    def __getitem__(self, index):

        data_dict = super().__getitem__(index)

        # ----------------------------------------------------
        # Project Gaussian centers into projection images
        # ----------------------------------------------------

        points_gs = deepcopy(self.points_gs)

        points_gs_proj = self.project_points(
            points_gs,
            data_dict["angles"]
        )

        data_dict.update({
            "points_gs": points_gs,
            "points_gs_proj": points_gs_proj
        })

        # ----------------------------------------------------
        # Sample Importance Score for Training Points
        # ----------------------------------------------------

        if self.is_train and self.importance_atlas is not None:
            from scipy.ndimage import map_coordinates
            pts = data_dict['points']  # [N, 3] in [0, 1]
            grid_coords = pts * (np.array(self.importance_atlas.shape) - 1.0)
            imp_vals = map_coordinates(self.importance_atlas, grid_coords.T, order=1, mode='nearest').astype(np.float32)
            data_dict['points_importance'] = imp_vals[None, :]  # [1, N]
        else:
            data_dict['points_importance'] = np.zeros((1, data_dict['points'].shape[0]), dtype=np.float32)

        return data_dict

