import os
import numpy as np
from copy import deepcopy

from utils import resolve_generated_artifact
from datasets.base import CBCT_dataset


# ============================================================
# INITIALIZATION FOR FAST BASELINE EXPERIMENT (SCAGA-aniso)
# ============================================================

POINTS_FILE = "sampled_points_weighted_sum.npy"
ATLAS_FILE = "importance_score_weighted_sum.nii.gz"

# ============================================================


class CBCT_dataset_gs(CBCT_dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ----------------------------------------------------
        # Load pre-generated Gaussian initialization (STRICT RESOLUTION)
        # ----------------------------------------------------

        points_filename = getattr(self.cfg, "points_file", POINTS_FILE)
        gen_dir = getattr(self.cfg, "generated_dir", None)

        points_path = resolve_generated_artifact(
            filename=points_filename,
            generated_dir=gen_dir,
            subfolder="points",
            expected_stage="python run.py --preprocess (Stage 4: Gaussian Sampling)"
        )

        self.points_gs = np.load(points_path).astype(np.float32)
        expected = self.cfg.gs_res ** 3

        # ----------------------------------------------------
        # Load Importance Atlas for Loss Supervision (STRICT RESOLUTION)
        # ----------------------------------------------------

        if getattr(self, 'is_train', True):
            atlas_filename = getattr(self.cfg, "atlas_file", ATLAS_FILE)
            atlas_path = resolve_generated_artifact(
                filename=atlas_filename,
                generated_dir=gen_dir,
                subfolder="atlases",
                expected_stage="python run.py --preprocess (Stage 3: Atlas & Importance Score Calculation)"
            )
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

