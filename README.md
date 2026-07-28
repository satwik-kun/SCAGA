# Structural Complexity and Anisotropy Guided Attention (SCAGA)
### Advanced Sparse-View Cone-Beam CT Reconstruction via 3D Gaussian Splatting

This repository implements the **SCAGA** framework, extending anisotropic 3D Gaussian Splatting (DIF-Gaussian) with anatomical edge guidance, structural complexity atlases, and Hybrid Saturated Residual-Aware Loss for extremely sparse-view Cone-Beam CT (CBCT) reconstruction.

---

## 🚀 Teammate Quickstart & Onboarding

This codebase is designed for seamless team collaboration across different machines, graphics cards, and operating systems. **You do not need to modify any Python source code to configure paths or run pipelines.**

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd SCAGA-Repo
```

### Step 2: Configure Your Machine (`machine_config.yaml`)
Open `machine_config.yaml` located in the root directory. Modify this **single file** to set your local dataset location, compute device, and output paths:

```yaml
# Hardware Compute Configuration
device: "cuda:0"          # Compute device ('cuda:0' or 'cpu')
num_workers: 2            # Number of parallel PyTorch dataloader workers
batch_size: 1             # Default batch size (keep 1 for sparse-view memory safety)

# Local Filesystem Paths
dataset_root: "./Preprocessing/data"    # Absolute or relative path to raw/processed dataset
output_root: "./outputs"                # Destination for training checkpoints and CSV logs
generated_dir: "./generated"            # Destination for generated preprocessing atlases & coordinates
temporary_directory: "./tmp"            # Temp folder for processing
```

### Step 3: Run the Repository Health Check
Verify your environment, Python package dependencies, PyTorch CUDA readiness, write permissions, and relative paths by running:

```bash
python run.py --check
```

If any check fails, the diagnostic utility will print an **Actionable Remediation Guide** explaining exactly how to fix your environment before starting computationally expensive runs.

---

## 🛠️ Unified Command Line Interface (`run.py`)

All pipeline operations, diagnostics, training runs, and evaluations are controlled through the global entry point `run.py`.

| Command | Description |
| :--- | :--- |
| `python run.py --check` | Runs `repo_health_check.py` diagnostic tool to verify workspace and environment integrity. |
| `python run.py --preprocess` | Executes CT image processing, TotalSegmentator boundary extraction, probability atlas creation, and Gaussian center sampling into `generated/`. |
| `python run.py --regenerate` | Safely archives existing `generated/` contents into timestamped backups under `generated_archive/` before regenerating cleanly from the dataset. |
| `python run.py --train` | Validates preprocessing artifacts via strict resolution and trains the neural reconstruction model for 400 epochs. |
| `python run.py --evaluate` | Loads the trained checkpoint from `outputs/`, runs evaluation over test scans, and writes PSNR, SSIM, and 3D Edge F1 CSV reports. |
| `python run.py --smoke` | Runs a 1-batch development smoke test (`SCAGA/run_smoke_loss.py`) verifying numerical stability, VRAM usage, and gradient backpropagation. |
| `python run.py --all` | End-to-end automation execution: Check ➔ Preprocess ➔ Train ➔ Evaluate. |

### Selecting & Customizing Experiments
You can tailor parameters without modifying Python source code by adjusting `SCAGA/configs/experiment.yaml` and passing `--experiment` or `--config` flags:

```bash
# Run training on active Development branch with custom epochs
python run.py --train --experiment Development --epochs 400

# Evaluate a specific experiment target
python run.py --evaluate --experiment WeightedSum_SCAGA
```

---

## 🧪 Experiment Structure & Snapshots

The repository separates active algorithmic development from validated, frozen historical baselines:

```
├── machine_config.yaml              # Global hardware and path configuration (USER EDITABLE)
├── run.py                           # Global unified automation CLI
├── repo_health_check.py             # Diagnostic and onboarding utility
├── run_full_pipeline.ps1            # Portable Windows PowerShell automation launcher
├── Preprocessing/                   # Dataset ingestion and raw projection generation
├── TotalSegmentator/                # Anatomical edge segmentation and importance atlases
├── generated/                       # Clean, dedicated directory for new preprocessing outputs
│   ├── atlases/                     # Generated importance & complexity atlases (.nii.gz)
│   ├── points/                      # Generated Gaussian point coordinates (.npy)
│   └── edges/                       # Intermediate binary edge volumes
├── outputs/                         # Run logs, model weights (.pth), and evaluation CSVs
└── SCAGA/
    ├── configs/experiment.yaml      # Active algorithmic hyperparameters
    ├── train.py / evaluate.py / utils.py / models/ / datasets/
    ├── WeightedSum_SCAGA/           # [FROZEN] Self-contained verified experiment snapshot
    └── SRAL_SCAGA/                  # [FROZEN] Self-contained verified experiment snapshot
```

### Frozen Verified Experiment Benchmarks

1. **`SCAGA/WeightedSum_SCAGA/` (Weighted Sum SCAGA)**
   - **Formulation**: Anisotropic structural complexity mixed sampling (80% Guided / 20% Uniform) with Pure MSE Loss.
   - **Verified Performance**: **23.1640 dB PSNR**, **0.6729 SSIM**, **0.1333 Edge F1**.
   - **Isolation Guarantee**: Completely self-contained repository with independent training, dataset, and evaluation code. **Never modified automatically.**

2. **`SCAGA/SRAL_SCAGA/` (Saturated Residual-Aware SCAGA)**
   - **Formulation**: Hybrid Saturated Residual-Aware Loss ($L = \text{mean}((1 + \lambda I \cdot (1 - \exp(-|e|))) \cdot e^2)$).
   - **Verified Performance**: **23.1702 dB PSNR**, **0.6730 SSIM**, **0.1334 Edge F1**.
   - **Isolation Guarantee**: Independently executable snapshot preserving verified scientific findings.

---

## 🔒 Strict Artifact Resolution & No Silent Fallbacks

To ensure absolute determinism and experimental reproducibility across teams, **all silent fallback mechanisms have been removed** from the active Development pipeline (`SCAGA/`).

- Preprocessing artifacts (such as `sampled_points_weighted_sum.npy` or `importance_score_weighted_sum.nii.gz`) are loaded **exclusively** from the directory defined in `machine_config.yaml` under `generated_dir`.
- The pipeline will **never** automatically substitute historical data, fallback to different atlases, or silently search arbitrary legacy directories.

### Troubleshooting: `[STRICT RESOLUTION ERROR] Required preprocessing artifact not found!`
If you encounter a `FileNotFoundError` detailing a strict resolution failure when starting `train.py` or `evaluate.py`:
1. Check that your `dataset_root` in `machine_config.yaml` points to your active CT scan folder.
2. Run artifact generation cleanly to populate `generated/`:
   ```bash
   python run.py --preprocess
   ```
3. If switching to a new dataset size, safely archive old outputs and regenerate:
   ```bash
   python run.py --regenerate
   ```

---

## 📚 Reference & License
This work builds upon DIF-Gaussian (Lin et al., MICCAI 2024). Please adhere to team archival best practices when pushing changes and ensure all diagnostic health checks pass prior to submitting code reviews.
