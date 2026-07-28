# Structural Complexity and Anisotropy Guided Attention (SCAGA)
### Advanced Sparse-View Cone-Beam CT Reconstruction via 3D Gaussian Splatting

This repository implements **SCAGA (Structural Complexity and Anisotropy Guided Attention)**, a framework for sparse-view Cone-Beam CT (CBCT) reconstruction built upon the DIF-Gaussian architecture. SCAGA introduces anatomically guided Gaussian initialization through structural complexity and anisotropy priors, together with multiple experimentally validated reconstruction objectives, including the Saturated Residual-Aware Loss (SRAL).

---

# Quick Start

The repository has been designed so that collaborators can set up and run experiments without modifying any Python source code. Follow the steps below to get started on a new machine.

### Step 1: Clone the Repository

```bash
git clone https://github.com/satwik-kun/SCAGA.git
cd SCAGA
```

### Step 2: Install Dependencies

Install the appropriate PyTorch build for your GPU and CUDA version by following the official PyTorch installation guide:

https://pytorch.org/get-started/locally/

Then install the remaining project dependencies:

```bash
pip install -r requirements.txt
```

### Step 3: Configure Your Machine (`machine_config.yaml`)

Open `machine_config.yaml` located in the repository root. This is the **only file** that should be modified when moving the project to a different machine.

```yaml
# Hardware Compute Configuration
device: "cuda:0"
num_workers: 2
batch_size: 1

# Local Filesystem Paths
dataset_root: "./Preprocessing/data"
output_root: "./outputs"
generated_dir: "./generated"
temporary_directory: "./tmp"
```

Update the following fields to match your local machine:

- `dataset_root` – Location of the dataset
- `output_root` – Directory for checkpoints and evaluation outputs
- `generated_dir` – Directory for preprocessing artifacts
- `temporary_directory` – Temporary working directory
- `device` – Compute device (`cuda:0` or `cpu`)
- `num_workers` – Number of PyTorch dataloader workers

No Python source files should require modification.

### Step 4: Verify the Repository

Run the repository health check:

```bash
python run.py --check
```

This verifies:

- Repository structure
- Python dependencies
- CUDA availability
- Dataset configuration
- Path resolution
- Runtime configuration
- Filesystem permissions

All checks should report **PASS** before continuing.

### Step 5: Run a Smoke Test

Before launching preprocessing or long training runs, verify that the repository is functioning correctly:

```bash
python run.py --smoke
```

This executes a lightweight forward and backward pass to verify numerical stability, gradient propagation, VRAM usage, and runtime configuration.

### Step 6: Generate Preprocessing Artifacts

If you are using a fresh dataset:

```bash
python run.py --preprocess
```

To safely regenerate preprocessing artifacts from scratch:

```bash
python run.py --regenerate
```

### Step 7: Train the Model

```bash
python run.py --train
```

### Step 8: Evaluate the Model

```bash
python run.py --evaluate
```

### Recommended Workflow

For a fresh machine or a new dataset, the recommended execution order is:

1. Clone the repository.
2. Install dependencies.
3. Configure `machine_config.yaml`.
4. Run `python run.py --check`.
5. Run `python run.py --smoke`.
6. Generate preprocessing artifacts (`--preprocess` or `--regenerate`).
7. Train the model (`--train`).
8. Evaluate the trained model (`--evaluate`).

Following this sequence ensures the repository is correctly configured before launching long-running experiments.

---

# Unified Command Line Interface (`run.py`)

The repository is controlled through a single entry point (`run.py`). Most users will only need the following commands.

| Command | Purpose | Typical Usage |
| :--- | :--- | :--- |
| `python run.py --check` | Verify repository and environment | First run after cloning |
| `python run.py --preprocess` | Generate preprocessing artifacts | New dataset |
| `python run.py --regenerate` | Archive and regenerate preprocessing artifacts | Switching datasets |
| `python run.py --train` | Train the reconstruction model | Main experiment |
| `python run.py --evaluate` | Evaluate trained checkpoints | After training |
| `python run.py --smoke` | Verify pipeline correctness | Before overnight runs |
| `python run.py --all` | Run the complete workflow | End-to-end automation |

### Selecting & Customizing Experiments

You can tailor experiment parameters without modifying Python source code by editing `SCAGA/configs/experiment.yaml`.

```bash
# Example training command
python run.py --train

# Example evaluation command
python run.py --evaluate
```

---

# Experiment Structure & Snapshots

The repository separates active algorithmic development from validated, frozen historical baselines.

```
├── machine_config.yaml
├── run.py
├── repo_health_check.py
├── run_full_pipeline.ps1
├── Preprocessing/
├── TotalSegmentator/
├── generated/
│   ├── atlases/
│   ├── points/
│   └── edges/
├── outputs/
└── SCAGA/
    ├── configs/
    ├── train.py
    ├── evaluate.py
    ├── utils.py
    ├── datasets/
    ├── models/
    ├── WeightedSum_SCAGA/
    └── SRAL_SCAGA/
```

### Frozen Verified Experiment Snapshots

#### 1. `SCAGA/WeightedSum_SCAGA/`

- **Formulation:** Weighted Sum Gaussian initialization with Pure MSE Loss.
- **Representative Benchmark (10-volume subset):**
  - PSNR: **23.1640 dB**
  - SSIM: **0.6976**
  - Edge F1: **0.2197**
- **Status:** Frozen, self-contained, and never modified automatically.

#### 2. `SCAGA/SRAL_SCAGA/`

- **Formulation:** Weighted Sum Gaussian initialization with Saturated Residual-Aware Loss (SRAL).
- **Representative Benchmark (10-volume subset):**
  - PSNR: **23.2145 dB**
  - SSIM: **0.7511**
  - Edge F1: **0.1969**
- **Status:** Frozen, self-contained, and independently executable.

---

# Strict Artifact Resolution

To ensure deterministic and reproducible experiments, the active development pipeline (`SCAGA/`) no longer performs silent fallbacks.

- Preprocessing artifacts are loaded **only** from the directory specified by `generated_dir` in `machine_config.yaml`.
- Historical artifacts are never substituted automatically.
- Missing preprocessing files generate explicit error messages explaining exactly which preprocessing stage must be executed.

If you encounter a **STRICT RESOLUTION ERROR**:

1. Verify `dataset_root` inside `machine_config.yaml`.
2. Generate preprocessing artifacts:

```bash
python run.py --preprocess
```

3. If switching datasets, archive previous outputs and regenerate:

```bash
python run.py --regenerate
```

---

# Need Help?

Run

```bash
python run.py --check
```

The diagnostic utility automatically checks repository integrity, dependencies, CUDA configuration, filesystem permissions, dataset paths, preprocessing artifacts, and runtime configuration, providing actionable guidance if any issue is detected.

---

# Reference

This work builds upon **DIF-Gaussian** (Lin et al., MICCAI 2024).

The SCAGA framework extends the original architecture with structural complexity-guided Gaussian initialization, anatomically informed sampling strategies, and experimentally validated reconstruction objectives for sparse-view Cone-Beam CT reconstruction.