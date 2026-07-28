import os
import sys
import shutil
import argparse
import subprocess
from datetime import datetime
import yaml


def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def run_command(cmd, cwd, env_extra=None, desc=None):
    if desc:
        print(f"\n---> {desc}")
    print(f"Executing: {' '.join(cmd)} in [{cwd}]")
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    res = subprocess.run(cmd, cwd=cwd, env=env)
    if res.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {res.returncode}: {' '.join(cmd)}")
        sys.exit(res.returncode)
    return res


def backup_generated(generated_dir, project_root):
    if os.path.exists(generated_dir) and os.listdir(generated_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_root = os.path.join(project_root, "generated_archive")
        backup_dir = os.path.join(archive_root, f"backup_{timestamp}")
        os.makedirs(archive_root, exist_ok=True)
        print_header(f"ARCHIVE: Backing up existing generated/ artifacts to {backup_dir}")
        shutil.copytree(generated_dir, backup_dir)
        print("Backup complete. Proceeding with clean regeneration...")
        # Clear contents of generated_dir after backup
        for item in os.listdir(generated_dir):
            item_path = os.path.join(generated_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
    os.makedirs(os.path.join(generated_dir, "atlases"), exist_ok=True)
    os.makedirs(os.path.join(generated_dir, "points"), exist_ok=True)
    os.makedirs(os.path.join(generated_dir, "edges"), exist_ok=True)


def load_machine_config(root_dir):
    cfg_path = os.path.join(root_dir, "machine_config.yaml")
    if not os.path.isfile(cfg_path):
        print(f"[ERROR] machine_config.yaml not found at repository root: {root_dir}")
        sys.exit(1)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def main():
    parser = argparse.ArgumentParser(description="SCAGA Unified Pipeline & Onboarding CLI")
    parser.add_argument("--check", action="store_true", help="Run diagnostic repo_health_check.py")
    parser.add_argument("--preprocess", action="store_true", help="Run Preprocessing and TotalSegmentator edge/atlas/point sampling into generated/")
    parser.add_argument("--regenerate", action="store_true", help="Safely archive existing generated/ contents to generated_archive/ and regenerate from dataset")
    parser.add_argument("--train", action="store_true", help="Run model training with strict artifact verification")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on trained model checkpoints and generate CSV results")
    parser.add_argument("--smoke", action="store_true", help="Run quick numerical and loss gradient smoke test (run_smoke_loss.py)")
    parser.add_argument("--all", action="store_true", help="Run end-to-end pipeline (check -> preprocess -> train -> evaluate)")
    
    parser.add_argument("--experiment", type=str, default="Development",
                        help="Experiment target: 'Development', 'WeightedSum_SCAGA', 'SRAL_SCAGA', or custom name")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml",
                        help="Relative path to configuration YAML file inside active experiment folder")
    parser.add_argument("--epochs", type=int, default=400, help="Number of training epochs")
    parser.add_argument("--python_exec", type=str, default=sys.executable, help="Python interpreter path to invoke sub-stages")
    
    args = parser.parse_args()
    
    # Defaults if no command specified
    if not (args.check or args.preprocess or args.regenerate or args.train or args.evaluate or args.smoke or args.all):
        parser.print_help()
        sys.exit(0)

    project_root = os.path.dirname(os.path.abspath(__file__))
    machine_cfg = load_machine_config(project_root)
    generated_dir = os.path.abspath(os.path.join(project_root, machine_cfg.get("generated_dir", "./generated")))

    py_exe = args.python_exec

    # --------------------------------------------------------------------------
    # 1. HEALTH CHECK
    # --------------------------------------------------------------------------
    if args.check or args.all:
        print_header("STAGE: REPOSITORY HEALTH CHECK")
        run_command([py_exe, "repo_health_check.py"], cwd=project_root, desc="Verifying workspace environment")
        if not (args.preprocess or args.regenerate or args.train or args.evaluate or args.smoke or args.all):
            return

    # --------------------------------------------------------------------------
    # 2. REGENERATE / PREPROCESS
    # --------------------------------------------------------------------------
    if args.regenerate:
        backup_generated(generated_dir, project_root)

    if args.preprocess or args.regenerate or args.all:
        print_header("STAGE: DATASET PREPROCESSING & ARTIFACT GENERATION")
        
        # A. Raw Scans Preprocessing via run_preprocessing.ps1 (if on Windows)
        ps1_path = os.path.join(project_root, "run_preprocessing.ps1")
        if os.path.exists(ps1_path) and os.name == "nt":
            run_command(["powershell", "-ExecutionPolicy", "Bypass", "-File", ps1_path],
                        cwd=project_root, desc="Running scan image preprocessing (run_preprocessing.ps1)")

        # B. TotalSegmentator Edge Extraction
        totalseg_dir = os.path.join(project_root, "TotalSegmentator")
        if os.path.exists(os.path.join(totalseg_dir, "extract_edges.py")):
            run_command([py_exe, "extract_edges.py"], cwd=totalseg_dir,
                        env_extra={"PYTHONPATH": totalseg_dir}, desc="Extracting anatomical edge boundaries")
        
        # C. Probability Atlas Creation
        if os.path.exists(os.path.join(totalseg_dir, "create_atlas.py")):
            run_command([py_exe, "create_atlas.py"], cwd=totalseg_dir, desc="Generating edge probability atlas")

        # D. Importance Scores (if present)
        imp_dir = os.path.join(totalseg_dir, "Importance_Score")
        if os.path.exists(os.path.join(imp_dir, "compute_structural_complexity.py")):
            run_command([py_exe, os.path.join(imp_dir, "compute_structural_complexity.py")], cwd=totalseg_dir,
                        desc="Computing local 3D structural complexity / anisotropy")
        if os.path.exists(os.path.join(imp_dir, "compute_importance_score.py")):
            run_command([py_exe, os.path.join(imp_dir, "compute_importance_score.py")], cwd=totalseg_dir,
                        desc="Synthesizing weighted sum importance score atlas")

        # E. Gaussian Center Sampling
        if os.path.exists(os.path.join(imp_dir, "scaga_sampling.py")):
            run_command([py_exe, os.path.join(imp_dir, "scaga_sampling.py")], cwd=totalseg_dir,
                        desc="Sampling anisotropic Gaussian initialization points")
        elif os.path.exists(os.path.join(totalseg_dir, "edge_guided_sampling.py")):
            run_command([py_exe, "edge_guided_sampling.py"], cwd=totalseg_dir,
                        desc="Sampling edge-guided Gaussian initialization points")
        
        print_header(f"PREPROCESSING COMPLETED: Artifacts saved to {generated_dir}")

    # --------------------------------------------------------------------------
    # 3. SMOKE TEST
    # --------------------------------------------------------------------------
    if args.smoke:
        print_header("STAGE: NUMERICAL & LOSS SMOKE TEST")
        scaga_dir = os.path.join(project_root, "SCAGA")
        run_command([py_exe, "run_smoke_loss.py"], cwd=scaga_dir, env_extra={"PYTHONPATH": scaga_dir},
                    desc="Verifying Hybrid Importance-Aware Loss gradients & VRAM stability")

    # --------------------------------------------------------------------------
    # 4. RESOLVE EXPERIMENT REPOSITORY TARGET
    # --------------------------------------------------------------------------
    if args.experiment == "Development":
        target_dir = os.path.join(project_root, "SCAGA")
        exp_name = "SCAGA-Development"
        is_frozen = False
    elif args.experiment in ["WeightedSum_SCAGA", "SRAL_SCAGA"]:
        target_dir = os.path.join(project_root, "SCAGA", args.experiment)
        exp_name = args.experiment
        is_frozen = True
        if args.config == "configs/experiment.yaml":
            # Default fallback for frozen repos which use default.yaml or run_training.ps1
            args.config = "configs/default.yaml"
    else:
        target_dir = os.path.join(project_root, "SCAGA")
        exp_name = args.experiment
        is_frozen = False

    if not os.path.isdir(target_dir):
        print(f"[ERROR] Target experiment folder not found: {target_dir}")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # 5. TRAIN
    # --------------------------------------------------------------------------
    if args.train or args.all:
        print_header(f"STAGE: MODEL TRAINING [{exp_name}]")
        
        if not is_frozen:
            # Strict artifact verification before launching Development training
            pts_dir = os.path.join(generated_dir, "points")
            atl_dir = os.path.join(generated_dir, "atlases")
            pts_file = os.path.join(pts_dir, "sampled_points_weighted_sum.npy")
            if not os.path.isfile(pts_file):
                # Check directly in generated_dir if not in subfolder
                if os.path.isfile(os.path.join(generated_dir, "sampled_points_weighted_sum.npy")):
                    pts_file = os.path.join(generated_dir, "sampled_points_weighted_sum.npy")
                else:
                    print_header("STRICT RESOLUTION PRE-CHECK ERROR")
                    print(f"Required preprocessing artifact missing: {pts_file}")
                    print("Execute 'python run.py --preprocess' or 'python run.py --regenerate' first.")
                    sys.exit(1)

        train_cmd = [
            py_exe, "train.py",
            "--name", exp_name,
            "--batch_size", "1",
            "--epoch", str(args.epochs),
            "--dst_name", "LUNA16",
            "--num_views", "6",
            "--random_views",
            "--cfg_path", args.config
        ]
        if not is_frozen:
            train_cmd.extend(["--machine_config", os.path.join(project_root, "machine_config.yaml")])
        
        run_command(train_cmd, cwd=target_dir, env_extra={"PYTHONPATH": target_dir}, desc=f"Training {exp_name}")

    # --------------------------------------------------------------------------
    # 6. EVALUATE
    # --------------------------------------------------------------------------
    if args.evaluate or args.all:
        print_header(f"STAGE: MODEL EVALUATION [{exp_name}]")
        eval_cmd = [
            py_exe, "evaluate.py",
            "--name", exp_name,
            "--epoch", str(args.epochs),
            "--dst_name", "LUNA16",
            "--split", "test",
            "--num_views", "6",
            "--out_res_scale", "1.0",
            "--save_results",
            "--cfg_path", args.config
        ]
        if not is_frozen:
            eval_cmd.extend(["--machine_config", os.path.join(project_root, "machine_config.yaml")])
            
        run_command(eval_cmd, cwd=target_dir, env_extra={"PYTHONPATH": target_dir}, desc=f"Evaluating {exp_name}")

    print_header("CLI OPERATION COMPLETED SUCCESSFULLY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
