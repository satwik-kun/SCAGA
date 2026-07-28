import os
import sys
import shutil
import importlib

def run_health_check():
    print("=" * 70)
    print("SCAGA REPOSITORY HEALTH CHECK & ONBOARDING DIAGNOSTIC")
    print("=" * 70)

    root_dir = os.path.dirname(os.path.abspath(__file__))
    results = []
    errors = []

    def log_check(name, passed, msg="", error_advice=""):
        status = "PASS" if passed else "FAIL"
        symbol = "OK" if passed else "XX"
        print(f" [{symbol:2}] {name:<35} : {status} {msg}")
        results.append(passed)
        if not passed and error_advice:
            errors.append((name, error_advice))

    # --------------------------------------------------------------------------
    # 1. Check machine_config.yaml
    # --------------------------------------------------------------------------
    machine_cfg_path = os.path.join(root_dir, "machine_config.yaml")
    cfg = None
    if os.path.isfile(machine_cfg_path):
        try:
            import yaml
            with open(machine_cfg_path, 'r') as f:
                cfg = yaml.safe_load(f)
            log_check("machine_config.yaml exists", True, f"({machine_cfg_path})")
        except Exception as e:
            log_check("machine_config.yaml exists", False, f"Failed to parse YAML: {e}",
                      "Check machine_config.yaml formatting for valid YAML syntax.")
    else:
        log_check("machine_config.yaml exists", False, "Not found!",
                  f"Create machine_config.yaml at repository root ({root_dir}).")

    # --------------------------------------------------------------------------
    # 2. Check active experiment configuration exists
    # --------------------------------------------------------------------------
    exp_cfg_path = os.path.join(root_dir, "SCAGA", "configs", "experiment.yaml")
    exp_cfg = None
    if os.path.isfile(exp_cfg_path):
        try:
            with open(exp_cfg_path, 'r') as f:
                exp_cfg = yaml.safe_load(f)
            log_check("experiment configuration exists", True, "(SCAGA/configs/experiment.yaml)")
        except Exception as e:
            log_check("experiment configuration exists", False, f"YAML parse error: {e}",
                      "Fix syntax in SCAGA/configs/experiment.yaml.")
    else:
        log_check("experiment configuration exists", False, "Not found!",
                  "Create SCAGA/configs/experiment.yaml with experiment parameters.")

    # --------------------------------------------------------------------------
    # 3. Check relative path resolution
    # --------------------------------------------------------------------------
    resolved_dataset_root = None
    resolved_generated_dir = None
    resolved_output_root = None
    if cfg:
        try:
            resolved_dataset_root = os.path.abspath(os.path.join(root_dir, cfg.get("dataset_root", "./Preprocessing/data")))
            resolved_generated_dir = os.path.abspath(os.path.join(root_dir, cfg.get("generated_dir", "./generated")))
            resolved_output_root = os.path.abspath(os.path.join(root_dir, cfg.get("output_root", "./outputs")))
            log_check("relative path resolution", True, "(Successfully resolved absolute paths)")
        except Exception as e:
            log_check("relative path resolution", False, str(e),
                      "Ensure machine_config.yaml path entries are valid relative or absolute paths.")
    else:
        log_check("relative path resolution", False, "Skipped (machine_config missing)",
                  "Fix machine_config.yaml first.")

    # --------------------------------------------------------------------------
    # 4. Check dataset_root exists
    # --------------------------------------------------------------------------
    if resolved_dataset_root and os.path.exists(resolved_dataset_root):
        log_check("dataset_root exists", True, f"({resolved_dataset_root})")
    else:
        log_check("dataset_root exists", False, f"Not found: {resolved_dataset_root}",
                  "Update dataset_root in machine_config.yaml to point to your CT dataset directory.")

    # --------------------------------------------------------------------------
    # 5. Check generated directory exists (or can be created)
    # --------------------------------------------------------------------------
    if resolved_generated_dir:
        try:
            os.makedirs(resolved_generated_dir, exist_ok=True)
            os.makedirs(os.path.join(resolved_generated_dir, "atlases"), exist_ok=True)
            os.makedirs(os.path.join(resolved_generated_dir, "points"), exist_ok=True)
            os.makedirs(os.path.join(resolved_generated_dir, "edges"), exist_ok=True)
            log_check("generated directory ready", True, f"({resolved_generated_dir})")
        except Exception as e:
            log_check("generated directory ready", False, str(e),
                      f"Check directory write permissions for {resolved_generated_dir}.")
    else:
        log_check("generated directory ready", False, "Skipped", "Fix machine_config.yaml.")

    # --------------------------------------------------------------------------
    # 6. Check output directory & checkpoint directory
    # --------------------------------------------------------------------------
    if resolved_output_root:
        try:
            os.makedirs(resolved_output_root, exist_ok=True)
            ckpt_dir = os.path.join(resolved_output_root, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            log_check("output & checkpoint dirs ready", True, f"({resolved_output_root})")
        except Exception as e:
            log_check("output & checkpoint dirs ready", False, str(e),
                      "Ensure appropriate permissions for creating output directories.")
    else:
        log_check("output & checkpoint dirs ready", False, "Skipped", "Fix machine_config.yaml.")

    # --------------------------------------------------------------------------
    # 7. Check write permissions
    # --------------------------------------------------------------------------
    can_write = True
    for test_dir in [resolved_generated_dir, resolved_output_root]:
        if test_dir and os.path.exists(test_dir):
            test_file = os.path.join(test_dir, ".permission_test")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                can_write = False
    if can_write:
        log_check("write permissions verified", True, "(Generated & outputs are writable)")
    else:
        log_check("write permissions verified", False, "Permission denied!",
                  "Grant write permissions to the repository directory and output targets.")

    # --------------------------------------------------------------------------
    # 8. Check PyTorch installation
    # --------------------------------------------------------------------------
    torch_installed = False
    try:
        import torch
        torch_installed = True
        log_check("PyTorch installation", True, f"(v{torch.__version__})")
    except ImportError:
        log_check("PyTorch installation", False, "Module 'torch' not found!",
                  "Install PyTorch: pip install torch torchvision torchaudio (or consult PyTorch documentation).")

    # --------------------------------------------------------------------------
    # 9. Check CUDA availability
    # --------------------------------------------------------------------------
    if torch_installed:
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            log_check("CUDA availability", True, f"({torch.cuda.get_device_name(0)})")
        else:
            log_check("CUDA availability", False, "CUDA not available (will fallback to CPU / slower)",
                      "Ensure Nvidia GPU drivers and CUDA toolkit match your PyTorch wheel, or set device to 'cpu' in machine_config.yaml if testing CPU-only.")
    else:
        log_check("CUDA availability", False, "Skipped (no PyTorch)", "Install PyTorch with CUDA support.")

    # --------------------------------------------------------------------------
    # 10. Check required Python packages
    # --------------------------------------------------------------------------
    required_pkgs = ["numpy", "nibabel", "skimage", "tqdm", "yaml", "psutil"]
    missing_pkgs = []
    for pkg in required_pkgs:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing_pkgs.append(pkg)
    if not missing_pkgs:
        log_check("required Python packages", True, f"({', '.join(required_pkgs)})")
    else:
        log_check("required Python packages", False, f"Missing: {', '.join(missing_pkgs)}",
                  f"Install missing packages: pip install {' '.join(missing_pkgs).replace('skimage', 'scikit-image').replace('yaml', 'pyyaml')}")

    # --------------------------------------------------------------------------
    # 11. Check preprocessing prerequisites
    # --------------------------------------------------------------------------
    totalseg_folder = os.path.join(root_dir, "TotalSegmentator")
    if os.path.isdir(totalseg_folder):
        log_check("preprocessing prerequisites", True, "(TotalSegmentator module present)")
    else:
        log_check("preprocessing prerequisites", False, "TotalSegmentator folder missing!",
                  "Ensure the repository was cloned completely including the TotalSegmentator directory.")

    # --------------------------------------------------------------------------
    # 12. Check active experiment configuration validity
    # --------------------------------------------------------------------------
    if exp_cfg and "loss" in exp_cfg and "model" in exp_cfg and "initialization" in exp_cfg:
        log_check("active experiment config valid", True, f"(Loss: {exp_cfg['loss'].get('loss_type', 'mse')})")
    else:
        log_check("active experiment config valid", False, "Missing core sections in experiment.yaml!",
                  "Ensure experiment.yaml specifies 'loss', 'model', and 'initialization' sections.")

    print("=" * 70)
    if all(results):
        print(" FINAL REPORT: PASS - REPOSITORY IS HEALTHY & READY FOR COLLABORATION")
        print("=" * 70)
        return 0
    else:
        print(" FINAL REPORT: FAIL - REPOSITORY HEALTH CHECK DETECTED ISSUES")
        print("=" * 70)
        print("\nACTIONABLE ERROR REMEDIATION GUIDE:")
        for name, advice in errors:
            print(f" * [{name}]:\n     -> {advice}")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(run_health_check())
