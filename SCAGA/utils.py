import os
import yaml
import numpy as np
import SimpleITK as sitk
from easydict import EasyDict


def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return EasyDict(cfg)


def find_machine_config(start_dir=None):
    if start_dir is None:
        start_dir = os.path.dirname(os.path.abspath(__file__))
    curr = os.path.abspath(start_dir)
    while True:
        candidate = os.path.join(curr, "machine_config.yaml")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(curr)
        if parent == curr:
            raise FileNotFoundError("Could not find machine_config.yaml in workspace roots.")
        curr = parent


def load_runtime_config(exp_cfg_path, machine_cfg_path=None):
    """
    Loads experiment config and merges with machine_config.yaml.
    All relative paths in machine_config are resolved absolute relative to its location.
    """
    if not machine_cfg_path or not os.path.isfile(machine_cfg_path):
        machine_cfg_path = find_machine_config()
    
    with open(machine_cfg_path, 'r') as f:
        m_cfg = yaml.safe_load(f) or {}
    with open(exp_cfg_path, 'r') as f:
        exp_cfg = yaml.safe_load(f) or {}
    
    m_dir = os.path.dirname(os.path.abspath(machine_cfg_path))
    dataset_root = os.path.abspath(os.path.join(m_dir, m_cfg.get("dataset_root", "./Preprocessing/data")))
    generated_dir = os.path.abspath(os.path.join(m_dir, m_cfg.get("generated_dir", "./generated")))
    output_root = os.path.abspath(os.path.join(m_dir, m_cfg.get("output_root", "./outputs")))
    temp_dir = os.path.abspath(os.path.join(m_dir, m_cfg.get("temporary_directory", "./tmp")))

    if "dataset" not in exp_cfg:
        exp_cfg["dataset"] = {}
    
    exp_cfg["dataset"]["root_dir"] = dataset_root
    exp_cfg["dataset"]["generated_dir"] = generated_dir
    exp_cfg["output_root"] = output_root
    exp_cfg["temporary_directory"] = temp_dir
    exp_cfg["num_workers"] = m_cfg.get("num_workers", 2)
    exp_cfg["device"] = m_cfg.get("device", "cuda:0")
    
    # Preserve backwards compatibility for dataset resolutions in default configurations
    if "gs" in exp_cfg.get("model", {}):
        exp_cfg["dataset"]["gs_res"] = exp_cfg["model"]["gs"].get("res", 12)
    elif "initialization" in exp_cfg:
        exp_cfg["dataset"]["gs_res"] = exp_cfg["initialization"].get("gs_res", 12)
        
    return EasyDict(exp_cfg)


def resolve_generated_artifact(filename, generated_dir=None, subfolder=None, expected_stage=None):
    """
    STRICT ARTIFACT RESOLUTION (NO SILENT FALLBACKS).
    Resolves preprocessing artifacts ONLY from machine_config.generated_dir.
    If a required artifact does not exist, immediately terminates with clear FileNotFoundError.
    """
    if generated_dir is None:
        m_path = find_machine_config()
        with open(m_path, 'r') as f:
            m_cfg = yaml.safe_load(f) or {}
        m_dir = os.path.dirname(os.path.abspath(m_path))
        generated_dir = os.path.abspath(os.path.join(m_dir, m_cfg.get("generated_dir", "./generated")))

    candidates = []
    if subfolder:
        candidates.append(os.path.join(generated_dir, subfolder, filename))
    candidates.append(os.path.join(generated_dir, filename))
    
    for c in candidates:
        if os.path.isfile(c):
            return c
            
    expected_location = candidates[0] if subfolder else os.path.join(generated_dir, filename)
    stage_msg = expected_stage if expected_stage else "python run.py --preprocess"
    
    err_msg = (
        f"\n======================================================================\n"
        f"[STRICT RESOLUTION ERROR] Required preprocessing artifact not found!\n"
        f"======================================================================\n"
        f" * Missing Filename : {filename}\n"
        f" * Expected Location: {expected_location}\n"
        f" * Required Stage   : Execute '{stage_msg}' to cleanly generate this file.\n"
        f"======================================================================\n"
        f"NOTE: Silent fallbacks are strictly disabled in active development to guarantee\n"
        f"deterministic and reproducible experiments. Never substitute another file."
    )
    raise FileNotFoundError(err_msg)


def convert_cuda(item):
    for key in item.keys():
        if key not in ['name', 'dst_name']:
            item[key] = item[key].float().cuda()
    return item


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def sitk_load(path, uint8=False, spacing_unit='mm'):
    # load as float32
    itk_img = sitk.ReadImage(path)
    spacing = np.array(itk_img.GetSpacing(), dtype=np.float32)
    origin = np.array(itk_img.GetOrigin(), dtype=np.float32)
    if spacing_unit == 'm':
        spacing *= 1000.
        origin *= 1000
    elif spacing_unit != 'mm':
        raise ValueError
    image = sitk.GetArrayFromImage(itk_img)
    image = image.transpose(2, 1, 0) # to [x, y, z]
    image = image.astype(np.float32)
    if uint8:
        # if data is saved as uint8, [0, 255] => [0, 1]
        image /= 255.
    return image, spacing, origin


def sitk_save(path, image, spacing=None, origin=None, uint8=False):
    # default: float32 (input)
    image = image.astype(np.float32)
    image = image.transpose(2, 1, 0)
    if uint8:
        # value range should be [0, 1]
        image = (image * 255).astype(np.uint8)
    out = sitk.GetImageFromArray(image)
    if spacing is not None:
        out.SetSpacing(spacing.astype(np.float64)) # unit: mm
    if origin is not None:
        out.SetOrigin(origin.astype(np.float64)) # unit: mm
    sitk.WriteImage(out, path)
