import os
import sys
import torch
import time
import numpy as np

# Ensure code path is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dst_gs import CBCT_dataset_gs
from models.model import DIF_Gaussian
from utils import convert_cuda, load_config

def main():
    print("=" * 60)
    print("HYBRID IMPORTANCE-AWARE LOSS SMOKE TEST")
    print("=" * 60)

    # 1. Load config and model
    cfg = load_config("configs/default.yaml")
    model = DIF_Gaussian(cfg.model).cuda()
    model.train()


    # 2. Create dataset & loader
    dataset = CBCT_dataset_gs(
        dst_name="LUNA16",
        cfg=cfg.dataset,
        split="train",
        num_views=6,
        npoint=10000,
        out_res_scale=1.0
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # 3. Process one batch
    for item in loader:
        item = convert_cuda(item)
        
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        
        pred = model(item)
        
        importance = item.get('points_importance', torch.zeros_like(item['points_gt']))
        lambda_val = 1.0
        e = pred['points_pred'] - item['points_gt']
        sq_err = e ** 2
        weight = 1.0 + lambda_val * importance * (1.0 - torch.exp(-torch.abs(e.detach())))
        loss = torch.mean(weight * sq_err)

        
        # Test backward
        backward_succeeded = False
        try:
            loss.backward()
            backward_succeeded = True
        except Exception as err:
            print(f"Backward failed with error: {err}")
            backward_succeeded = False

        t1 = time.time()
        peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # Print statistics
        imp_np = importance.detach().cpu().numpy()
        w_np = weight.detach().cpu().numpy()
        
        min_w = float(w_np.min())
        max_w = float(w_np.max())
        mean_w = float(w_np.mean())
        loss_val = float(loss.item())

        print(f"\nVerification Metrics:")
        print(f"  - Minimum weight   : {min_w:.6f}  (Expected in [1.0, 2.0])")
        print(f"  - Maximum weight   : {max_w:.6f}  (Expected in [1.0, 2.0])")
        print(f"  - Mean weight      : {mean_w:.6f}")
        print(f"  - Loss value       : {loss_val:.6f}")
        print(f"  - Backward succeeds: {backward_succeeded}")
        print(f"  - Peak GPU memory  : {peak_gpu_mem:.2f} MB")
        print(f"  - Forward/backward : {(t1 - t0)*1000:.2f} ms")
        
        # Check numerical safety conditions
        cond1 = (min_w >= 1.0 and max_w <= 2.0001)
        cond2 = not np.isnan(w_np).any() and not np.isnan(loss_val)
        cond3 = (importance.grad_fn is None) # No graph attached to importance
        cond4 = backward_succeeded

        print("\nNumerical Safety Checks:")
        print(f"  [OK] Weights in [1.0, 2.0]     : {cond1}")
        print(f"  [OK] No NaNs                   : {cond2}")
        print(f"  [OK] No graph on importance    : {cond3}")
        print(f"  [OK] Gradient backward success : {cond4}")

        status = "PASS" if (cond1 and cond2 and cond3 and cond4) else "FAIL"
        print(f"\nFinal Result: {status}")
        print("=" * 60)
        break

if __name__ == "__main__":
    main()
