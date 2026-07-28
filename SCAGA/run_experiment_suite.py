import os
import sys
import time
import subprocess
import csv
import re

def run_experiment():
    print("="*80)
    print("STARTING OFFICIAL EXPERIMENT: SCAGA-weighted_sum_saturated_residual_loss-6v-subset0")
    print("="*80)

    exp_name = "SCAGA-weighted_sum_saturated_residual_loss-6v-subset0"
    epochs = 100
    num_views = 6

    # 1. Run Training
    print(f"\n[1/2] Running Training for {epochs} epochs...")
    train_cmd = [
        sys.executable, "train.py",
        "--name", exp_name,
        "--dst_name", "LUNA16",
        "--num_views", str(num_views),
        "--epoch", str(epochs),
        "--cfg_path", "configs/default.yaml",
        "--checkpoint_interval", "10",
        "--eval_interval", "0", # Only evaluate at final checkpoint for maximum training throughput
        "--lambda_importance", "1.0",
        "--loss_type", "saturated_residual"
    ]

    
    t_start = time.time()
    train_proc = subprocess.run(train_cmd, capture_output=True, text=True)
    t_train = time.time() - t_start
    
    if train_proc.returncode != 0:
        print("Training failed with return code:", train_proc.returncode)
        print("STDOUT:", train_proc.stdout)
        print("STDERR:", train_proc.stderr)
        return

    print("Training Completed Successfully!")
    
    # Extract Peak GPU and CPU memory from train output
    peak_gpu = "Unknown"
    peak_cpu = "Unknown"
    for line in train_proc.stdout.splitlines():
        if "Peak GPU memory:" in line:
            peak_gpu = line.split(":")[-1].strip()
        if "Peak CPU memory:" in line:
            peak_cpu = line.split(":")[-1].strip()
            
    print(f"  - Total Training Time : {t_train:.2f} seconds ({t_train/60:.2f} minutes)")
    print(f"  - Peak GPU Memory     : {peak_gpu}")
    print(f"  - Peak CPU Memory     : {peak_cpu}")

    # 2. Run Evaluation
    print(f"\n[2/2] Running Comprehensive Evaluation on 10 Full Test Volumes...")
    eval_cmd = [
        sys.executable, "evaluate.py",
        "--name", exp_name,
        "--dst_name", "LUNA16",
        "--num_views", str(num_views),
        "--epoch", str(epochs),
        "--cfg_path", f"./logs/{exp_name}/config.yaml",
        "--save_results"
    ]
    
    e_start = time.time()
    eval_proc = subprocess.run(eval_cmd, capture_output=True, text=True)
    t_eval = time.time() - e_start

    if eval_proc.returncode != 0:
        print("Evaluation failed with return code:", eval_proc.returncode)
        print("STDOUT:", eval_proc.stdout)
        print("STDERR:", eval_proc.stderr)
        return

    print("Evaluation Completed Successfully!")
    print(f"  - Total Evaluation Time: {t_eval:.2f} seconds")

    # 3. Read Metrics from CSV
    csv_path = f"./logs/{exp_name}/results/ep_{epochs}/results_1.0x.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Could not find evaluation results CSV at {csv_path}")
        return

    psnr, ssim, edge_f1, edge_precision, edge_recall = 0, 0, 0, 0, 0
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['obj_id'] == 'average':
                psnr = float(row['psnr'])
                ssim = float(row['ssim'])
                edge_f1 = float(row['edge_f1'])
                edge_precision = float(row['edge_precision'])
                edge_recall = float(row['edge_recall'])

    # Historical Benchmark Numbers
    benchmarks = {
        "Baseline": {
            "init": "Uniform Grid", "loss": "Pure MSE",
            "psnr": 22.2568, "ssim": 0.7829, "f1": 0.1679, "prec": 0.2988, "rec": 0.1256
        },
        "Product SCAGA": {
            "init": "Edge x Aniso", "loss": "Pure MSE",
            "psnr": 22.4476, "ssim": 0.7440, "f1": 0.1824, "prec": 0.2563, "rec": 0.1517
        },
        "Weighted Sum SCAGA": {
            "init": "Weighted Sum", "loss": "Pure MSE",
            "psnr": 23.1640, "ssim": 0.6976, "f1": 0.2197, "prec": 0.2745, "rec": 0.1910
        },
        "Weighted Sum SCAGA + Imp. Loss": {
            "init": "Weighted Sum", "loss": "Imp-Aware",
            "psnr": 22.2040, "ssim": 0.7319, "f1": 0.1655, "prec": 0.2509, "rec": 0.1374
        },
        "Weighted Sum SCAGA + Residual Loss": {
            "init": "Weighted Sum", "loss": "Residual-Aw",
            "psnr": 23.0754, "ssim": 0.7596, "f1": 0.1903, "prec": 0.2597, "rec": 0.1647
        }
    }

    print("\n" + "="*96)
    print("FINAL OFFICIAL BENCHMARK REPORT")
    print("="*96)
    print(f"1. Total Training Time : {t_train:.2f} seconds ({t_train/60:.2f} minutes)")
    print(f"2. Evaluation Time     : {t_eval:.2f} seconds")
    print(f"3. Peak GPU Memory     : {peak_gpu}")
    print(f"4. Peak CPU Memory     : {peak_cpu}")
    print("-" * 96)
    print(f"{'Method':<38} {'Init':<15} {'Loss':<12} {'PSNR':<8} {'SSIM':<7} {'F1':<7} {'Prec':<7} {'Recall':<7}")
    print("-" * 96)
    
    for k, v in benchmarks.items():
        print(f"{k:<38} {v['init']:<15} {v['loss']:<12} {v['psnr']:<8.4f} {v['ssim']:<7.4f} {v['f1']:<7.4f} {v['prec']:<7.4f} {v['rec']:<7.4f}")
    
    new_method = "Weighted Sum SCAGA + Sat. Res. Loss"
    print(f"{new_method:<38} {'Weighted Sum':<15} {'Sat-Res-Aw':<12} {psnr:<8.4f} {ssim:<7.4f} {edge_f1:<7.4f} {edge_precision:<7.4f} {edge_recall:<7.4f}")
    print("-" * 96)

    print("\nAbsolute Improvement of [Weighted Sum SCAGA + Sat. Res. Loss] over:")
    print(f"  • Baseline             : PSNR: {psnr - 22.2568:+0.4f} dB | Edge F1: {edge_f1 - 0.1679:+0.4f} | Edge Recall: {edge_recall - 0.1256:+0.4f}")
    print(f"  • Product SCAGA        : PSNR: {psnr - 22.4476:+0.4f} dB | Edge F1: {edge_f1 - 0.1824:+0.4f} | Edge Recall: {edge_recall - 0.1517:+0.4f}")
    print(f"  • Weighted Sum SCAGA   : PSNR: {psnr - 23.1640:+0.4f} dB | Edge F1: {edge_f1 - 0.2197:+0.4f} | Edge Recall: {edge_recall - 0.1910:+0.4f}")
    print(f"  • Importance Loss model: PSNR: {psnr - 22.2040:+0.4f} dB | Edge F1: {edge_f1 - 0.1655:+0.4f} | Edge Recall: {edge_recall - 0.1374:+0.4f}")
    print(f"  • Residual Loss model  : PSNR: {psnr - 23.0754:+0.4f} dB | Edge F1: {edge_f1 - 0.1903:+0.4f} | Edge Recall: {edge_recall - 0.1647:+0.4f}")
    print("="*96)


if __name__ == "__main__":
    run_experiment()
