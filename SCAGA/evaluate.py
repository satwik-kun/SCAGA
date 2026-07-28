import os
import csv
import json
import argparse
import time
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.segmentation import find_boundaries

from datasets.dst_gs import CBCT_dataset_gs
from models.model import DIF_Gaussian
from utils import convert_cuda, load_config, load_runtime_config, sitk_save



def compute_edge_f1(gt_vol, pred_vol, threshold=0.5):
    """Compute edge F1 score between ground truth and predicted 3D volumes.
    
    Binarizes both volumes at `threshold`, extracts 3D morphological boundaries
    via find_boundaries, then computes precision, recall and F1.
    """
    pred_bin = (pred_vol > threshold).astype(np.uint8)
    gt_bin   = (gt_vol   > threshold).astype(np.uint8)

    gt_edges   = find_boundaries(gt_bin,   connectivity=3, mode='outer')
    pred_edges = find_boundaries(pred_bin, connectivity=3, mode='outer')

    tp = np.logical_and(gt_edges, pred_edges).sum()
    fp = np.logical_and(pred_edges, ~gt_edges).sum()
    fn = np.logical_and(gt_edges,  ~pred_edges).sum()

    precision = float(tp / (tp + fp + 1e-8))
    recall    = float(tp / (tp + fn + 1e-8))
    f1        = float(2 * precision * recall / (precision + recall + 1e-8))
    return f1, precision, recall


def eval_one_epoch(model, loader, npoint=50000, save_dir=None, ignore_msg=True,
                   use_tqdm=False, compute_edge_metrics=True, profile=False):
    model.eval()
    results = {}
    metrics = {}
    metrics_tmp = {key:[] for key in ['psnr', 'ssim', 'edge_f1', 'edge_precision', 'edge_recall']}
    if use_tqdm:
        loader = tqdm(loader, ncols=50)
    
    with torch.no_grad():
        for item in loader:
            timings = {}
            start = time.perf_counter()
            item = convert_cuda(item)
            torch.cuda.synchronize()
            timings['data_transfer_s'] = time.perf_counter() - start

            dst_name = item['dst_name'][0]
            name = item['name'][0]
            image = item['points_gt'].cpu().numpy()
            image = image[0] # W, H, D

            start = time.perf_counter()
            pred = model(item, is_eval=True, eval_npoint=npoint) # B, 1, N
            torch.cuda.synchronize()
            timings['reconstruction_s'] = time.perf_counter() - start
            output = pred['points_pred']
            start = time.perf_counter()
            output = output[0, 0].data.cpu().numpy()
            
            output = output.reshape(image.shape)
            output = np.clip(output, 0, 1)

            psnr = peak_signal_noise_ratio(image, output, data_range=1.)
            ssim = structural_similarity(image, output, data_range=1.)
            timings['image_metrics_s'] = time.perf_counter() - start
            if compute_edge_metrics:
                start = time.perf_counter()
                edge_f1, edge_prec, edge_rec = compute_edge_f1(image, output)
                timings['edge_metrics_s'] = time.perf_counter() - start
            else:
                edge_f1 = edge_prec = edge_rec = float('nan')

            if profile:
                print('{} evaluation timing: {}'.format(
                    name, ', '.join(f'{key}={value:.2f}s' for key, value in timings.items())
                ))

            if not ignore_msg:
                print('{}, PSNR: {:.4f}, SSIM: {:.4f}, Edge F1: {:.4f}'.format(
                    name, psnr, ssim, edge_f1
                ))

            dst_res = results.get(dst_name, [])
            dst_met = metrics.get(dst_name, deepcopy(metrics_tmp))

            dst_res.append({
                'name': name, 
                'psnr': psnr,
                'ssim': ssim,
                'edge_f1':        edge_f1,
                'edge_precision': edge_prec,
                'edge_recall':    edge_rec,
            })
            for key in dst_met.keys():
                dst_met[key].append(dst_res[-1][key])
            
            results[dst_name] = dst_res
            metrics[dst_name] = dst_met

            if save_dir is not None:
                spacing = item['spacing'][0].cpu().numpy()
                origin = item['origin'][0].cpu().numpy()
                save_path = os.path.join(save_dir, f'{name}.nii.gz')
                sitk_save(save_path, output, spacing=spacing, origin=origin, uint8=True)

    for dst_name in metrics.keys():
        dst_met = metrics[dst_name]
        m = {key:np.mean(val) for key, val in dst_met.items()}
        metrics[dst_name] = m

    return metrics, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval')

    parser.add_argument('--name', type=str, default='baseline')
    parser.add_argument('--dst_name', type=str, default='LUNA16')
    parser.add_argument('--epoch', type=int, default=400)
    parser.add_argument('--num_views', type=int, default=10)
    parser.add_argument('--cfg_path', type=str, default=None)
    parser.add_argument('--machine_config', type=str, default=None)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--view_offset', type=int, default=0)
    parser.add_argument('--out_res_scale', type=float, default=1.0)
    parser.add_argument('--eval_npoint', type=int, default=100000)
    parser.add_argument('--save_results', action='store_true', default=False)

    args = parser.parse_args()
    if args.cfg_path is None:
        args.cfg_path = f'./logs/{args.name}/config.yaml'
        if not os.path.exists(args.cfg_path):
            args.cfg_path = 'configs/experiment.yaml'
    
    print(args)

    try:
        cfg = load_runtime_config(args.cfg_path, args.machine_config)
    except Exception:
        cfg = load_config(args.cfg_path)

    out_root = getattr(cfg, 'output_root', './logs')
    exp_dir = os.path.join(out_root, args.name)

    # -- dataloader
    eval_loader = DataLoader(
        CBCT_dataset_gs(
            dst_name=args.dst_name,
            cfg=cfg.dataset,
            split=args.split, 
            num_views=args.num_views,
            out_res_scale=args.out_res_scale,
            view_offset=args.view_offset,
        ), 
        batch_size=1, 
        shuffle=False,
        pin_memory=False
    )

    # -- model, load ckpt
    ckpt_path = os.path.join(exp_dir, f'ep_{args.epoch}.pth')
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    print('load ckpt from', ckpt_path)
    
    model = DIF_Gaussian(cfg.model)
    model.load_state_dict(ckpt)
    model = model.cuda()

    # -- output dir
    tag = '{:.1f}x'.format(args.out_res_scale)
    save_dir = None
    if args.save_results:
        save_dir = os.path.join(exp_dir, f'results/ep_{args.epoch}/predictions_{tag}')
        os.makedirs(save_dir, exist_ok=True)

    # -- evaluate
    metrics, results = eval_one_epoch(
        model, 
        eval_loader, 
        args.eval_npoint,
        save_dir=save_dir,
        use_tqdm=True,
        ignore_msg=False
    )
    print(metrics)

    # -- save results [csv]
    pred_dir = os.path.join(exp_dir, f'results/ep_{args.epoch}')
    os.makedirs(pred_dir, exist_ok=True)

    csv_file = open(os.path.join(pred_dir, f'results_{tag}.csv'), 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['dataset', 'obj_id', 'psnr', 'ssim', 'edge_f1', 'edge_precision', 'edge_recall'])

    for dst_name in results.keys():
        dst_res = results[dst_name]
        for res in dst_res:
            csv_writer.writerow([
                dst_name, res['name'],
                res['psnr'], res['ssim'],
                res['edge_f1'], res['edge_precision'], res['edge_recall']
            ])

        dst_avg = metrics[dst_name]
        csv_writer.writerow([
            dst_name, 'average',
            dst_avg['psnr'], dst_avg['ssim'],
            dst_avg['edge_f1'], dst_avg['edge_precision'], dst_avg['edge_recall']
        ])
    
    csv_file.close()
    
    # -- save config [args]
    with open(os.path.join(pred_dir, 'args.json'), 'w') as f:
        args = vars(args)
        json.dump(args, f, indent=4)
