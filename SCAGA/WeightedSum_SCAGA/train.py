import os
import shutil
import argparse
import numpy as np
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.dst_gs import CBCT_dataset_gs
from models.model import DIF_Gaussian
from utils import convert_cuda, load_config
from evaluate import eval_one_epoch

torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id):
    np.random.seed((worker_id + torch.initial_seed()) % np.iinfo(np.int32).max)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    
    parser.add_argument('--name', type=str, default='WeightedSum_SCAGA')
    parser.add_argument('--dst_name', type=str, default='LUNA16')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--num_views', type=int, default=6)
    parser.add_argument('--cfg_path', type=str, default='configs/default.yaml')
    parser.add_argument('--out_res_scale', type=float, default=1.0)
    parser.add_argument('--eval_npoint', type=int, default=100000)

    parser.add_argument('--local-rank', dest='local_rank', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', action='store_true', default=False)
    
    # Fast baseline used batch_size=1 to avoid VRAM over-allocation and PCIe swap bottleneck
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_points', type=int, default=10000)
    parser.add_argument('--random_views', action='store_true', default=False)
    parser.add_argument('--lambda_importance', type=float, default=1.0,
                        help='Weighting factor lambda for Hybrid Importance-Aware Loss (L = mean((1 + lambda*I)*(pred - gt)^2)).')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'importance', 'residual', 'saturated_residual'],
                        help='Loss function to use: mse, importance-aware loss, residual-aware importance loss, or saturated residual loss.')

    parser.add_argument('--resume', type=int, default=None)


    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='Set to 0 to disable periodic evaluation.')
    parser.add_argument('--eval_out_res_scale', type=float, default=0.5)
    parser.add_argument('--skip_edge_metrics', action='store_true')
    parser.add_argument('--profile_eval', action='store_true')

    args = parser.parse_args()

    if args.dist:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)

    cfg = load_config(args.cfg_path)
    if args.local_rank == 0:
        print(args)
        print(cfg)

        save_dir = f'./logs/{args.name}'
        os.makedirs(save_dir, exist_ok=True)
        if os.path.exists(os.path.join(save_dir, 'config.yaml')):
            time_str = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
            shutil.copyfile(
                os.path.join(save_dir, 'config.yaml'), 
                os.path.join(save_dir, f'config_{time_str}.yaml')
            )
        shutil.copyfile(args.cfg_path, os.path.join(save_dir, 'config.yaml'))

    train_dst = CBCT_dataset_gs(
        dst_name=args.dst_name,
        cfg=cfg.dataset,
        split='train', 
        num_views=args.num_views, 
        npoint=args.num_points,
        out_res_scale=args.out_res_scale,
        random_views=args.random_views
    )
    train_sampler = None
    if args.dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dst)
    train_loader = DataLoader(
        train_dst,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=worker_init_fn
    )

    eval_loader = DataLoader(
        CBCT_dataset_gs(
            dst_name=args.dst_name,
            cfg=cfg.dataset,
            split='eval',
            num_views=args.num_views,
            out_res_scale=args.eval_out_res_scale,
        ), 
        batch_size=1, 
        shuffle=False,
        pin_memory=False
    )

    model = DIF_Gaussian(cfg.model)
    if args.resume:
        print(f'resume model from epoch {args.resume}')
        ckpt = torch.load(
            os.path.join(f'./logs/{args.name}/ep_{args.resume}.pth'),
            map_location=torch.device('cpu')
        )
        model.load_state_dict(ckpt)
    
    model = model.cuda()
    if args.dist:
        model = nn.parallel.DistributedDataParallel(
            model, 
            find_unused_parameters=False,
            device_ids=[args.local_rank]
        )
    
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=0.98, 
        weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=1, 
        gamma=np.power(args.lr_decay, 1 / max(args.epoch, 1))
    )
    loss_func = nn.MSELoss()

    start_epoch = 0
    if args.resume:
        start_epoch = args.resume + 1
        if lr_scheduler is not None:
            lr_scheduler.step(epoch=args.resume)

    import time
    import psutil
    process = psutil.Process()
    epoch_times = {}

    for epoch in range(start_epoch, args.epoch + 1):
        epoch_start_time = time.time()
        if args.dist:
            train_loader.sampler.set_epoch(epoch)

        loss_list = []
        model.train()
        optimizer.zero_grad()

        for k, item in enumerate(train_loader):
            item = convert_cuda(item)

            pred = model(item)
            
            # Loss calculations
            loss_type = getattr(args, 'loss_type', 'mse')
            importance = item.get('points_importance', torch.zeros_like(item['points_gt']))
            lambda_val = getattr(args, 'lambda_importance', 1.0)
            
            e = pred['points_pred'] - item['points_gt']
            sq_err = e ** 2
            
            if loss_type == 'mse':
                loss = torch.mean(sq_err)
            elif loss_type == 'importance':
                weight = 1.0 + lambda_val * importance
                loss = torch.mean(weight * sq_err)
            elif loss_type == 'residual':
                # Residual-Aware Importance Loss: weight = 1 + lambda * importance * abs(e.detach())
                weight = 1.0 + lambda_val * importance * torch.abs(e.detach())
                loss = torch.mean(weight * sq_err)
            elif loss_type == 'saturated_residual':
                # Saturating Residual-Aware Loss: weight = 1 + lambda * importance * (1 - exp(-abs(e.detach())))
                weight = 1.0 + lambda_val * importance * (1.0 - torch.exp(-torch.abs(e.detach())))
                loss = torch.mean(weight * sq_err)
            else:

                raise ValueError(f"Unknown loss_type: {loss_type}")

            loss_list.append(loss.item())



            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        if 'item' in locals():
            del item, pred, loss
        
        if args.local_rank == 0:
            if epoch % 10 == 0:
                loss = np.mean(loss_list)
                print('epoch: {}, loss: {:.4}'.format(epoch, loss))
            
            if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
                if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    model_state = model.module.state_dict()
                else:
                    model_state = model.state_dict()
                torch.save(
                    model_state,
                    os.path.join(save_dir, f'ep_{epoch}.pth')
                )

            if args.eval_interval > 0 and epoch % args.eval_interval == 0:
                metrics, _ = eval_one_epoch(
                    model, 
                    eval_loader, 
                    args.eval_npoint,
                    ignore_msg=True,
                    compute_edge_metrics=not args.skip_edge_metrics,
                    profile=args.profile_eval,
                )
                msg = f' --- epoch {epoch}'
                for dst_name in metrics.keys():
                    msg += f', {dst_name}'
                    met = metrics[dst_name]
                    for key, val in met.items():
                        msg += ', {}: {:.4}'.format(key, val)
                print(msg)
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        epoch_times[epoch] = time.time() - epoch_start_time

    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB")
    print(f"Peak CPU memory: {process.memory_info().rss / (1024**2):.2f} MB")
    for ep, t in epoch_times.items():
        print(f"Time for Epoch {ep}: {t:.2f} seconds")
    
    if len(epoch_times) > 0:
        avg_time = sum(epoch_times.values()) / len(epoch_times)
        print(f"Estimated time per 10 epochs: {avg_time * 10 / 60:.2f} minutes")
