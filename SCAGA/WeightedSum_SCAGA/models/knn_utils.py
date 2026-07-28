import torch


def knn_points(p1, p2, K=1):
    """
    CPU KNN points implementation with memory-efficient advanced indexing
    """
    B, N, D = p1.shape
    M = p2.shape[1]

    if K > M:
        K = M

    device = p1.device

    # CPU KNN implementation to avoid GPU memory spikes
    p1_cpu = p1.float().cpu()
    p2_cpu = p2.float().cpu()
    dists = torch.cdist(p1_cpu, p2_cpu)
    dists_k, idx = torch.topk(dists, k=K, dim=-1, largest=False, sorted=True)

    dists_k = dists_k.to(device)
    idx = idx.to(device)

    # Memory-efficient knn_gather via advanced indexing (no [B, N, M, C] expand)
    nn = knn_gather(p2, idx)

    return dists_k, idx, nn


def knn_gather(x, idx):
    """
    x   : [B, M, C]
    idx : [B, N, K]

    return:
        [B, N, K, C]
    """
    B, M, C = x.shape
    _, N, K = idx.shape

    batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(-1, N, K)
    
    return x[batch_idx, idx, :]

