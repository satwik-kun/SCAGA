import torch
import numpy as np
from models.knn_utils import knn_points, knn_gather


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + 
        r[:, 1] * r[:, 1] + 
        r[:, 2] * r[:, 2] + 
        r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.size(0), 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def build_covariance(s, r):
    b, n = s.shape[:2]
    s = s.reshape(b * n, 3)
    r = r.reshape(b * n, 4)
    
    L = build_scaling_rotation(s, r)
    Cov = L @ L.transpose(1, 2)

    Cov = Cov.reshape(b, n, 3, 3)
    return Cov


def query_gs(points, gs_points, gs_params):
    B, K = gs_points.shape[:2]
    N = points.shape[1]
    k = 3

    det = gs_params['det']
    inv = gs_params['inv']
    
    _, neb_idx, _ = knn_points(points, gs_points, K=k)
    points_ext = points.unsqueeze(2).repeat(1, 1, k, 1)
    neb_gs_xyz = knn_gather(
        gs_points + gs_params['offsets'],
        neb_idx
    )
    neb_gs_feats = knn_gather(
        gs_params['feats'].transpose(1, 2),
        neb_idx
    ).permute(0, 3, 1, 2)
    neb_det = knn_gather(
        det.unsqueeze(-1),
        neb_idx
    ).squeeze(-1)
    neb_inv = knn_gather(
        inv.reshape(B, K, 9),
        neb_idx
    ).reshape(B, N, k, 3, 3)

    diff = points_ext - neb_gs_xyz
    quad = torch.einsum("bnki,bnkij,bnkj->bnk", diff, neb_inv, diff)
    weights = np.power(2 * np.pi, -3/2) * \
        torch.pow(neb_det, -1/2) * \
        torch.exp(-1/2 * quad)

    weights = weights.unsqueeze(1)
    sum_feats = (weights * neb_gs_feats).sum(dim=-1)

    return sum_feats
