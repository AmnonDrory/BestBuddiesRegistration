import numpy as np
import torch
from copy import deepcopy

from ..utils.tools_3d import euler_angles_to_rotation_matrix_torch, apply_rot_trans
from ..utils.normals import point_to_plane_dist_sparse_torch
from ..net.net_steps import record_in_logs


def prep_sparse_subset_of_points_torch(torch_A, torch_B, torch_A_normals, torch_B_normals, torch_inds):

    torch_points = {'HARD_BEST_BUDDY_PAIRS': {}}
    torch_points['HARD_BEST_BUDDY_PAIRS']['A'] = torch_A[torch_inds['HARD_BEST_BUDDY_PAIRS']['A'], :]
    torch_points['HARD_BEST_BUDDY_PAIRS']['B'] = torch_B[torch_inds['HARD_BEST_BUDDY_PAIRS']['B'], :]
    torch_normals = {'HARD_BEST_BUDDY_PAIRS': {}}
    torch_normals['HARD_BEST_BUDDY_PAIRS']['A'] = torch_A_normals[torch_inds['HARD_BEST_BUDDY_PAIRS']['A'], :]
    torch_normals['HARD_BEST_BUDDY_PAIRS']['B'] = torch_B_normals[torch_inds['HARD_BEST_BUDDY_PAIRS']['B'], :]
    torch_points = {'points': torch_points, 'normals': torch_normals}

    return torch_points


def SG_apply_rot_trans_torch(torch_points, WT, transT):

    rotated_points = deepcopy(torch_points)
    for fld in ['points', 'normals']:
        rotated_points[fld]['HARD_BEST_BUDDY_PAIRS']['B'] = torch.matmul(torch_points[fld]['HARD_BEST_BUDDY_PAIRS']['B'].double(), WT)
        if fld == 'points':
            # also translate
            rotated_points[fld]['HARD_BEST_BUDDY_PAIRS']['B'] += transT

    return rotated_points


def calc_loss(torch_points, rotated_points):

    dist_Rus = point_to_plane_dist_sparse_torch(
        torch_points['points']['HARD_BEST_BUDDY_PAIRS']['A'].unsqueeze(2),
        rotated_points['points']['HARD_BEST_BUDDY_PAIRS']['B'].unsqueeze(2),
        torch_points['normals']['HARD_BEST_BUDDY_PAIRS']['A'].unsqueeze(2),
        rotated_points['normals']['HARD_BEST_BUDDY_PAIRS']['B'].unsqueeze(2))
    BD = dist_Rus.mean()

    return BD


def torch_intersect(Na, Nb, i_ab,j_ab,i_ba,j_ba):    
    def make_sparse_mat(i_,j_,sz):
        inds = torch.cat([i_.reshape([1,-1]),
                            j_.reshape([1,-1])],dim=0)
        vals = torch.ones_like(inds[0,:])
        
        M = torch.sparse.FloatTensor(inds,vals,sz)
        return M

    sz = [Na,Nb]
    M_ab = make_sparse_mat(i_ab,j_ab,sz)
    M_ba = make_sparse_mat(i_ba,j_ba,sz)

    M = M_ab.add(M_ba).coalesce()
    i, j = M._indices()
    v = M._values()
    is_both = (v == 2)
    i_final = i[is_both]
    j_final = j[is_both]

    return i_final, j_final


def knn_gpu(P, Q):
    nn_max_n = 500

    def knn_dist(p, q):
        # Fast implementation with torch.einsum()
        with torch.no_grad():      
            # L2 distance:
            dist2 = torch.sum(p**2, dim=1).reshape([-1,1]) + torch.sum(q**2, dim=1).reshape([1,-1]) -2*torch.einsum('ac,bc->ab', p, q)
            dist = dist2.clamp_min(1e-30).sqrt_()
            # Cosine distance:
            # dist = 1-torch.einsum('ac,bc->ab', p, q)                  
            min_dist, ind = dist.min(dim=1, keepdim=True)      
        return ind
    
    N = len(P)
    C = int(np.ceil(N / nn_max_n))
    stride = nn_max_n
    inds = []
    for i in range(C):
        with torch.no_grad():
            ind = knn_dist(P[i * stride:(i + 1) * stride], Q)
            inds.append(ind)
    
    inds = torch.cat(inds)
    assert len(inds) == N

    corres_idx0 = torch.arange(len(inds), device=P.device).long().squeeze()
    corres_idx1 = inds.long().squeeze()
    return corres_idx0, corres_idx1


def gpu_BB(P,Q):
    corres_idx0, corres_idx1 = knn_gpu(P, Q)
    
    uniq_inds_1=torch.unique(corres_idx1)
    inv_corres_idx1, inv_corres_idx0 = knn_gpu(Q[uniq_inds_1,:], P)
    inv_corres_idx1 = uniq_inds_1

    final_corres_idx0, final_corres_idx1 = torch_intersect(
    P.shape[0], Q.shape[0],
    corres_idx0, corres_idx1,
    inv_corres_idx0, inv_corres_idx1)

    return final_corres_idx0, final_corres_idx1


def prerun_gpu(torch_A, torch_B, WT, transT):
    torch_B_rot = (torch_B.double() @ WT) + transT
    pairs0, pairs1 = gpu_BB(torch_A,torch_B_rot)
    inds = {}
    inds['HARD_BEST_BUDDY_PAIRS'] = {}
    inds['HARD_BEST_BUDDY_PAIRS']['A'] = pairs0
    inds['HARD_BEST_BUDDY_PAIRS']['B'] = pairs1
    return inds


def BBR_F_step(A, B, torch_A, torch_B,
               torch_A_normals, torch_B_normals,
               theta, phi, psi, alpha,
               trans_x, trans_y, trans_z,
               optimizer,
               angles_np, alpha_np, trans_np, loss_np):

    angles_np, trans_np, alpha_np = record_in_logs(angles_np, trans_np, alpha_np, theta, phi, psi, trans_x, trans_y,
                                                   trans_z, alpha)

    W = torch.squeeze(euler_angles_to_rotation_matrix_torch(theta, phi, psi))
    trans = torch.cat([trans_x, trans_y, trans_z], dim=0).unsqueeze(1)

    WT = W.T.double()
    transT = trans.T.double()

    torch_inds = prerun_gpu(torch_A, torch_B, WT, transT)

    torch_points = prep_sparse_subset_of_points_torch(torch_A, torch_B, torch_A_normals, torch_B_normals, torch_inds)

    rotated_points = SG_apply_rot_trans_torch(torch_points, WT, transT)

    loss = calc_loss(torch_points, rotated_points)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    loss_np.append(loss.item())
    return loss, optimizer, loss_np, angles_np, alpha_np, trans_np

