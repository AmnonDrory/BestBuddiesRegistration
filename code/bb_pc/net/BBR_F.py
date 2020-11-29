import numpy as np
import torch
from copy import deepcopy
from sklearn.neighbors import NearestNeighbors
import sortednp as snp

from ..utils.tools_3d import euler_angles_to_rotation_matrix_torch, apply_rot_trans
from ..utils.normals import point_to_plane_dist_sparse_torch
from ..net.net_steps import record_in_logs

def bidirectional_nearest_neighbors_postprocess(A, Brot, j_inds_h, i_inds_v, K=1):
    i_inds_h = np.repeat(np.arange(A.shape[0]).reshape([-1, 1]), K, axis=1)
    j_inds_v = np.repeat(np.arange(Brot.shape[0]).reshape([-1, 1]), K, axis=1)
    return i_inds_h, j_inds_h, i_inds_v, j_inds_v


def bidirectional_nearest_neighbors(A, Brot, K=1):
    NN = NearestNeighbors(n_neighbors=1,
                          radius=None,
                          algorithm='kd_tree',
                          leaf_size=30,
                          metric='minkowski',
                          p=2,
                          metric_params=None,
                          n_jobs=None)

    NN.fit(Brot)
    j_inds_h = NN.kneighbors(A, K, return_distance=False)

    NN.fit(A)
    i_inds_v = NN.kneighbors(Brot, K, return_distance=False)

    return bidirectional_nearest_neighbors_postprocess(A, Brot, j_inds_h, i_inds_v, K)


def prep_BBP_inds_for_soft_argmin(A, Brot):
    # Quick CPU run to find the nearest neighbors in (rotated) B for each point in A,
    # and vice versa.

    i_inds_h, j_inds_h, i_inds_v, j_inds_v = bidirectional_nearest_neighbors(A, Brot)

    num_points_B = Brot.shape[0]

    comb_inds_h = num_points_B * i_inds_h.flatten() + j_inds_h.flatten()
    comb_inds_v = num_points_B * i_inds_v.flatten() + j_inds_v.flatten()
    comb_inds_h_s = np.sort(comb_inds_h)
    comb_inds_v_s = np.sort(comb_inds_v)
    comb_inds_pairs = snp.intersect(comb_inds_h_s, comb_inds_v_s)
    i_inds_both, j_inds_both = np.divmod(comb_inds_pairs, num_points_B)
    pairs = [i_inds_both.astype(int), j_inds_both.astype(int)]

    inds = {}
    inds['HARD_BEST_BUDDY_PAIRS'] = {}
    inds['HARD_BEST_BUDDY_PAIRS']['A'] = pairs[0]
    inds['HARD_BEST_BUDDY_PAIRS']['B'] = pairs[1]

    return inds


def prep_sparse_subset_of_points_torch(torch_A, torch_B, torch_A_normals, torch_B_normals, torch_inds):

    torch_points = {'HARD_BEST_BUDDY_PAIRS': {}}
    torch_points['HARD_BEST_BUDDY_PAIRS']['A'] = torch_A[torch_inds['HARD_BEST_BUDDY_PAIRS']['A'], :]
    torch_points['HARD_BEST_BUDDY_PAIRS']['B'] = torch_B[torch_inds['HARD_BEST_BUDDY_PAIRS']['B'], :]
    torch_normals = {'HARD_BEST_BUDDY_PAIRS': {}}
    torch_normals['HARD_BEST_BUDDY_PAIRS']['A'] = torch_A_normals[torch_inds['HARD_BEST_BUDDY_PAIRS']['A'], :]
    torch_normals['HARD_BEST_BUDDY_PAIRS']['B'] = torch_B_normals[torch_inds['HARD_BEST_BUDDY_PAIRS']['B'], :]
    torch_points = {'points': torch_points, 'normals': torch_normals}

    return torch_points


def convert_inds_to_torch(inds):
    torch_inds = {}
    for k1 in inds.keys():
        torch_inds[k1] = {}
        for k2 in inds[k1].keys():
            torch_inds[k1][k2] = torch.tensor(inds[k1][k2], requires_grad=False)
            if torch.cuda.is_available():
                torch_inds[k1][k2] = torch_inds[k1][k2].cuda()

    return torch_inds


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

def prerun_cpu(A, B, theta, phi, psi, trans_x, trans_y, trans_z):
    angles = np.degrees([theta.item(), phi.item(), psi.item()])
    trans = np.array([trans_x.item(), trans_y.item(), trans_z.item()])
    Brot = apply_rot_trans(B, angles, trans)
    Brot = Brot.astype(np.float32)
    sparse_inds = prep_BBP_inds_for_soft_argmin(A, Brot)
    torch_inds = convert_inds_to_torch(sparse_inds)
    return torch_inds


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

    torch_inds = prerun_cpu(A, B, theta, phi, psi, trans_x, trans_y, trans_z)

    torch_points = prep_sparse_subset_of_points_torch(torch_A, torch_B, torch_A_normals, torch_B_normals, torch_inds)

    rotated_points = SG_apply_rot_trans_torch(torch_points, WT, transT)

    loss = calc_loss(torch_points, rotated_points)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    loss_np.append(loss.item())
    return loss, optimizer, loss_np, angles_np, alpha_np, trans_np

