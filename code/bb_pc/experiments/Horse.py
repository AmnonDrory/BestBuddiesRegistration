import open3d as o3d
import numpy as np
import os
import sys
# select which GPU to work on (read from first command line argument) (this code has to come before "import tensorflow")
if len(sys.argv) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]  # select the gpu to use

from ..net.optimize_neural_network import optimize_neural_network
from ..general.TicToc import *
from ..general.bb_pc_path import bb_pc_path
from ..utils.data_tools import import_ply, calc_normals
from ..utils.torch_numpy_3d_tools import rotate_around_random_axis, random_unit_vector
from ..utils.GT_utils import print_GT_vs_result, invert_motion

np.set_printoptions(precision=3, floatmode='maxprec', suppress=True, linewidth=200, threshold=10**6)

def create_subsets_with_normals(full_PC, num_samples):
    full_normals = calc_normals(full_PC)
    inds_A = np.random.permutation(full_PC.shape[0])[:num_samples]
    A = full_PC[inds_A,:]
    A_normals = full_normals[inds_A, :]
    inds_B = np.random.permutation(full_PC.shape[0])[:num_samples]
    B = full_PC[inds_B,:]
    B_normals = full_normals[inds_B, :]
    return A, B, A_normals, B_normals

def optimize(mode, A, B, A_normals, B_normals):
    config = {}

    config['BBR-softBBS'] = {
        'loss_type': 'BBS',
        'distance_measure': 'point2point',
        'nIterations': 750,
        'alpha_lr': 5e-6,
        'angles_lr': 5e-3,
        'trans_lr': 1e-5,
    }
    config['BBR-softBD'] = {
        'loss_type': 'BD',
        'distance_measure': 'point2point',
        'nIterations': 750,
        'alpha_lr': 5e-6,
        'angles_lr': 5e-3,
        'trans_lr': 1e-5,
    }
    config['BBR-N'] = {
        'loss_type': 'BD',
        'distance_measure': 'point2plane',
        'nIterations': 1000,
        'alpha_lr': 3e-6,
        'angles_lr': 8e-3,
        'trans_lr': 9e-5,
        'LR_step_size': 500,
        'LR_factor': 1e-1
    }

    if mode == 'BBR-F':
        print("BBR-F currently not implemented")
        return None


    _, res_motion_raw, _, _ = \
        optimize_neural_network(A, B, A_normals, B_normals, **config[mode])
    res_motion = [[mode, res_motion_raw]]

    return res_motion

def main():
    """
    Take two random subsets A and B from the Horse point cloud.
    Select a random rotation and translation to be the ground truth motion,
    apply it to one of the subsets .
    Use Best Buddies Registration to find the rotation and translation that best
    aligns the two subsets.
    Compare results of algorithm to ground truth motion and meaure error.
    Uses three varieties of Best Buddies Registration:
    1. BBR: Optimize with Best Buddies Similarity (BBS) loss, then refine
            with the Buddy-Distance (BD) loss.
    2. BD:  Optimize with the Buddy-Distance loss
    3. BDN: Optimize with the BD loss, using the point-to-plane distance
            measure, ehich makes use of normals.
    """
    ANGLE = 10 # degrees
    OFFSET = 0.005 # METERS
    NUM_SAMPLES = 500

    full_PC = import_ply(bb_pc_path['horse_points_file'])
    PC_A, PC_B, A_normals, B_normals = create_subsets_with_normals(full_PC, NUM_SAMPLES)

    PC_A_rotated, gt_angle, n, R = rotate_around_random_axis(PC_A, ANGLE)
    trans = np.expand_dims(OFFSET * random_unit_vector(), axis=1)
    PC_A_rotated_translated = PC_A_rotated + trans.reshape([1,3])
    GT_motion = {'angles': gt_angle.flatten(), 'trans': trans.flatten()}
    A_normals_rotated = np.matmul(A_normals, R)

    for mode in ['BBR-softBBS', 'BBR-softBD', 'BBR-N', 'BBR-F']:
        print("\n\n================= Running in mode: %s =================" % mode)
        estimated_motions = optimize(mode, PC_A_rotated_translated, PC_B, A_normals_rotated, B_normals)
        for mot in estimated_motions:
            cur_mode = mot[0]
            if len(estimated_motions) > 1:
                print('\n' + cur_mode + ':\n---')
            print_GT_vs_result(GT_motion, mot[1])

    print("\n\n========= Finished Successfully ========")

if __name__ == "__main__":
    main()
