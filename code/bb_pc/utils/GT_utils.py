import pandas as pd
import numpy as np
from ..utils.tools_3d import euler_angles_to_rotation_matrix, rotation_matrix_euler_angles, calc_angular_error, calc_translational_error
import sys

def load_GT_poses(filename, sep, header_row):
    if header_row:
        raw_data = pd.read_csv(filename, sep=sep)
    else:
        raw_data = pd.read_csv(filename, sep=sep, header=None)
    D = raw_data.values
    return D

def calc_remaining_motion(GT_motion, partial_motion):
    """
    We have the ground truth motion from B to A, and a partial motion
    from B to C. calculate the ground truth remaining motion from C to A.
    :param GT_motion:
    :param partial_motion:
    :return:
    """
    R_B_to_C = euler_angles_to_rotation_matrix(partial_motion['angles'])
    R_B_to_A = euler_angles_to_rotation_matrix(GT_motion['angles'])
    T_B_to_C = partial_motion['trans'].reshape([-1, 1])
    T_B_to_A = GT_motion['trans'].reshape([-1, 1])
    R_C_to_A = np.matmul(R_B_to_A, R_B_to_C.T)
    T_C_to_A = T_B_to_A - np.matmul(R_C_to_A, T_B_to_C)
    remaining_motion = {
        'angles': rotation_matrix_euler_angles(R_C_to_A),
        'trans': T_C_to_A.flatten()
    }

    return remaining_motion

def invert_motion(R, t):
    inv_R = R.T
    inv_t = -np.matmul(t.reshape([1,3]),inv_R)
    inv_angles = rotation_matrix_euler_angles(inv_R)
    inv_motion = { 'angles': inv_angles, 'trans': inv_t.flatten()}
    return inv_motion

def calc_relative_GT(GT_pair):
    delta = {}
    try:
        delta['raw'] = GT_pair[1]['raw'] - GT_pair[0]['raw'] # for time offsets
    except:
        pass
    Rot0 = euler_angles_to_rotation_matrix(GT_pair[0]['angles'])
    Rot1 = euler_angles_to_rotation_matrix(GT_pair[1]['angles'])
    delta_trans_raw = GT_pair[1]['trans'] - GT_pair[0]['trans']
    delta['trans'] = np.matmul(Rot0.T, delta_trans_raw)

    DeltaRot = np.matmul(Rot0.T, Rot1)
    delta['angles'] = rotation_matrix_euler_angles(DeltaRot)

    return delta

def print_GT_pair(GT_pair, mode='full'):
    if mode == "full":
        print("From:")
        print_GT(GT_pair[0])
        print("To:")
        print_GT(GT_pair[1])
        print("Delta:")
        delta = calc_relative_GT(GT_pair)
        print_GT(delta)
    elif mode == "short":
        try:
            str = "%d->%d: " % (GT_pair[0]['raw'][0], GT_pair[1]['raw'][0])
        except:
            str = ""
        delta = calc_relative_GT(GT_pair)
        print_GT_motion(delta, str)
    return delta

def print_GT_vs_result(gt, result, file=None):
    outfiles = [sys.stdout]
    if file is not None:
        outfiles.append(file)
    remaining = calc_remaining_motion(gt, result)

    for f in outfiles:
        print("Ground Truth:", file=f)
        print_GT(gt, motion_only=True, file=f, fileonly=True)
        print("Result:", file=f)
        print_GT(result, motion_only=True, file=f, fileonly=True)
        print("Remaining Motion (Error):", file=f)
        print_GT(remaining, motion_only=True, file=f, fileonly=True)

    angular_error = calc_angular_error(gt['angles'],result['angles'])
    print('Angular Error: %.4g' % angular_error)
    translation_error = calc_translational_error(gt,result)
    print('Translation Error: %.4g' % translation_error)

    return remaining


def print_GT(gt, motion_only=False, file=None, fileonly=False):
    str = ""
    if not motion_only:
        try:
            str += "ind: %d | " % gt['raw'][0]
            str += "time: %.4f | " % gt['raw'][1]
        except:
            pass
    print_GT_motion(gt, str, file, fileonly)

def print_GT_motion(gt, str="", file=None, fileonly=False):

    if (file is None):
        outfiles = [sys.stdout]
    else:
        if fileonly:
            outfiles = [file]
        else:
            outfiles = [sys.stdout, file]

    str += "trans: [x: %.4f, y: %.4f, z:%.4f] | " % (gt['trans'][0], gt['trans'][1], gt['trans'][2])
    str += "angles: [%.4f, %.4f, %.4f]" % (gt['angles'][0], gt['angles'][1], gt['angles'][2])
    for f in outfiles:
        print(str, file=f)