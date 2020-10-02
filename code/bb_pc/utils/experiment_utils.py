import numpy as np

from ..utils.tools_3d import apply_rot_trans
import matplotlib.pyplot as plt
from ..utils.visualization import draw_registration_result

def show_result_graphs(full_log, title=None, blocking=True):
    plt.figure()
    plt.subplot(3, 3, 1)
    plt.plot(full_log["loss"])
    plt.title('loss')
    plt.subplot(3, 3, 2)
    plt.plot(full_log["alpha"])
    plt.title('alpha')
    plt.subplot(3, 3, 3)
    plt.plot(full_log["trans"][:, 0], full_log["loss"])
    plt.title('loss per x')

    for i in range(3):
        plt.subplot(3, 3, 4 + i)
        plt.plot(full_log["angles"][:, i])
        plt.title('angle[%d]' % i)
    for i in range(3):
        plt.subplot(3, 3, 7 + i)
        plt.plot(full_log["trans"][:, i])
        plt.title('translation[%d]' % i)
    plt.suptitle(title)
    if blocking:
        plt.show()

def show_result_point_clouds(PCs, A,B, A_sub, B_sub, res_motion, mot_grid, mot_BBS):
    B_sub_corrected = apply_rot_trans(B_sub, res_motion['angles'], res_motion['trans'])
    draw_registration_result(A_sub, B_sub_corrected)
    B_corrected = apply_rot_trans(B, res_motion['angles'], res_motion['trans'])
    draw_registration_result(A, B_corrected)
    B0 = PCs[1]
    B1 = apply_rot_trans(B0, mot_grid['angles'], mot_grid['trans'])
    B2 = apply_rot_trans(B1, mot_BBS['angles'], mot_BBS['trans'])
    B3 = apply_rot_trans(B2, res_motion['angles'], res_motion['trans'])
    draw_registration_result(A, B3)
    

def get_quantiles(data, Q):
    Q = np.array(Q)
    assert (np.min(Q) >= 0.) and (np.max(Q) <= 1.)
    s = np.sort(data)
    inds = np.round((len(s)-1)*Q).astype(int)
    return s[inds]

def print_to_file_and_screen(outfile, s, *args, **vargs):
    print(s, *args, **vargs)
    print(s, *args, **vargs, file=outfile)