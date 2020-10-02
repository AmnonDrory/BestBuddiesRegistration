import numpy as np

def initialize_vars():
    angles_np = []
    loss_np = []
    alpha_np = []
    elapsed = []
    trans_np = []
    return angles_np, loss_np, alpha_np, elapsed, trans_np

def step_debug_prints(DEBUG, epoch,\
                    theta, phi, psi, trans_x, trans_y, trans_z, loss, alpha,\
                    GT_rot, GT_trans, N=10):
    if DEBUG and epoch%N is 0:
        angle = [180/np.pi*theta.item(), 180/np.pi*phi.item(), 180/np.pi*psi.item()]
        trans_np = np.expand_dims([trans_x.item(), trans_y.item(), trans_z.item()], axis=1)
        loss_val = (loss.cpu().data.numpy()).T
        print('n = {}'.format(epoch))
        print('--------')
        if GT_rot is not None:
            print('angle error={}'.format(angle-GT_rot))
        if GT_trans is not None:
            print('trans error={}'.format(trans_np-GT_trans))
        print('Est rot: {}'.format(angle))
        if GT_rot is not None:
            print('GT rot = ',GT_rot)
        print('Est trans = {}'.format(trans_np))
        if GT_trans is not None:
            print('GT trans = ',GT_trans)
        print('alpha = {}'.format(alpha.item()))
        print('Loss = {}'.format(loss_val))
        return

def finalize(epoch, angles_np, trans_np, loss_np, alpha_np):
    print('number of iterations = ', epoch+1)
    angles_np = np.rad2deg(angles_np).T
    trans_np = trans_np.T

    best_ind = np.argmin(loss_np)
    loss_of_res = loss_np[best_ind]
    res_angle = angles_np[best_ind,:]
    res_trans = trans_np[best_ind,:]

    res_motion = { 'trans': res_trans,
                   'angles': res_angle}

    full_log = {}
    full_log["angles"] = angles_np
    full_log["loss"] = loss_np
    full_log["trans"] = trans_np
    full_log["alpha"] = alpha_np

    optimization_succeeded = loss_of_res < loss_np[0]


    return optimization_succeeded, res_motion, full_log, loss_of_res
