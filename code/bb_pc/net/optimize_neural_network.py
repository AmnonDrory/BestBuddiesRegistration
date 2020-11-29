import torch
import numpy as np
from ..net.define_vars import define_input_PCs, get_device, define_weights, define_optimizer
from ..net.net_steps import gd_step
from ..net.utils import finalize, step_debug_prints, initialize_vars
from ..net.BBR_F import BBR_F_step

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def optimize_neural_network(A, B,
                            A_normals=None,
                            B_normals=None,
                            loss_type='BBS', # 'BBS' or 'BD'
                            distance_measure='point2point', # 'point2point' or 'point2plane'
                            BBR_F=False,
                            init_alpha=1e-2,
                            nIterations=300,
                            angles_lr=5e-2,
                            alpha_lr=1e-8,
                            trans_lr = 1e-3,
                            LR_step_size = 1000,
                            LR_factor = 1.0,
                            order = 'first',
                            trainable = None,
                            init_trans = None,
                            init_angles = None,
                            GT_rot = None,
                            GT_trans = None,
                            ):

    if init_trans is None:
        init_trans = np.array([0,0,0],dtype=float)
    if init_angles is None:
        init_angles = np.array([0, 0, 0], dtype=float)

    if trainable is None:
        trainable = {"trans_x": True, "trans_y": True, "trans_z": True, "theta": True, "phi": True, "psi": True, "alpha": True}

    train, label, train_normals, label_normals = define_input_PCs(B, A, B_normals, A_normals)
    if BBR_F:
        torch_A = label.t()
        torch_B = train.t()
        torch_A_normals = label_normals.t()
        torch_B_normals = train_normals.t()

    device, DEBUG = get_device()
    print("device=" + str(device))

    trans_x, trans_y, trans_z, theta, phi, psi, alpha = define_weights(init_angles, \
                                                                         init_alpha, init_trans,\
                                                                         trainable, device)

    optimizer = define_optimizer(order, theta, psi, phi, alpha, trans_lr, trans_x, trans_y, trans_z,
                                angles_lr, alpha_lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=LR_factor)

    angles_np, loss_np, alpha_np, elapsed, trans_np = initialize_vars()

    epoch = 0

    while (epoch<nIterations):

        epoch+=1

        if BBR_F:
            loss, optimizer, loss_np, angles_np, alpha_np, trans_np = BBR_F_step(A, B, torch_A, torch_B,
                                                        torch_A_normals, torch_B_normals,
                                                        theta, phi, psi, alpha, \
                                                        trans_x, trans_y, trans_z, \
                                                        optimizer, \
                                                        angles_np, alpha_np, trans_np, loss_np)
																
        else:

            loss, optimizer, loss_np, angles_np, alpha_np, trans_np = gd_step(train, label, train_normals, label_normals, loss_type, distance_measure, \
                                                                              theta, phi, psi, alpha, \
                                                                              trans_x, trans_y, trans_z, \
                                                                              optimizer, \
                                                                              loss_np, angles_np, alpha_np, \
                                                                              order, trans_np)
        scheduler.step()
        MIN_ALPHA = 1e-8
        alpha.data = alpha.data.clamp(MIN_ALPHA)

        step_debug_prints(DEBUG, epoch,theta, phi, psi, trans_x, trans_y, trans_z, loss, alpha,GT_rot, GT_trans, N=10)

    optimization_succeeded, res_motion, full_log, loss_of_res =\
        finalize(epoch, angles_np, trans_np, loss_np, alpha_np)

    return optimization_succeeded, res_motion, full_log, loss_of_res
