import torch
import numpy as np
from ..utils.tools_3d import euler_angles_to_rotation_matrix_torch, subsample_randomly, apply_rot_trans_torch, rotate_3d_torch
from ..utils.algo_utils import soft_BD_loss_torch, soft_BBS_loss_torch
from ..utils.torch_numpy_3d_tools import point_to_plane_dist

def record_in_logs(angles_np,trans_np,alpha_np,theta, phi, psi, trans_x, trans_y, trans_z, alpha):
    angle = np.expand_dims([theta.item(), phi.item(), psi.item()], axis=1)
    trans = np.expand_dims([trans_x.item(), trans_y.item(), trans_z.item()], axis=1)
    if angles_np != [] and np.shape(angles_np.shape)[0]<2:
        angles_np = np.expand_dims(angles_np, axis=1)
    if trans_np != [] and np.shape(trans_np.shape)[0]<2:
        trans_np = np.expand_dims(trans_np, axis=1)
    if angles_np == []:
        angles_np = angle
    else:
        angles_np = np.append(angles_np, angle, axis=1)
    if trans_np == []:
        trans_np = trans
    else:
        trans_np = np.append(trans_np, trans, axis=1)
    alpha_np.append(alpha.item())
    
    return angles_np,trans_np,alpha_np,

def gd_step(train, label, train_normals, label_normals, loss_type, distance_measure, \
            theta, phi, psi, alpha, \
            trans_x, trans_y, trans_z, \
            optimizer, \
            loss_np, angles_np, alpha_np, \
            order, trans_np):
    angles_np, trans_np, alpha_np = record_in_logs(angles_np, trans_np, alpha_np, theta, phi, psi, trans_x, trans_y,
                                                   trans_z, alpha)
    W = torch.squeeze(euler_angles_to_rotation_matrix_torch(theta, phi, psi))
    trans = torch.cat([trans_x, trans_y, trans_z], dim=0).unsqueeze(1)
    pred = apply_rot_trans_torch(train, W, trans)
    if train_normals is not None:
        pred_normals = rotate_3d_torch(train_normals, W)
    if order == 'first':
        if loss_type=='BD':
            if distance_measure == 'point2plane':
                D = point_to_plane_dist(pred, label, pred_normals, label_normals)
            else:
                D = None

            loss = soft_BD_loss_torch(torch.transpose(pred, 0, 1), \
                                      label, \
                                      alpha, D)
        else:
            loss = soft_BBS_loss_torch(torch.transpose(pred, 0, 1), \
                                       label, \
                                       alpha)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        def closure():
            optimizer.zero_grad()
            loss = soft_BD_loss_torch(torch.transpose(pred, dim0=0, dim1=1), \
                                      torch.transpose(label, dim0=0, dim1=1), \
                                      alpha) + (theta.double()).pow(2)
            loss.backward(retain_graph=True)
            return loss
        loss = optimizer.step(closure)

    loss_np.append(loss.item())
    return loss, optimizer, loss_np, angles_np, alpha_np, trans_np

def sgd_step(train, label, batch_size, loss_type,\
            theta, phi, psi, alpha, \
            trans_x, trans_y, trans_z, \
            optimizer, \
            loss_np, angles_np, alpha_np,
            trans_np):

    train_batch = subsample_randomly(train, batch_size, axis=1)
    label_batch = subsample_randomly(label, batch_size, axis=1)
    angles_np, trans_np, alpha_np = record_in_logs(angles_np, trans_np, alpha_np, theta, phi, psi, trans_x, trans_y,
                                                   trans_z, alpha)
    trans = torch.cat([trans_x, trans_y, trans_z], dim=0).unsqueeze(1)
    W = torch.squeeze(euler_angles_to_rotation_matrix_torch(theta, phi, psi))
    pred_batch = apply_rot_trans_torch(train_batch, W, trans)
    if loss_type=='BD':
        loss = soft_BD_loss_torch(torch.transpose(pred_batch, dim0=0, dim1=1), \
                                  torch.transpose(label_batch, dim0=0, dim1=1), \
                                  alpha)
    else:
        loss = soft_BBS_loss_torch(torch.transpose(pred_batch, dim0=0, dim1=1), \
                                   torch.transpose(label_batch, dim0=0, dim1=1), \
                                   alpha)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_np.append(loss.item())
    return loss, optimizer, loss_np, angles_np, alpha_np, trans_np