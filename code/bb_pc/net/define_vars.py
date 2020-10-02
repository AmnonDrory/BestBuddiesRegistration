import torch
import numpy as np
from torch.autograd import Variable

def define_input_PCs(X,Y, X_normals, Y_normals):
    #  datasets
    train = torch.tensor(X.T, requires_grad=False)
    label = torch.tensor(Y.T, requires_grad=False)
    if X_normals is None:
        train_normals = None
        label_normals = None
    else:
        train_normals = torch.tensor(X_normals.T, requires_grad=False)
        label_normals = torch.tensor(Y_normals.T, requires_grad=False)
    
    if torch.cuda.is_available():
        train = train.cuda()
        label = label.cuda()
        if X_normals is not None:
            train_normals = train_normals.cuda()
            label_normals = label_normals.cuda()

    return train, label, train_normals, label_normals

def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device is not 'cpu':
        torch.cuda.empty_cache()
        DEBUG = False
    else:
        DEBUG = True
    device = torch.device(device)
    return device, DEBUG


def define_weights(init_angles, init_alpha, init_trans, trainable, device):
    init_angles = np.radians(init_angles)
    theta = torch.tensor([init_angles[0]], requires_grad=trainable["theta"]).to(device).detach().requires_grad_(
        trainable["theta"])
    phi = torch.tensor([init_angles[1]], requires_grad=trainable["phi"]).to(device).detach().requires_grad_(
        trainable["phi"])
    psi = torch.tensor([init_angles[2]], requires_grad=trainable["psi"]).to(device).detach().requires_grad_(
        trainable["psi"])
    trans_x = torch.tensor([init_trans[0]], requires_grad=trainable["trans_x"]).to(device).detach().requires_grad_(
        trainable["trans_x"])
    trans_y = torch.tensor([init_trans[1]], requires_grad=trainable["trans_y"]).to(device).detach().requires_grad_(
        trainable["trans_y"])
    trans_z = torch.tensor([init_trans[2]], requires_grad=trainable["trans_z"]).to(device).detach().requires_grad_(
        trainable["trans_z"])
    alpha = Variable(init_alpha * torch.ones(1), requires_grad=trainable["alpha"]).to(device).detach().requires_grad_(
        trainable["alpha"])
    return trans_x, trans_y, trans_z, theta, phi, psi, alpha

def define_optimizer(order, theta, psi, phi,\
                                 alpha, trans_lr,\
                                trans_x, trans_y, trans_z,
                                angles_lr, alpha_lr):
    if order == 'first':
        optimizer = torch.optim.Adam([{'params':[theta, phi, psi], 'lr': angles_lr}, \
                                    {'params':[alpha], 'lr': alpha_lr}, \
                                    {'params':[trans_x, trans_y, trans_z], 'lr': trans_lr}])
    else:
        print('Using second order optimization method...')
        optimizer = torch.optim.LBFGS([theta, phi, psi,trans_x, trans_y, trans_z], max_iter=2, lr = 1e-3)
    return optimizer