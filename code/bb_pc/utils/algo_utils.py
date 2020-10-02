from ..utils.PointCloudUtils import dist, cdist, cdist_torch, representative_neighbor_dist
from ..utils.net_tools import soft_argmin_on_rows, argmin_on_cols, argmin_on_rows, softargmin_rows_torch
import numpy as np
import torch

def guess_best_alpha(A):
        """
        A good guess for the temperature of the soft argmin (alpha) can
        be calculated as a linear function of the representative (e.g. median)
        distance of points to their nearest neighbor in a point cloud.

        :param A: Point Cloud of size Nx3
        :return: Estimated value of alpha
        """

        if A.shape[1] != 3:
                A = A[:,:3]

        COEFF = 0.1 # was estimated using the code in experiments/alpha_regression.py
        rep = representative_neighbor_dist(dist(A, A))
        return COEFF * rep

def hard_BBS_loss(T, S):
        D = cdist(T, S)
        R = argmin_on_cols(D)
        C = argmin_on_cols(D.T)
        B = R * (np.transpose(C))
        loss = -np.sum(B)
        loss /= np.mean([T.shape[0],S.shape[0]])
        return loss

def soft_BBS_loss(T, S, t):
        D = cdist(T, S)
        R = soft_argmin_on_rows(D, t)
        C = soft_argmin_on_rows(D.T, t).T
        B = np.multiply(R, C)
        loss = -np.sum(B)
        loss /= np.mean([T.shape[0],S.shape[0]])
        return loss

def hard_BD_loss(T, S):
        D = cdist(T, S)
        R = argmin_on_rows(D)
        C = argmin_on_cols(D)
        B = R * C # element-wise multiplication
        P = D * B
        loss = np.sum(P) / np.sum(B)
        return loss

def soft_BD_loss(T, S, t):
        D = cdist(T, S)
        R = soft_argmin_on_rows(D, t)
        # R = argmin_on_cols(D)
        C = soft_argmin_on_rows(D.T, t).T
        # C = argmin_on_cols(D.T)
        B = R * C # element-wise multiplication
        P = D * B
        loss = np.sum(P) / np.sum(B)
        return loss

def soft_BD_loss_torch(T, S, t, D=None):
        if D is None:
                D = cdist_torch(T, S).double()
        R = torch.squeeze(softargmin_rows_torch(D, t))
        C = torch.squeeze(softargmin_rows_torch(torch.transpose(D, dim0=0, dim1=1), t))
        C = torch.transpose(C, dim0=0, dim1=1)
        B = torch.mul(R, C)
        loss = torch.sum(torch.mul(B,D))/ (torch.sum(B))
        return loss

def soft_BBS_loss_torch(T, S, t):
        D = cdist_torch(T, S)
        R = torch.squeeze(softargmin_rows_torch(D, t))
        C = torch.squeeze(softargmin_rows_torch(torch.transpose(D, dim0=0, dim1=1), t))
        C = torch.transpose(C, dim0=0, dim1=1)
        B = torch.mul(R, C)
        loss = -torch.sum(B)
        return loss
