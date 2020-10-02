import numpy as np
import torch
from ..utils.consts import EPS

def soft_argmin_on_rows(X, t):
    y = []
    for x in X:
        y.append(np.exp(-x/t)/(np.sum(np.exp(-x/t),axis=0)+EPS))
    return np.array(y)

def my_softmax(x, eps = EPS, dim=0):
    x_exp = torch.exp(x)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp/(x_exp_sum + eps)

def softargmin_rows_torch(X, t, eps=EPS):
    t = t.double()
    X = X.double()
    weights = my_softmax(-X/t, eps=EPS, dim=1)
    return weights

def softargmin_cols_torch(X, t, eps=EPS):
    t = t.double()
    X = X.double()
    weights = my_softmax(-X/t, eps=EPS, dim=0)
    return weights

def argmin_on_cols(X):
    one_hot_y = np.zeros_like(X)
    Y = np.argmin(X, axis=0)
    one_hot_y[Y,np.arange(X.shape[1])] = 1
    return one_hot_y

def argmin_on_rows(X):
    one_hot_y = np.zeros_like(X)
    Y = np.argmin(X, axis=1)
    one_hot_y[np.arange(X.shape[0]),Y] = 1
    return one_hot_y

def argmin_on_cols_torch(X):
    one_hot_y = torch.zeros_like(X)
    Y = torch.argmin(X, dim=0)
    one_hot_y[Y,torch.arange(X.shape[1])] = 1
    return one_hot_y

