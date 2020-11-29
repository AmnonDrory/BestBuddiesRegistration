import open3d as o3d
import numpy as np
import torch
from plyfile import PlyData

def calc_normals(X, knn_for_normals=13, radius=0.01):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius, max_nn=knn_for_normals))
    return np.asarray(pcd.normals)

def point_to_plane_dist_np(X, Y, X_normals, Y_normals, align_normals=True, do_sqrt=True):
    assert(X.shape[1]==3)
    assert(Y.shape[1] == 3)
    assert(X_normals.shape[1]==3)
    assert(Y_normals.shape[1]==3)

    N = X.shape[0]
    M = Y.shape[0]
    assert X_normals.shape[0]==N
    assert Y_normals.shape[0] == M
    X = X[:,np.newaxis,:]
    Y = Y[np.newaxis, :, :]
    X_normals = X_normals[:, np.newaxis, :]
    Y_normals = Y_normals[np.newaxis, :, :]
    X = np.repeat(X,M,axis=1)
    Y = np.repeat(Y, N, axis=0)
    X_normals = np.repeat(X_normals,M,axis=1)
    Y_normals = np.repeat(Y_normals, N, axis=0)

    if align_normals:
        normal_product = np.sum(X_normals*Y_normals,axis=2)
        product_sign = np.sign(normal_product)
        product_sign[product_sign==0]=1
    else:
        product_sign = np.ones([N,M],dtype=X.dtype)
    product_sign = product_sign[:,:,np.newaxis]

    normal_sum = X_normals + product_sign*Y_normals
    pts_diff = X-Y

    dot = pts_diff * normal_sum
    D = np.sum(dot**2, axis=2)
    if do_sqrt:
        D = np.sqrt(D)

    return D

def point_to_plane_dist_torch(X, Y, X_normals, Y_normals, align_normals=True, do_sqrt=True):
    assert(X.shape[1]==3)
    assert(Y.shape[1] == 3)
    assert(X_normals.shape[1]==3)
    assert(Y_normals.shape[1]==3)

    N = X.shape[0]
    M = Y.shape[0]
    assert X_normals.shape[0]==N
    assert Y_normals.shape[0] == M
    X = X.unsqueeze(1)
    Y = Y.unsqueeze(0)
    X_normals = X_normals.unsqueeze(1)
    Y_normals = Y_normals.unsqueeze(0)
    X = X.expand(N,M,3)
    Y = Y.expand(N,M,3)
    X_normals = X_normals.expand(N,M,3)
    Y_normals = Y_normals.expand(N,M,3)

    if align_normals:
        normal_product = torch.sum(X_normals*Y_normals,dim=2)
        product_sign = torch.sign(normal_product)
        product_sign[product_sign==0]=1
    else:
        product_sign = torch.ones([N,M], dtype=X.dtype, device=X.device)
    product_sign = product_sign.unsqueeze(2)

    normal_sum = X_normals + product_sign*Y_normals
    pts_diff = X-Y

    # inner product:
    inplace_mult = pts_diff * normal_sum
    dot = torch.sum(inplace_mult, dim=2)
    D = dot**2

    if do_sqrt:
        D = D.clamp_min_(1e-30).sqrt_() #

    return D

def simple_point_to_plane_dist_sparse_torch(X, Y, X_normals, do_sqrt=True):
    assert (X.shape[1] == 3)
    assert (Y.shape[1] == 3)
    assert (X_normals.shape[1] == 3)

    N = X.shape[0]
    assert Y.shape[0] == N, "point_to_plane_dist_sparse_torch calculates distances between corresponding points in two lists (and not from every point in list A and to every point in list B)"
    assert X_normals.shape[0] == N

    pts_diff = X - Y

    # inner product:
    inplace_mult = pts_diff * X_normals
    dot = torch.sum(inplace_mult, dim=1, keepdim=True)
    D = dot ** 2

    if do_sqrt:
        D = D.clamp_min_(1e-30).sqrt_()  #

    return D

def point_to_plane_dist_sparse_torch(X, Y, X_normals, Y_normals, align_normals=True, do_sqrt=True):
    assert(X.shape[1]==3)
    assert(Y.shape[1] == 3)
    assert(X_normals.shape[1]==3)
    assert(Y_normals.shape[1]==3)

    N = X.shape[0]
    M = Y.shape[2]
    assert Y.shape[0] == N, "point_to_plane_dist_sparse_torch calculates distances between corresponding points in two lists (and not from every point in list A and to every point in list B)"
    assert X_normals.shape[0]==N
    assert Y_normals.shape[0] == N

    if align_normals:
        normal_product = torch.sum(X_normals*Y_normals,dim=1)
        product_sign = torch.sign(normal_product)
        product_sign[product_sign==0]=1
    else:
        product_sign = torch.ones([N,M], dtype=X.dtype, device=X.device)
    product_sign = product_sign.unsqueeze(1)

    normal_sum = X_normals + product_sign*Y_normals
    pts_diff = X-Y

    # inner product:
    inplace_mult = pts_diff * normal_sum
    dot = torch.sum(inplace_mult, dim=1, keepdim=True)
    D = dot**2

    if do_sqrt:
        D = D.clamp_min_(1e-30).sqrt_() #

    return D

def load_points_and_normals_from_ply(path):

    plydata = PlyData.read(path)
    x = plydata.elements[0].data['x']
    y = plydata.elements[0].data['y']
    z = plydata.elements[0].data['z']
    nx = plydata.elements[0].data['nx']
    ny = plydata.elements[0].data['ny']
    nz = plydata.elements[0].data['nz']
    P = np.array([x,y,z]).T
    N = np.array([nx,ny,nz]).T

    return P, N

