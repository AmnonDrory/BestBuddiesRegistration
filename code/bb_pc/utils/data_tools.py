import open3d as o3d
import copy
from plyfile import PlyData, PlyElement
import numpy as np
import os
import random
import torch

def import_ply(ply_path):
    '''
    Import ply file from ply_path
    Convert 3d coordinates to a Nx3 numpy array
    '''
    plydata = PlyData.read(ply_path)
    x = plydata.elements[0].data['x']
    y = plydata.elements[0].data['y']
    z = plydata.elements[0].data['z']
    X = np.array([x, y, z]).T
    # o3d.visualization.draw_geometries([pcd])
    return X

def calc_normals(X, knn_for_normals=13):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.01, max_nn=knn_for_normals))
    return np.asarray(pcd.normals)