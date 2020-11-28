import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import open3d as o3d
import torch
from torch.autograd import Variable
# import trimesh
import random


def naive_point_to_plane_dist_np(X, Y, Y_normals, X_normals):

    # d(x,y) = (x-y)^T dot (n_y + n_x)
    N = X.shape[1]
    D = np.zeros((N, N))
    for i, x in enumerate(X.T):
        for j, y in enumerate(Y.T):
            nx = X_normals[:,i]
            ny = Y_normals[:,j]
            D[i, j] = np.sum((x-y) * (nx + ny))
    return D


def point_to_plane_dist_np(X, Y, X_normals, Y_normals):

    # d(x,y) = (x-y)^T dot (n_y + n_x)
    N = X.shape[1]
    X = np.expand_dims(X, axis=2)
    X1 = np.repeat(X, N, axis=2)
    Y = np.expand_dims(Y, axis=2)
    Y1 = np.repeat(Y, N, axis=2).transpose(0, 2, 1)
    pts = X1 - Y1 #[3xNxN]
    normals_Y = np.expand_dims(Y_normals, axis=2)
    normals_Y = np.repeat(normals_Y, N, axis=2).transpose(0, 2, 1)
    normals_X = np.expand_dims(X_normals, axis=2)
    normals_X = np.repeat(normals_X, N, axis=2)
    normals = normals_Y + normals_X
    dot = pts * normals
    # D = np.sqrt(np.sum(pts**2, axis=0)) # Auclidian Distance OK
    D = np.sum(dot**2, axis=0)
    res = D
    return res


def euler_to_quaternion(yaw, pitch, roll):

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]

def quaternion_to_euler(x, y, z, w):
        'Euler angles output are in deg'
        import math
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = math.degrees(math.atan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.degrees(math.asin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = math.degrees(math.atan2(t3, t4))

        return X, Y, Z

def square_dist(A, B):

    AA = np.sum(A**2, axis=1, keepdims=True)
    BB = np.sum(B**2, axis=1, keepdims=True)
    inner = np.matmul(A,B.T)

    R = AA + (-2)*inner + BB.T

    return R

def calc_sizes(A,B):

    A_bs_center, A_bs_r = get_bounding_sphere(A)
    B_bs_center, B_bs_r = get_bounding_sphere(B)
    center_of_rotation = 0.5 * (A_bs_center + B_bs_center)
    PC_size = A_bs_r + B_bs_r
    typical_nei_dist = representative_neighbor_dist(dist(A,A))
    return PC_size, center_of_rotation, typical_nei_dist


def random_unit_vector():

    uv = np.random.rand(2)
    phi = uv[0] * 2 * np.pi
    z = uv[1] * 2 - 1
    xy = np.sqrt(1. - z**2)
    res = np.array([ xy * np.cos(phi),
            xy * np.sin(phi),
            z])
    return res

def dist(A,B):

    s = square_dist(A,B)
    s[s<0]=0
    return np.sqrt(s)

def from_quat_to_rot_matrix(angle, n, deg_or_rads='deg'):
    'angle - the rotation angle size'
    'n - the axis of rotation'
    if deg_or_rads is 'deg':
        angle = angle/180*np.pi
    e1 = n[0]
    e2 = n[1]
    e3 = n[2]
    A11 = (1-np.cos(angle))*np.power(e1,2)+np.cos(angle)
    A12 = (1-np.cos(angle))*e1*e2-e3*np.sin(angle)
    A13 = (1-np.cos(angle))*e1*e3+e2*np.sin(angle)
    A21 = (1-np.cos(angle))*e1*e2+e3*np.sin(angle)
    A22 = (1-np.cos(angle))*np.power(e2,2)+np.cos(angle)
    A23 = (1-np.cos(angle))*e2*e3-e1*np.sin(angle)
    A31 = (1-np.cos(angle))*e1*e3-e2*np.sin(angle)
    A32 = (1-np.cos(angle))*e3*e2+e1*np.sin(angle)
    A33 = (1-np.cos(angle))*np.power(e3,2)+np.cos(angle)
    R = np.asarray([[A11, A12, A13],[A21, A22, A23],[A31, A32, A33]])
    return np.squeeze(R)

def rotate_around_axis(PC, n, angle):

    if PC.shape[0] is not 3:
        PC = PC.T
    R = from_quat_to_rot_matrix(angle, n)
    newPC = rotate_3d(PC, R)
    return newPC, R

# def angular_error(vec1, vec2):
def angular_error(R1, R2):
    # if np.shape(vec1)[0] is not 3:
    #     vec1 = vec1.T
    # if np.shape(vec2)[0] is not 3:
    #     vec2 = vec2.T
    # R1 = euler_angles_to_rotation_matrix(vec1)
    # R2 = euler_angles_to_rotation_matrix(vec2)
    norm = Frobenius_Norm(R1 - R2)
    theta = 2 * np.arcsin(norm/np.sqrt(8)) * 180 / np.pi
    return theta

def Frobenius_Norm(R):

    norm = np.sqrt(np.trace(np.matmul(R, R.T)))
    return norm

def representative_neighbor_dist(D):

    m = min_without_self_per_row(D)
    neighbor_dist = np.median(m)
    return neighbor_dist

def rotate_around_axis_and_center_of_rotation(PC, center_point_of_rotation, axis_of_rotation, angle):

    input_was_transposed = False

    if PC.shape[0] is not 3:
        input_was_transposed = True
        PC = PC.T

    center_point_of_rotation = center_point_of_rotation.reshape([-1, 1])
    PC_centered = PC - center_point_of_rotation

    new_PC_centered, R = rotate_around_axis(PC_centered, axis_of_rotation, angle)
    angle = rotation_matrix_to_euler_angles(R) * 180 / np.pi
    newPC = new_PC_centered + center_point_of_rotation

    if input_was_transposed:
        newPC = newPC.T
    return newPC, R, angle

# def get_bounding_sphere(PC):
#     bsphere = trimesh.PointCloud(PC).bounding_sphere
#     radius = bsphere.primitive.radius
#     center =  bsphere.primitive.center
#     return center, radius

def sample_pc(pc, n_samples):
    # downsample
    pcd_A = o3d.geometry.PointCloud()
    pcd_A.points = o3d.utility.Vector3dVector(pc)
    downsampled_A = voxel_filter(pcd_A, n_samples)
    A = pad_with_samples(downsampled_A, pcd_A, n_samples)
    return A

def pad_pc_with_samples(pc, pc_samples, N):
    # pad pcd with samples from pcd_samples to the size N.
    assert(np.shape(pc)[1] is 3)
    assert(np.shape(pc_samples)[1] is 3)
    K = np.shape(pc)[0]
    samples =  np.random.choice(np.shape(pc)[0], N-K)
    pcd_xyz = np.asarray(pc)
    pcd_samples_xyz = np.asarray(pc_samples)
    padded = np.concatenate((pcd_xyz, pcd_samples_xyz[samples]))
    return padded

def pad_with_samples(pcd, pcd_samples, N):
    # pad pcd with samples from pcd_samples to the size N.
    assert(np.shape(pcd.points)[1] is 3)
    assert(np.shape(pcd_samples.points)[1] is 3)
    K = np.shape(pcd.points)[0]
    samples =  np.random.choice(np.shape(pcd_samples.points)[0], N-K)
    pcd_xyz = np.asarray(pcd.points)
    pcd_samples_xyz = np.asarray(pcd_samples.points)
    padded = np.concatenate((pcd_xyz, pcd_samples_xyz[samples]))
    return padded

def voxel_filter(pcd, N):
    # pcd is of open3d point cloud class
    K = np.shape(pcd.points)[0]
    vs = 1e-3
    while K>N:
        pcd = o3d.geometry.voxel_down_sample(pcd, voxel_size=vs)
        vs += 1e-5
        K = np.shape(pcd.points)[0]
    return pcd

def euclidian_distance(x1, x2):
    '''3D Euclidian distance'''
    return np.linalg.norm(x1-x2)


def plot_3d(coords, figure=1, show=True, fig=None):
    markers = ['o','^','*','-']
    colors = ['r','b','g','y']
    if np.shape(coords.shape)[0] < 3:
        N = 1
        coords = np.expand_dims(coords, axis=0)
    else:
        N = coords.shape[0]
    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot(111,projection='3d') # must uncomment import from mpl_toolkits for this to run correctly
    for n in range(N):
        x = coords[n, 0, :]
        y = coords[n, 1, :]
        z = coords[n, 2, :]
        ax.scatter(x, y, z, marker=markers[n], c=colors[n])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    if show:
        plt.show()
    return fig, ax



def rotate_3d(X,R):
    # assert(isRotationMatrix(R))
    'X is a matrix of 3xN'
    'R is a rotation matrix 3x3'

    if X.shape[0] is not 3:
        X = X.T
    Y = np.matmul(R, X)
    return Y

def rotate_around_random_axis(PC, angle_size, n=None):
    if n is None:
        n = random_unit_vector()
    new_PC, R = rotate_around_axis(PC, n, angle_size)
    angle = rotation_matrix_to_euler_angles(R) * 180 / np.pi
    new_PC = new_PC.T
    R = R.T
    return new_PC, angle, n, R


def euler_angles_to_rotation_matrix(theta_vec, deg_or_rad='deg'):
    if np.shape(theta_vec)[0] is not 3:
        theta_vec = theta_vec.T
    if deg_or_rad is 'deg':
        theta_vec = np.radians(theta_vec)
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta_vec[0]), -math.sin(theta_vec[0]) ],
                    [0,         math.sin(theta_vec[0]), math.cos(theta_vec[0])  ]
                    ])
    R_y = np.array([[math.cos(theta_vec[1]),    0,      math.sin(theta_vec[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta_vec[1]),   0,      math.cos(theta_vec[1])  ]
                    ])
    R_z = np.array([[math.cos(theta_vec[2]),    -math.sin(theta_vec[2]),    0],
                    [math.sin(theta_vec[2]),    math.cos(theta_vec[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotation_matrix_to_euler_angles(R) :

    # assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def plot_A_B(A,B, show=True):
    if A.shape[0] is not 3:
        A = A.T
    if B.shape[0] is not 3:
        B = B.T
    coords = np.concatenate([np.expand_dims(A, axis=0), np.expand_dims(B, axis=0)])
    _,ax = plot_3d(coords, show=show)
    return ax

def plot_A_B_new(A,B, ax=None, show=True, colors=None):
    if A.shape[0] is not 3:
        A = A.T
    if B.shape[0] is not 3:
        B = B.T
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d') # must uncomment import from mpl_toolkits for this to run correctly
    if colors is None:
        for i in range(A.shape[1]):
            c = np.random.rand(3,)
            ax.scatter(A[0,i], A[1,i], A[2,i], marker='*', c=c)
            ax.scatter(B[0,i], B[1,i], B[2,i], marker='o', c=c)
    else:
        ax.scatter(A[0,:], A[1,:], A[2,:], marker='^', c=colors[0])
        ax.scatter(B[0,:], B[1,:], B[2,:], marker='^', c=colors[1])
    if show:
        plt.show()
    return ax

def pc_size(PC):
    if PC.shape[1] is not 3:
        PC = PC.T
    range = np.max(PC,axis=0) - np.min(PC,axis=0)
    return np.linalg.norm(range)

def apply_random_rot_trans(PC, rot_range, trans_range):
    vec = np.random.uniform(-rot_range,rot_range,3)
    R = euler_angles_to_rotation_matrix(vec)
    T = np.random.uniform(-trans_range,trans_range,3)
    return rotate_3d(PC, R).T + T.T, vec, T

def apply_rot(PC, vec):
    R = euler_angles_to_rotation_matrix(vec)
    return rotate_3d(PC, R)

def apply_rot_trans(PC, R, T):
    if T.shape[0] is not 3:
        T = T.T
    newPC = rotate_3d(PC, R) + T
    return newPC

def dilate_randomly(PC, K):
    if PC.shape[0] is not 3:
        PC = PC.T
    if K>=PC.shape[1]:
        return PC, np.linspace(0, K-1, K, dtype=int)
    if K is not 0:
        samples = random.sample(population=range(PC.shape[1]), k=K)
        return PC[:,samples], samples
    else:
        return []

def argmin_on_cols(X):
    one_hot_y = np.zeros_like(X)
    Y = np.argmin(X, axis=0)
    one_hot_y[np.arange(X.shape[1]),Y] = 1
    return one_hot_y

def combined_hard_loss(A, B, A_normals, B_normals):
    D = cdist(torch.tensor(A), torch.tensor(B))
    D = D.cpu().data.numpy()
    R = argmin_on_cols(D)
    C = argmin_on_cols(D.T).T
    Q = np.multiply(R, C)
    D = point_to_plane_dist_np(A, B, A_normals, B_normals)
    loss = np.sum(Q * D) / np.sum(Q)
    return loss

def one_hot_to_index(Q):
    if torch.is_tensor(Q):
        Q = Q.cpu().data.numpy()
        Q = np.squeeze(Q)
    N = Q.shape[0]
    indexes = np.expand_dims(np.linspace(0, N-1, N, dtype=int), axis=1)
    idx_x = np.repeat(indexes, N, axis=1)
    idx_y = idx_x.T
    idx_x = idx_x[Q==1]
    idx_y = idx_y[Q==1]
    return idx_x, idx_y

def get_best_buddies_pairs(A, B, A_normals, B_normals):
    # find points in A and B which are the closest from A to B and from B to A
    # D = cdist(torch.tensor(A), torch.tensor(B))
    D = point_to_plane_dist_np(A, B, A_normals, B_normals)
    # D = D.cpu().data.numpy()
    R = argmin_on_cols(D)
    C = argmin_on_cols(D.T).T
    Q = np.multiply(R, C)
    new_A_idx, new_B_idx = one_hot_to_index(Q)
    # new_B_indexes = np.argmax(Q, axis=1)
    # new_A_indexes = np.argmax(Q, axis=0)
    new_B = B[:, new_B_idx]
    new_A = A[:, new_A_idx]
    new_B_normals = B_normals[:, new_B_idx]
    new_A_normals = A_normals[:, new_A_idx]
    return new_A, new_B, new_A_normals, new_B_normals

def hard_loss(A, B, A_normals, B_normals, use_normals = True):
    if use_normals:
        D = point_to_plane_dist_np(A, B, A_normals, B_normals)
        # D_ = naive_point_to_plane_dist_np(A, B, A_normals, B_normals)
        # new_A_BBS, new_B_BBS, new_A_normals_BBS, new_B_normals_BBS = get_best_buddies_pairs(A, B, A_normals, B_normals)
        # loss = -new_A_BBS.shape[1]
    else:
        D = cdist(torch.tensor(A), torch.tensor(B))
        D = D.cpu().data.numpy()
    R = argmin_on_cols(D)
    C = argmin_on_cols(D.T).T
    Q = np.multiply(R, C)
    # loss = np.sum(Q * D) / np.sum(Q)
    loss = -np.sum(Q)
    return loss

def point_to_plane_dist(X, Y, X_normals, Y_normals):

    # d(x,y) = (Rx+t-y)^T dot (n_y + n_x)
    N = X.shape[1]
    X = torch.unsqueeze(X, dim=2)
    X1 = X.repeat((1, 1, N))
    Y = torch.unsqueeze(Y, dim=2)
    Y1 = Y.repeat((1, 1, N)).permute(0, 2, 1)
    pts = X1 - Y1  # [3xNxN]
    X_normals = torch.unsqueeze(X_normals, dim=2)
    X_normals1 = X_normals.repeat((1, 1, N))
    Y_normals = torch.unsqueeze(Y_normals, dim=2)
    Y_normals1 = Y_normals.repeat((1, 1, N)).permute(0, 2, 1)
    normals = X_normals1 + Y_normals1
    dot = torch.mul(pts, normals)
    res = torch.sum(dot, dim=0) ** 2
    values = res.cpu().data.numpy().flatten()
    std3 = 3 * values.std()
    res[torch.where(res > std3)] = std3
    res = res.clamp_min_(1e-30).sqrt_()
    return res
