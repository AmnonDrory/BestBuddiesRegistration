import numpy as np
import torch

def square_dist(A, B):
    """
    Measure Squared Euclidean Distance from every point in point-cloud A, to every point in point-cloud B
    :param A: Point Cloud: Nx3 Array of real numbers, each row represents one point in x,y,z space
    :param B: Point Cloud: Mx3 Array of real numbers
    :return:  NxM array, where element [i,j] is the squared distance between the i'th point in A and the j'th point in B
    """

    AA = np.sum(A**2, axis=1, keepdims=True)
    BB = np.sum(B**2, axis=1, keepdims=True)
    inner = np.matmul(A,B.T)

    R = AA + (-2)*inner + BB.T

    return R

def new_cdist(x1, x2):
        x1 = x1.float()
        x2 = x2.float()
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True).float()
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True).float()
        res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add(x1_norm)
        res = res.clamp_min_(1e-30).sqrt_()
        return res

def PC_to_file(PC, filename):
    """
    Write point cloud to file

    :param PC: Point Cloud: an Nx3 array of real numbers
    :param filename: Name of file to write into
    """
    with open(filename, 'w') as fid:
        fid.write('%d\n' % PC.shape[0])
        for i in range(PC.shape[0]):
            fid.write("% 20.17f % 20.17f % 20.17f\n" % (PC[i, 0], PC[i, 1], PC[i, 2]))


def file_to_PC(filename):
    """
    Load point cloud from file

    :param filename: File to load from
    :return: Point Cloud: an Nx3 array of real numbers
    """
    with open(filename, 'r') as fid:
        text = fid.read().splitlines()

    num_points = int(text[0])
    PC = np.zeros([num_points, 3], dtype=float)

    for i, line in enumerate(text[1:]):
        PC[i, :] = np.array([float(x) for x in line.split()])

    return PC

def cdist(A, B):
        from ..utils.subsampling import num_features
        if (A.shape[1] is not num_features):
            A = A.T
        if (B.shape[1] is not num_features):
            B = B.T
        C = dist(A, B)
        return C

def dist(A,B):
    """
    Measure Squared Euclidean Distance from every point in point-cloud A, to every point in point-cloud B
    :param A: Point Cloud: Nx3 Array of real numbers, each row represents one point in x,y,z space
    :param B: Point Cloud: Mx3 Array of real numbers
    :return:  NxM array, where element [i,j] is the squared distance between the i'th point in A and the j'th point in B
    """
    s = square_dist(A,B)
    s[s<0]=0
    return np.sqrt(s)

def cdist_torch(A,B):
    from ..utils.subsampling import num_features
    if (A.shape[1] is not num_features):
        A = torch.transpose(A, dim0=0, dim1=1)
    if (B.shape[1] is not num_features):
        B = torch.transpose(B, dim0=0, dim1=1)
    A = A.double().contiguous()
    B = B.double().contiguous()
    C = new_cdist(A,B)
    return C

def min_without_self_per_row(D):
    """
    Accepts a distance matrix between all points in a set. For each point,
    returns its distance from the closest point that is not itself.

    :param D: Distance matrix, where element [i,j] is the distance between i'th point in the set and the j'th point in the set. Should be symmetric with zeros on the diagonal.
    :return: vector of distances to nearest neighbor for each point.
    """
    E = D.copy()
    for i in range(E.shape[0]):
        E[i,i] = np.inf
    m = np.min(E,axis=1)
    return m

def representative_neighbor_dist(D):
    """
    Accepts a distance matrix between all points in a set,
    returns a number that is representative of the distances in this set.

    :param D: Distance matrix, where element [i,j] is the distance between i'th point in the set and the j'th point in the set. Should be symmetric with zeros on the diagonal.
    :return: The representative distance in this set
    """

    assert D.shape[0] == D.shape[1], "Input to representative_neighbor_dist should be a matrix of distances from a point cloud to itself"
    m = min_without_self_per_row(D)
    neighbor_dist = np.median(m)
    return neighbor_dist


