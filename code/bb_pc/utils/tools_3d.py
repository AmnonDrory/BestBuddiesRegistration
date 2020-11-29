import numpy as np
import math
import torch
from torch.autograd import Variable
import random
from ..utils.subsampling import num_features

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


def calc_translational_error(GT_mot, res_mot):
    err = np.linalg.norm(GT_mot['trans'].flatten() - res_mot['trans'].flatten())
    return err

def calc_angular_error(vec1, vec2):
    R1 = euler_angles_to_rotation_matrix(vec1)
    R2 = euler_angles_to_rotation_matrix(vec2)
    norm = Frobenius_Norm(R1 - R2)
    theta = np.rad2deg( 2 * np.arcsin(norm/np.sqrt(8)) )
    return theta

def Frobenius_Norm(R):
    norm = np.sqrt(np.trace(np.matmul(R, R.T)))
    return norm

def apply_rot_trans_torch(X,R,T):
    res = rotate_3d_torch(X,R) + T.double()
    return res

def rotate_3d_torch(X,R):
    'X is a matrix of 3xN'
    'R is a rotation matrix 3x3'
    if(X.shape[0] is not 3):
        X = torch.transpose(X, dim0=0, dim1=1)
    Y = torch.matmul(R.double(), X.double())
    return Y

def rotate_3d(X, R):
    """

    :param X: a matrix of 3xN or Nx3
    :param R: a rotation matrix 3x3, or a vector of 3 euler angles
    :return:
    """
    R = np.array(R) # to handle lists
    if R.size == 3:
        R = euler_angles_to_rotation_matrix(R)
    assert len(R.shape)==2 and (R.shape[0] == 3) and (R.shape[1] == 3), 'Error: input should be rotation matrix, but has a shape: ' + str(R.shape)
    trans_flag = False
    if X.shape[0] is not 3:
        X = X.T
        trans_flag = True
    Y = np.matmul(R, X)
    if trans_flag:
        Y = Y.T
    return Y

def euler_angles_to_rotation_matrix_torch(theta, phi, psi):
    if torch.cuda.is_available():
        one = Variable(torch.ones(1, dtype=theta.dtype)).cuda()
        zero = Variable(torch.zeros(1, dtype=theta.dtype)).cuda()
    else:
        one = Variable(torch.ones(1, dtype=theta.dtype))
        zero = Variable(torch.zeros(1, dtype=theta.dtype))
    rot_x = torch.cat((
        torch.unsqueeze(torch.cat((one, zero, zero), 0), dim=1),
        torch.unsqueeze(torch.cat((zero, theta.cos(), theta.sin()), 0), dim=1),
        torch.unsqueeze(torch.cat((zero, -theta.sin(), theta.cos()), 0), dim=1),
    ), dim=1)
    rot_y = torch.cat((
        torch.unsqueeze(torch.cat((phi.cos(), zero, -phi.sin()), 0), dim=1),
        torch.unsqueeze(torch.cat((zero, one, zero), 0), dim=1),
        torch.unsqueeze(torch.cat((phi.sin(), zero, phi.cos()), 0), dim=1),
    ), dim=1)
    rot_z = torch.cat((
        torch.unsqueeze(torch.cat((psi.cos(), psi.sin(), zero), 0), dim=1),
        torch.unsqueeze(torch.cat((-psi.sin(), psi.cos(), zero), 0), dim=1),
        torch.unsqueeze(torch.cat((zero, zero, one), 0), dim=1),
    ), dim=1)
    A = torch.mm(rot_z, torch.mm(rot_y, rot_x))
    if torch.cuda.is_available():
        A = A.cuda()
    return A

def euler_angles_to_rotation_matrix(theta_vec, deg_or_rad='deg'):
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

def rotation_matrix_euler_angles(R, deg_or_rads='deg') :

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

    res= np.array([x, y, z])
    if deg_or_rads=="deg":
        res = np.degrees(res)

    return res

def calc_inverse_motion(mot):
    R = euler_angles_to_rotation_matrix(mot['angles'])
    T = mot['trans'].reshape([-1,1])
    # y = Rx + T
    # R.T(y - T) = x
    # x = R.T*y + (-R.T*T)
    inverse_R = R.T
    inverse_T = -np.matmul(inverse_R,T)
    inverse_mot = {}
    inverse_mot['angles'] = rotation_matrix_euler_angles(inverse_R)
    inverse_mot['trans'] = inverse_T.flatten()
    return inverse_mot

def apply_rot_trans(X,R,T):

    if np.array(R).size == 3:
        R = euler_angles_to_rotation_matrix(R)

    is_transposed = X.shape[0] != num_features

    if is_transposed:
        X = X.T

    T = np.reshape(T, [3,1])

    res = rotate_3d(X,R) + T

    if is_transposed:
        res = res.T

    return res


def subsample_randomly(PC, K, axis=0):

    if K>=np.max((PC.shape[0], PC.shape[1])):
        return PC
    if axis is 0:
        if K is not 0:
            samples = random.sample(population=range(PC.shape[0]), k=K)
            return PC[samples]
        else:
            return []
    else:
        if K is not 0:
            samples = random.sample(population=range(PC.shape[1]), k=K)
            return PC[:,samples]
        else:
            return []

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

