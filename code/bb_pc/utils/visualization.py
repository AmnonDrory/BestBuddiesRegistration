import open3d as o3d
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def draw_registration_result(A,B,title=None):
    if title is None:
        title = ''
    PC_1 = A[:,:3]
    PC_2 = B[:,:3]
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(PC_1)
    target.points = o3d.utility.Vector3dVector(PC_2)
    source_temp = deepcopy(source)
    target_temp = deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([source_temp, target_temp],window_name=title)

def DisplayPoints(A, B=None, A_emphasis=None, B_emphasis=None):
    fig = plt.gcf()
    ax = fig.add_subplot(111, projection='3d') # must uncomment import from mpl_toolkits for this to run correctly
    #ax.set_aspect('equal')

    display_with_emphasis(A, A_emphasis, ax, [0, 0, 1])
    display_with_emphasis(B, B_emphasis, ax, [1, 0, 0])

    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    set_axes_equal(ax)

def display_with_emphasis(PC, emphasis, ax, color):
    BACKGROUND_WASHOUT = 0.9
    if PC is None:
        return
    if emphasis is None:
        emphasis = np.ones(PC.shape[0], dtype=np.bool)

    front = PC[emphasis, ...]
    background = PC[~emphasis, ...]
    ax.scatter(front[:, 0], front[:, 1], front[:, 2], c=color)
    # background_color = (1-BACKGROUND_WASHOUT)*np.array(color) + BACKGROUND_WASHOUT*np.array([1.,1.,1.])
    background_color = [list(color) + [(1 - BACKGROUND_WASHOUT)]]
    ax.scatter(background[:, 0], background[:, 1], background[:, 2], c=background_color)


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
