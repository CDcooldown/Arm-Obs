import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
import matplotlib.pyplot as plt

from plot import plot_cylinder


class Cylinder:
    def __init__(self, p1, p2, r):
        """
        p1: np.array([x, y, z]) — bottom center of the cylinder
        p2: np.array([x, y, z]) — top center of the cylinder
        r: float — radius
        """
        self.p1 = np.array(p1, dtype=float)
        self.p2 = np.array(p2, dtype=float)
        self.r = float(r)

def ur3_fk(joint_angles):
    """
    Compute UR3 forward kinematics using modified DH parameters.
    :param joint_angles: List or array of 6 joint angles (in radians)
    :return: position (x, y, z) and orientation (alpha, beta, gamma) in radians, ZYX Euler
    """

    # Define the modified DH parameters
    L1 = RevoluteDH(d=0.1519,   a=0,        alpha=np.pi/2, offset=0, modified=True)
    L2 = RevoluteDH(d=0,        a=-0.24365, alpha=0,       offset=-np.pi/2, modified=True)
    L3 = RevoluteDH(d=0,        a=-0.21325, alpha=0,       offset=0, modified=True)
    L4 = RevoluteDH(d=0.11235,  a=0,        alpha=np.pi/2, offset=-np.pi/2, modified=True)
    L5 = RevoluteDH(d=0.08535,  a=0,        alpha=-np.pi/2,offset=0, modified=True)
    L6 = RevoluteDH(d=0.0817,   a=0,        alpha=0,       offset=0, modified=True)


    # Create robot model
    # ur3 = DHRobot([L1, L2, L3, L4, L5, L6], name='UR3')
    ur3_link2 = DHRobot([L1, L2], name='UR3_Link2')
    # ur3_link1 = DHRobot([L1], name='UR3_Link1')


    # Compute FK
    # T = ur3.fkine(joint_angles)  # T is SE3 object
    T = ur3_link2.fkine(joint_angles)  # T is SE3 object

    x_axis = T.R[:, 0]
    y_axis = T.R[:, 1]
    z_axis = T.R[:, 2]

    # Extract position
    pos = T.t  # numpy array of [x, y, z]

    # Extract orientation as ZYX Euler angles (gamma, beta, alpha)
    eul = T.rpy(order='xyz', unit='rad')  # returns [gamma, beta, alpha]

    

    return pos, eul, x_axis, y_axis, z_axis

if __name__ == '__main__':
    # Example joint angles in radians
    # q = [0.57, 0.7, 0, 0, 0, 0]
    q = [0.57, 0]

    # q = [0, -np.pi/4, np.pi/2, 0, np.pi/4, 0]
    # q = [0, -np.pi/4, np.pi/2, 0.3, np.pi/3, 0.3]
    # q = [0, 0, 0, 0, 0, 0] 

    # q = [0.57]

    

    pos, eul_0, x, y, z = ur3_fk(q)

    eul = eul_0 * 180 / np.pi
    
    pos2 = pos + 0.1118 * z 
    print("pos2", pos2)
    pos3 = pos2 + 0.24 * x
    print("pos3", pos3)

    cylinder = Cylinder(pos2, pos3, 0.05)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_cylinder(ax, cylinder)
    plt.show()


    print('End-effector pose from FK:')
    print(f'Position: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}')
    # print(f'Orientation (ZYX Euler): alpha={eul[2]:.4f}, beta={eul[1]:.4f}, gamma={eul[0]:.4f}')
    print("End-effector x-axis direction vector:", x)
    print("End-effector y-axis direction vector:", y)
    print("End-effector z-axis direction vector:", z)
