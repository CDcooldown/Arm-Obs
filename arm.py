import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3    

# Define the modified DH parameters

# 仿真中的UR3机械臂
L1 = RevoluteDH(d=0.1519,   a=0,        alpha=np.pi/2, offset=0, modified=True)
L2 = RevoluteDH(d=0,        a=-0.24365, alpha=0,       offset=-np.pi/2, modified=True)
L3 = RevoluteDH(d=0,        a=-0.21325, alpha=0,       offset=0, modified=True)
L4 = RevoluteDH(d=0.11235,  a=0,        alpha=np.pi/2, offset=-np.pi/2, modified=True)
L5 = RevoluteDH(d=0.08535,  a=0,        alpha=-np.pi/2,offset=0, modified=True)
L6 = RevoluteDH(d=0.0817,   a=0,        alpha=0,       offset=0, modified=True)





# Create robot model
ur3 = DHRobot([L1, L2, L3, L4, L5, L6], name='UR3')
ur3_link3 = DHRobot([L1, L2], name='UR3_Link3')

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

def get_cylinder1(joint_angles):
    # Compute FK
    # T = ur3.fkine(joint_angles)  # T is SE3 object
    T = ur3_link3.fkine(joint_angles)  # T is SE3 object

    x_axis = T.R[:, 0]
    y_axis = T.R[:, 1]
    z_axis = T.R[:, 2]

    # Extract position
    pos = T.t  # numpy array of [x, y, z]

    pos2 = pos + 0.1118 * z_axis 
    print("pos2", pos2)
    pos25 = pos2 - 0.04 * x_axis 
    pos3 = pos2 + 0.24 * x_axis 
    print("pos3", pos3)

    return Cylinder(pos25, pos3, 0.06)