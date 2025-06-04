import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3   
import matplotlib.pyplot as plt
from plot import *

# Define the modified DH parameters

# 实际使用的大象P3机械臂
L1 = RevoluteDH(d=0.1110,   a=0,        alpha=np.pi/2, offset=np.pi, modified=True)
L2 = RevoluteDH(d=0,        a=-0.264,   alpha=0,       offset=0, modified=True)
L3 = RevoluteDH(d=0,        a=-0.236,   alpha=0,       offset=0, modified=True)
L4 = RevoluteDH(d=0.1138,   a=0,        alpha=np.pi/2, offset=0, modified=True)
L5 = RevoluteDH(d=0.1018,   a=0,        alpha=-np.pi/2,offset=0, modified=True)
L6 = RevoluteDH(d=0.090238,  a=0,        alpha=0,      offset=0, modified=True)


# Create robot model
ur3 = DHRobot([L1, L2, L3, L4, L5, L6], name='UR3')
ur3_link1 = DHRobot([L1], name='UR3_Link1')
ur3_link2 = DHRobot([L1, L2], name='UR3_Link2')
ur3_link3 = DHRobot([L1, L2, L3], name='UR3_Link3')
ur3_link4 = DHRobot([L1, L2, L3, L4], name='UR3_Link4')
ur3_link5 = DHRobot([L1, L2, L3, L4, L5], name='UR3_Link5')

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

class Box:
    def __init__(self, center, size):
        """
        center: np.array([x, y, z]) — center of the box
        size: np.array([sx, sy, sz]) — dimensions of the box
        """
        self.center = np.array(center, dtype=float)
        self.size = np.array(size, dtype=float)

    def bounds(self):
        half = self.size / 2
        return self.center - half, self.center + half
    
def get_cylinder1(joint_angles):
    angles = joint_angles[:1]
    # Compute FK
    T = ur3_link1.fkine(angles)  # T is SE3 object

    x_axis = T.R[:, 0]
    y_axis = T.R[:, 1]
    z_axis = T.R[:, 2]

    # Extract position

    pos = T.t  
    # print("Link1_pos1", pos)

    pos2 = pos + 0.190 * z_axis 
    # print("Link1_pos2", pos2)

    return Cylinder(pos, pos2, 0.05)

def get_cylinder2(joint_angles):
    angles = joint_angles[:2]

    # Compute FK
    T = ur3_link2.fkine(angles)  # T is SE3 object

    x_axis = T.R[:, 0]
    y_axis = T.R[:, 1]
    z_axis = T.R[:, 2]
    # print("x axis for link2", x_axis)
    # print("z axis for link2", z_axis)

    # Extract position
    pos = T.t  # numpy array of [x, y, z]
    # print("pos for link2", pos)

    pos05 = pos + 0.1238 * z_axis

    pos1 = pos05 - 0.00 * x_axis
    # print("Link2_pos1", pos1)

    pos2 = pos05 + 0.264 * x_axis
    # print("Link2_pos2", pos2)

    return Cylinder(pos1, pos2, 0.05)

def get_cylinder3(joint_angles):
    angles = joint_angles[:2]

    # Compute FK
    T = ur3_link2.fkine(angles)  # T is SE3 object

    x_axis = T.R[:, 0]
    y_axis = T.R[:, 1]
    z_axis = T.R[:, 2]

    # Extract position
    pos = T.t  # numpy array of [x, y, z]
    pos1 = pos - 0.05 * z_axis
    # print("pos for link3", pos1)
    pos2 = pos + (0.1138 + 0.05) * z_axis

    return Cylinder(pos1, pos2, 0.05)

def get_cylinder4(joint_angles):
    angles = joint_angles[:3]

    # Compute FK
    T = ur3_link3.fkine(angles)  # T is SE3 object

    x_axis = T.R[:, 0]
    y_axis = T.R[:, 1]
    z_axis = T.R[:, 2]

    # print("    x axis for link3", x_axis)
    # print("    z axis for link3", z_axis)

    # Extract position
    pos = T.t  # numpy array of [x, y, z]

    pos05 = pos + 0.001 * z_axis

    pos1 = pos05 - 0.00 * x_axis
    # print("Cylinder4_pos1", pos1)

    pos2 = pos05 + 0.236 * x_axis
    # print("Cylinder4_pos2", pos2)

    return Cylinder(pos1, pos2, 0.05)

def get_cylinder5(joint_angles):
    angles = joint_angles[:3]

    # Compute FK
    T = ur3_link3.fkine(angles)  # T is SE3 object

    x_axis = T.R[:, 0]
    y_axis = T.R[:, 1]
    z_axis = T.R[:, 2]

    # print("    x axis for link3", x_axis)
    # print("    z axis for link3", z_axis)

    # Extract position
    pos = T.t  # numpy array of [x, y, z]
    pos1 = pos -0.05 * z_axis

    pos2 = pos + 0.1138 * z_axis

    # print("Cyl5_pos1", pos1)

    # print("Cyl5_pos2", pos2)

    return Cylinder(pos1, pos2, 0.05)

def get_cylinder6(joint_angles):
    angles = joint_angles[:4]

    # Compute FK
    T = ur3_link4.fkine(angles)  # T is SE3 object

    x_axis = T.R[:, 0]
    y_axis = T.R[:, 1]
    z_axis = T.R[:, 2]

    # print("    x axis for link4", x_axis)
    # print("    z axis for link4", z_axis)

    # Extract position
    pos = T.t  # numpy array of [x, y, z]
    pos1 = pos - 0.05 * z_axis

    pos2 = pos + 0.1138 * z_axis

    # print("Cyl6_pos1", pos1)

    # print("Cyl6_pos2", pos2)

    return Cylinder(pos1, pos2, 0.05)

def get_cylinder7(joint_angles):
    angles = joint_angles[:5]

    # Compute FK
    T = ur3_link5.fkine(angles)  # T is SE3 object

    x_axis = T.R[:, 0]
    y_axis = T.R[:, 1]
    z_axis = T.R[:, 2]

    # print("    x axis for link5", x_axis)
    # print("    z axis for link5", z_axis)

    # Extract position
    pos = T.t  # numpy array of [x, y, z]
    pos1 = pos - 0.05 * z_axis

    pos2 = pos + 0.1838 * z_axis

    # print("Cyl7_pos1", pos1)

    # print("Cyl7_pos2", pos2)

    return Cylinder(pos1, pos2, 0.05)

def get_cylinders(joint_angles):
    return [
        # get_cylinder1(joint_angles),
        # get_cylinder2(joint_angles),
        get_cylinder3(joint_angles),
        get_cylinder4(joint_angles),
        get_cylinder5(joint_angles),
        get_cylinder6(joint_angles),
        get_cylinder7(joint_angles)
    ]

def plot_cylinders_and_boxes(joint_angles, boxes):
    cylinder0 = Cylinder([0, 0, 0], [0, 0, 0.2], 0.05)
    cylinder1 = get_cylinder1(joint_angles)
    cylinder2 = get_cylinder2(joint_angles)
    cylinder3 = get_cylinder3(joint_angles)
    cylinder4 = get_cylinder4(joint_angles)
    cylinder5 = get_cylinder5(joint_angles)
    cylinder6 = get_cylinder6(joint_angles)
    cylinder7 = get_cylinder7(joint_angles)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_cylinder(ax, cylinder0)
    plot_cylinder(ax, cylinder1)
    plot_cylinder(ax, cylinder2)
    plot_cylinder(ax, cylinder3)
    plot_cylinder(ax, cylinder4)
    plot_cylinder(ax, cylinder5)
    plot_cylinder(ax, cylinder6)
    plot_cylinder(ax, cylinder7)
    for box in boxes:
        plot_box(ax, box)
    
    # plot_cylinder(ax, [0.00823718, 0.19858176, 0.31871104],  [-0.09051886,  0.06896938,  0.111], radius=0.05)
    plt.show()

def plot_cylinders(joint_angles):
    cylinder0 = Cylinder([0, 0, 0], [0, 0, 0.2], 0.05)
    cylinder1 = get_cylinder1(joint_angles)
    cylinder2 = get_cylinder2(joint_angles)
    cylinder3 = get_cylinder3(joint_angles)
    cylinder4 = get_cylinder4(joint_angles)
    cylinder5 = get_cylinder5(joint_angles)
    cylinder6 = get_cylinder6(joint_angles)
    cylinder7 = get_cylinder7(joint_angles)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_cylinder(ax, cylinder0)
    plot_cylinder(ax, cylinder1)
    plot_cylinder(ax, cylinder2)
    plot_cylinder(ax, cylinder3)
    plot_cylinder(ax, cylinder4)
    plot_cylinder(ax, cylinder5)
    plot_cylinder(ax, cylinder6)
    plot_cylinder(ax, cylinder7)
    
    # plot_cylinder(ax, [0.00823718, 0.19858176, 0.31871104],  [-0.09051886,  0.06896938,  0.111], radius=0.05)
    plt.show()

if __name__ == '__main__':
    joint_angles = [53.856, -55.162, -30.889, -105.960, -90.846, -156.482]
    joint_angles = [-112.946, -55.333, -38.871, -30.372, -31.192, 19.329]
    joint_angles = [-94.773, -38.358, 27.718, -81.775, -90.109, 109.984]

    # joint_angles = [0, 0, 0, 0, 0, 0]
    # joint_angles[1] -= 180.0
    # joint_angles[1] = 0
    # joint_angles = [53.856, -5, -30.889, -105.960, -90.846, -156.482]
    joint_angles = [angle * np.pi / 180 for angle in joint_angles]
    box1 = Box([0.20, -0.32, 0.165], [0.15, 0.075, 0.355])
    boxes = [box1]
    plot_cylinders_and_boxes(joint_angles, boxes)
