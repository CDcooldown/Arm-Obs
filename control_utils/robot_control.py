import numpy as np
import math
import time

from control_utils.elephant_command import *

INIT_STATE = {'X0':-300, 'Y0':-37, 'Z0':470, 'RX0':-179.5, 'RY0':0.1, 'RZ0':-111, 'vel':5000}
CAMERA_OFFSET = {'X':0, 'Y':10, 'Z':100}


class robot():
    def __init__(self):
        # 初始化
        self.robot_cmd = elephant_command()

    def init_move(self):
        # 设定初始位姿
        self.robot_cmd.set_coords([INIT_STATE['X0'], INIT_STATE['Y0'], INIT_STATE['Z0'], 
                                    INIT_STATE['RX0'], INIT_STATE['RY0'], INIT_STATE['RZ0']], INIT_STATE['vel'])
        time.sleep(0.1)
        while self.robot_cmd.check_running():
            time.sleep(0.1)
        self.robot_cmd.set_digital_out(0, 0)
        time.sleep(0.1)
    
    def move_to_target(self, x_c, y_c, z_c, angle):
        # 移动到目标位置
        coords = self.robot_cmd.get_coords()
        coords[0] += 1000 * x_c + CAMERA_OFFSET['X']
        coords[1] += -1000 * y_c + CAMERA_OFFSET['Y']
        coords[2] += -1000 * z_c + CAMERA_OFFSET['Z']
        coords[5] += angle
        self.robot_cmd.set_coords(coords, INIT_STATE['vel'])
        time.sleep(0.1)
        while self.robot_cmd.check_running():
            time.sleep(0.1)
        time.sleep(0.1)

    def gripper_close(self):
        # 夹爪关闭 
        time.sleep(0.1)
        self.robot_cmd.set_digital_out(0, 1)
        self.robot_cmd.set_digital_out(0, 1)
        self.robot_cmd.set_digital_out(0, 1)
        time.sleep(0.1)

    def gripper_open(self):
        # 夹爪打开
        time.sleep(0.1)
        self.robot_cmd.set_digital_out(0, 0)
        self.robot_cmd.set_digital_out(0, 0)
        self.robot_cmd.set_digital_out(0, 0)
        time.sleep(0.1)

    def set_angles(self, angles):
        self.robot_cmd.set_angles(angles, 1000)
        
    def get_coords(self):
        coords = self.robot_cmd.get_coords()
        print(coords)
        return coords
    
    def get_angles(self):
        angles = self.robot_cmd.get_angles()
        print(angles)
        return angles

    # def reached_goal(self, node, threshold=0.001):
    #     diff = np.abs(node.state - self.goal.state)
    #     print("Current state:", node.state)
    #     print("Goal state:", self.goal.state)
    #     print("Difference:", diff)
    #     return np.all(diff < threshold)


def reach_joints(joints, target):
    array_joints = np.array(joints)
    array_target = np.array(target)
    diff = np.abs(array_joints - array_target)
    print("diff is", diff)
    return np.all(diff < 1)

def main():
    # 初始化机器人
    robot_arm = robot()
    
    # 初始化机械臂到初始位置
    # robot_arm.init_move()
    
    # # 移动到目标位置 (-200, -200, 400)
    # target_x = -200
    # target_y = -200
    # target_z = 300
    # angle = 0  # 如果需要旋转角度，可以在这里设置
    # robot_arm.move_to_target(target_x, target_y, target_z, angle)

    # robot_arm.get_angles()
    # robot_arm.get_coords()

    # joint_angles = [[53.856, -55.162, -30.889, -105.960, -90.846, -156.482], 
    #                 [43.856, -55.162, -30.889, -95.960, -90.846, -156.482]]
    joint_angles = [[-90, -90, 0, -86.482, 0, 200.201],
                    [-98.115, -30.464, -25.248, -38.886, -90.727, 108.630],
                    [0.000, -24.550, 1.211, -68.514, -92.894, 200.190]]
    robot_arm.gripper_open()
    robot_arm.set_angles(joint_angles[0])
    time.sleep(3)
    robot_arm.set_angles(joint_angles[1])
    # print(angles)
    while(1):
        angles = robot_arm.get_angles()

        if reach_joints(angles, joint_angles[1]):
            robot_arm.gripper_close()
            break

    time.sleep(5)
    robot_arm.set_angles(joint_angles[2])
    while(1):
        angles = robot_arm.get_angles()

        if reach_joints(angles, joint_angles[2]):
            robot_arm.gripper_open()
            break





if __name__ == "__main__":
    main()