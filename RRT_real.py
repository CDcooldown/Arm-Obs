from time import sleep
import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
from arm_real import *
from collision import *
import matplotlib.pyplot as plt
from control_utils.robot_control import *

class Node:
    def __init__(self, state, parent=None):
        self.state = np.array(state)
        self.parent = parent

class RRT:
    def __init__(self, start, goal, sampling_bounds, max_iter=1000, step_size=0.1, goal_sample_rate=0.25):
        self.start = Node(start)
        self.goal = Node(goal)
        self.sampling_bounds = [(-np.pi, np.pi)] * 6
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.dimension = len(start)
        self.nodes = [self.start]
        self.collision_states = []  # 新增：记录有碰点

    def plan(self):
        for _ in range(self.max_iter):
            rand_state = self.sample()
            nearest_node = self.nearest(rand_state)
            new_state = self.steer(nearest_node.state, rand_state)

            if not self.is_in_collision(nearest_node.state, new_state):
                new_node = Node(new_state, parent=nearest_node)
                self.nodes.append(new_node)

                if self.reached_goal(new_node):
                    return self.get_path(new_node)
            else:
                self.collision_states.append(new_state)  # 新增：记录被拒绝的点

        closest = self.nearest(self.goal.state)
        print("Fail to find goal!!")
        return self.get_path(closest)

    def sample(self):
        if np.random.rand() < self.goal_sample_rate:
            return self.goal.state
        else:
            return np.array([
                np.random.uniform(low, high) for (low, high) in self.sampling_bounds
            ])

    def nearest(self, state):
        return min(self.nodes, key=lambda node: np.linalg.norm(node.state - state))

    def steer(self, from_state, to_state):
        direction = to_state - from_state
        length = np.linalg.norm(direction)
        if length == 0:
            return from_state
        direction = direction / length
        return from_state + self.step_size * direction

    def reached_goal(self, node, threshold=0.1):
        error = node.state - self.goal.state
        error[5] = 0
        return np.linalg.norm(error) < threshold

    def is_in_collision(self, from_state, to_state):
        cylinders = get_cylinders(to_state)
        box1 = Box([0.60, -0.08, 0.20], [0.205, 0.08, 0.41])
        box1 = Box([0.60, -0.08, 0.10], [0.205, 0.08, 0.21])

        box2 = Box([-0.04, -0.10, 0.170], [0.060, 0.100, 0.340])
        boxes = [box1]
        for cylinder in cylinders:
            for box in boxes:
                if check_collision(cylinder, box):
                    return True
        return False

    def get_path(self, node):
        path = []
        while node is not None:
            path.append(node.state)
            node = node.parent
        return path[::-1]


class RRTVisualizer:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(15, 5))
        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)
        self.setup_axes()

    def setup_axes(self):
        titles = ['Joints 1 & 2', 'Joints 3 & 4', 'Joints 5 & 6']
        for ax, title in zip([self.ax1, self.ax2, self.ax3], titles):
            ax.set_title(title)
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(-np.pi, np.pi)
            ax.grid(True)

    def update(self, rrt):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.setup_axes()

        for node in rrt.nodes:
            if node.parent is not None:
                self.ax1.plot([node.parent.state[0], node.state[0]], 
                              [node.parent.state[1], node.state[1]], 'b-', alpha=0.3)
                self.ax2.plot([node.parent.state[2], node.state[2]], 
                              [node.parent.state[3], node.state[3]], 'b-', alpha=0.3)
                self.ax3.plot([node.parent.state[4], node.state[4]], 
                              [node.parent.state[5], node.state[5]], 'b-', alpha=0.3)

        states = np.array([node.state for node in rrt.nodes])
        self.ax1.scatter(states[:,0], states[:,1], c='b', s=5, alpha=0.5)
        self.ax2.scatter(states[:,2], states[:,3], c='b', s=5, alpha=0.5)
        self.ax3.scatter(states[:,4], states[:,5], c='b', s=5, alpha=0.5)

        # 绘制有碰点（橙色）
        if rrt.collision_states:
            collision_states = np.array(rrt.collision_states)
            self.ax1.scatter(collision_states[:,0], collision_states[:,1], c='orange', s=10, alpha=0.6)
            self.ax2.scatter(collision_states[:,2], collision_states[:,3], c='orange', s=10, alpha=0.6)
            self.ax3.scatter(collision_states[:,4], collision_states[:,5], c='orange', s=10, alpha=0.6)

        for ax, idx in zip([self.ax1, self.ax2, self.ax3], [(0,1), (2,3), (4,5)]):
            ax.scatter(rrt.start.state[idx[0]], rrt.start.state[idx[1]], 
                       c='g', s=50, marker='o', label='Start')
            ax.scatter(rrt.goal.state[idx[0]], rrt.goal.state[idx[1]], 
                       c='r', s=50, marker='*', label='Goal')
            if rrt.collision_states:
                ax.scatter([], [], c='orange', s=10, label='Collision')
            ax.legend()

        plt.draw()
        plt.pause(0.01)


def visualize_rrt(rrt, interval=50):
    visualizer = RRTVisualizer()
    original_plan = rrt.plan

    def visualized_plan():
        for i in range(rrt.max_iter):
            rand_state = rrt.sample()
            nearest_node = rrt.nearest(rand_state)
            new_state = rrt.steer(nearest_node.state, rand_state)

            if not rrt.is_in_collision(nearest_node.state, new_state):
                new_node = Node(new_state, parent=nearest_node)
                rrt.nodes.append(new_node)

                if i % interval == 0:
                    visualizer.update(rrt)

                if rrt.reached_goal(new_node):
                    visualizer.update(rrt)
                    return rrt.get_path(new_node)
            else:
                rrt.collision_states.append(new_state)  # 新增：记录碰撞点

        visualizer.update(rrt)
        closest = rrt.nearest(rrt.goal.state)
        print("Failed to find exact goal path")
        return rrt.get_path(closest)

    rrt.plan = visualized_plan
    path = rrt.plan()
    plt.ioff()
    plt.show()
    return path


if __name__ == '__main__':
    # 你的原始设置
        # joint_angles = [[-90, -90, 0, -86.482, 0, 200.201],
        #             [-98.115, -30.464, -25.248, -38.886, -90.727, 108.630],
        #             [0.000, -24.550, 1.211, -68.514, -92.894, 200.190]]
    start = [-90, -90, 0, -86.482, 0, 19.306]
    start = [-98.115, -30.464, -25.248, -38.886, -90.727, 108.630]
    goal = [0.000, -24.550, 1.211, -68.514, -92.894, 108.190]
    # goal = [-33.355, -35.976, -55.926, -91.715, 2.628, 19.330]
    start = [angle * np.pi / 180 for angle in start]
    goal = [angle * np.pi / 180 for angle in goal]

    bounds = [(-np.pi, np.pi)] * 6
    

    rrt = RRT(start=start, goal=goal, sampling_bounds=bounds, max_iter=5000, step_size=0.1)
    path = visualize_rrt(rrt)

    robot_arm = robot()
    print("Found path with", len(path), "steps:")
    for i, state in enumerate(path):
        print(f"Step {i}: {state}")
        angles = [angle * 180 / np.pi for angle in state]
        robot_arm.set_angles(angles)
        # sleep(1)
