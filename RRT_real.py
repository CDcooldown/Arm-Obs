from time import sleep
import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
from arm_real import *
from collision import *
import matplotlib.pyplot as plt
# from control_utils import *
from control_utils.robot_control import *

class Node:
    def __init__(self, state, parent=None):
        self.state = np.array(state)
        self.parent = parent

class RRT:
    def __init__(self, start, goal, sampling_bounds, max_iter=1000, step_size=0.1, goal_sample_rate=0.20):
        self.start = Node(start)
        self.goal = Node(goal)
        self.sampling_bounds = [(-np.pi, np.pi)] * 6
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.dimension = len(start)
        self.nodes = [self.start]

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

        # Return path to the node closest to goal if direct goal not reached
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
        print(node.state)
        print(self.goal.state)
        return np.linalg.norm(node.state - self.goal.state) < threshold

    def is_in_collision(self, from_state, to_state):
        # cylinder1 = get_cylinder1(to_state)
        cylinders = get_cylinders(to_state)
        # box1 = Box([0.596, -0.109, 0.176], [0.094, 0.218, 0.408])
        # box1 = Box([0.596, -0.109, 0.176], [0.094, 0.218, 0.408])
        box2 = Box([-0.04, -0.10, 0.173], [0.060, 0.100, 0.345])
        # box2 = Box([-0.014, -0.179, 0.146], [0.170, 0.099, 0.352])
        boxes = [box2]
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
        plt.ion()  # 交互模式
        self.fig = plt.figure(figsize=(15, 5))
        
        # 创建三个投影视图
        self.ax1 = self.fig.add_subplot(131)  # 关节1-2
        self.ax2 = self.fig.add_subplot(132)  # 关节3-4
        self.ax3 = self.fig.add_subplot(133)  # 关节5-6
        
        self.setup_axes()
    
    def setup_axes(self):
        titles = ['Joints 1 & 2', 'Joints 3 & 4', 'Joints 5 & 6']
        for ax, title in zip([self.ax1, self.ax2, self.ax3], titles):
            ax.set_title(title)
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(-np.pi, np.pi)
            ax.grid(True)
    
    def update(self, rrt):
        """更新可视化"""
        # 清除之前的内容
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.setup_axes()
        
        # 绘制所有节点和连接
        for node in rrt.nodes:
            if node.parent is not None:
                # 在三个平面上绘制连接线
                self.ax1.plot([node.parent.state[0], node.state[0]], 
                             [node.parent.state[1], node.state[1]], 'b-', alpha=0.3)
                self.ax2.plot([node.parent.state[2], node.state[2]], 
                             [node.parent.state[3], node.state[3]], 'b-', alpha=0.3)
                self.ax3.plot([node.parent.state[4], node.state[4]], 
                             [node.parent.state[5], node.state[5]], 'b-', alpha=0.3)
        
        # 绘制所有节点
        states = np.array([node.state for node in rrt.nodes])
        self.ax1.scatter(states[:,0], states[:,1], c='b', s=5, alpha=0.5)
        self.ax2.scatter(states[:,2], states[:,3], c='b', s=5, alpha=0.5)
        self.ax3.scatter(states[:,4], states[:,5], c='b', s=5, alpha=0.5)
        
        # 标记起点和终点
        for ax, idx in zip([self.ax1, self.ax2, self.ax3], [(0,1), (2,3), (4,5)]):
            ax.scatter(rrt.start.state[idx[0]], rrt.start.state[idx[1]], 
                      c='g', s=50, marker='o', label='Start')
            ax.scatter(rrt.goal.state[idx[0]], rrt.goal.state[idx[1]], 
                      c='r', s=50, marker='*', label='Goal')
            ax.legend()
        
        plt.draw()
        plt.pause(0.01)

def visualize_rrt(rrt, interval=50):
    """包装RRT规划过程并可视化"""
    visualizer = RRTVisualizer()
    
    # 保存原始方法
    original_plan = rrt.plan
    
    def visualized_plan():
        for i in range(rrt.max_iter):
            rand_state = rrt.sample()
            nearest_node = rrt.nearest(rand_state)
            new_state = rrt.steer(nearest_node.state, rand_state)
            
            if not rrt.is_in_collision(nearest_node.state, new_state):
                new_node = Node(new_state, parent=nearest_node)
                rrt.nodes.append(new_node)
                
                # 定期更新可视化
                if i % interval == 0:
                    visualizer.update(rrt)
                
                if rrt.reached_goal(new_node):
                    visualizer.update(rrt)  # 最终更新
                    return rrt.get_path(new_node)
        
        visualizer.update(rrt)  # 最终更新
        closest = rrt.nearest(rrt.goal.state)
        print("Failed to find exact goal path")
        return rrt.get_path(closest)
    
    # 替换plan方法
    rrt.plan = visualized_plan
    
    # 执行规划
    path = rrt.plan()
    
    # 保持窗口打开
    plt.ioff()
    plt.show()
    
    return path

# 使用示例
if __name__ == '__main__':
    # 你的原始设置
        # joint_angles = [[-90, -90, 0, -86.482, 0, 200.201],
        #             [-98.115, -30.464, -25.248, -38.886, -90.727, 108.630],
        #             [0.000, -24.550, 1.211, -68.514, -92.894, 200.190]]
    start = [-90, -90, 0, -86.482, 0, 19.306]
    goal = [-98.115, -30.464, -25.248, -38.886, -90.727, 108.630]
    goal = [-33.355, -35.976, -55.926, -91.715, 2.628, 19.330]
    start = [angle * np.pi / 180 for angle in start]
    goal = [angle * np.pi / 180 for angle in goal]

    bounds = [(-np.pi, np.pi)] * 6
    
    # 创建RRT实例
    rrt = RRT(start=start, goal=goal, sampling_bounds=bounds, max_iter=1000, step_size=0.1)
    
    # 运行带可视化的规划
    path = visualize_rrt(rrt)

    robot_arm = robot()

    
    print("Found path with", len(path), "steps:")
    for i, state in enumerate(path):
        print(f"Step {i}: {state}")
        angles = path[i]
        angles = [angle * 180 / np.pi for angle in angles]

        robot_arm.set_angles(angles)
        sleep(1)


# def main():
#     # 定义起始和目标关节状态（单位：角度）
#     start = [0, 0, 0, 0, 0, 0]
#     goal = [1.57, 1.0, 0.8, -0.2, -1.57, 0]

#     # 转换为弧度（如果你希望在 RRT 中用弧度单位，可以直接用弧度）

#     # 定义采样边界（弧度）
#     bounds = [(-np.pi, np.pi)] * 6

#     # 初始化 RRT
#     rrt = RRT(start=start, goal=goal, sampling_bounds=bounds, max_iter=50000, step_size=0.02)

#     # 提供一个简单的无碰撞检测实现（默认全程无障碍）
#     def no_collision(from_state, to_state):
#         return False

#     # 替换碰撞检测函数
#     rrt.is_in_collision = no_collision

#     # 路径规划
#     path = rrt.plan()

#     # 打印路径结果
#     print("Path found:")
#     for i, state in enumerate(path):
#         print(f"Step {i}: {state}")  # 输出角度形式

# if __name__ == '__main__':
#     main()

# 目标位置 [45.300, -480.415, 358.080, -179.218, -4.580, -116.805] [-98.115, -30.464, -25.248, -38.886, -90.727, 108.630]
# 目标位置 []