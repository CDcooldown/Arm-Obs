import sim
import time
from ur3_api import UR3
import numpy as np

ur3_sim = UR3()

ur3_sim.run_coppelia()

# joint_values = [0, 0, 0, 0, 0, 0] # random position
joint_values = [0.57, 0.0, 0, 0, 0, 0]

# joint_values = [1.57, 1.0, 0.8, -0.2, -1.57, 0]
# q = [-78.65+180, -59.74+90, 27, -54.43+90, -90, 121.99]

# joint_values = [angle * np.pi / 180 for angle in q]


# joint_values = [0, -np.pi/4, np.pi/2, 0.3, np.pi/3, 0.3]
# joint_values = [0, -np.pi/4, np.pi/2, 0, np.pi/4, 0]
# joint_values = [0, -np.pi/4, np.pi/2, 0, np.pi/3, 0]
initial_state = [0, 0, 0, 0, 0, 0]

ur3_sim.joint_values(joint_values)

time.sleep(5)

# ur3_sim.joint_values(initial_state)

ur3_sim.stop_simulation()

# 下面是一些测试数据：
# 第一次：
# 关节角：
# joint_values = [0, -np.pi/4, np.pi/2, 0, np.pi/4, 0]
# 正运动学模块输出：
# Position: x=-0.3037, y=-0.1703, z=0.0721
# Orientation (ZYX Euler): alpha=-54.7356, beta=-30.0000, gamma=125.2644
# 仿真显示：
# x = -0.00618 y = 0.15893 z = 0.56828
# alpha = 125.231 beta = 30.011 gamma = -54.726

# 第二次：
# 关节角：
# joint_values = [0, -np.pi/4, np.pi/2, 0, np.pi/3, 0]
# 正运动学模块输出：
# Position: x=-0.3129, y=-0.1533, z=0.0629
# Orientation (ZYX Euler): alpha=-67.7923, beta=-20.7048, gamma=130.8934
# 仿真显示：
# x = -0.00121 y = 0.14534 z = 0.57568
# alpha = 140.736 beta = 37.774 gamma = -63.427