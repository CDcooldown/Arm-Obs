from time import sleep
import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
import spatialmath as sm
import matplotlib.pyplot as plt

# 定义你刚才的 ur3_fk 函数（不必改动）

# 设定一个 joint angle 示例（单位为弧度）
joint_angles = [0, -np.pi/2, 0, -np.pi/2, 0, 0]

# # 创建机器人模型（和你函数里的保持一致）
L1 = RevoluteDH(d=0.1519,   a=0,        alpha=np.pi/2, offset=0, modified=True)
L2 = RevoluteDH(d=0,        a=-0.24365, alpha=0,       offset=0, modified=True)
L3 = RevoluteDH(d=0,        a=-0.21325, alpha=0,       offset=0, modified=True)
L4 = RevoluteDH(d=0.11235,  a=0,        alpha=np.pi/2, offset=0, modified=True)
L5 = RevoluteDH(d=0.08535,  a=0,        alpha=-np.pi/2,offset=0, modified=True)
L6 = RevoluteDH(d=0.0655,   a=0,        alpha=0,       offset=0, modified=True)

# L1 = RevoluteDH(d=0.1519,   a=0,        alpha=np.pi/2, offset=0, modified=True)
# L2 = RevoluteDH(d=0,        a=-0.24365, alpha=0,       offset=-np.pi/2, modified=True)
# L3 = RevoluteDH(d=0,        a=-0.21325, alpha=0,       offset=0, modified=True)
# L4 = RevoluteDH(d=0.11235,  a=0,        alpha=np.pi/2, offset=-np.pi/2, modified=True)
# L5 = RevoluteDH(d=0.08535,  a=0,        alpha=-np.pi/2,offset=0, modified=True)
# L6 = RevoluteDH(d=0.0817,   a=0,        alpha=0,       offset=0, modified=True)

# ur3 = DHRobot([L1, L2], name='UR3')
ur3 = DHRobot([L1, L2, L3, L4, L5, L6], name='UR3')

# 可视化
ur3.plot(joint_angles, backend='pyplot')  # 或 backend='swift' 使用WebGL可交互可视化

sleep()