import numpy as np
from spatialmath import SE3

# 两组欧拉角（单位为度）
eul1 = [90, 89, -139]
eul2 = [0, 89, -49]

# 转换为旋转矩阵（ZYX 顺序）
T1 = SE3.RPY(eul1, order='xyz', unit='deg')
T2 = SE3.RPY(eul2, order='xyz', unit='deg')

print(T1.R)

print(T2.R)

# 比较两组是否代表相同的旋转（位姿）
print(np.allclose(T1.R, T2.R, atol=1e-2))  # True 表示它们是相同的姿态
