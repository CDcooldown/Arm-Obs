import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def smooth_cubic_interpolation(data, skip=3, num_points=None):
    data = np.array(data)
    num_joints = data.shape[1]
    
    # 下采样节点
    control_indices = np.arange(0, len(data), skip)
    if control_indices[-1] != len(data) - 1:
        control_indices = np.append(control_indices, len(data) - 1)  # 保证最后一点包含
    
    control_times = control_indices
    full_times = np.arange(len(data)) if num_points is None else np.linspace(0, len(data) - 1, num_points)

    smooth_data = []
    for j in range(num_joints):
        y = data[control_indices, j]
        cs = CubicSpline(control_times, y, bc_type='natural')
        smooth_joint = cs(full_times)
        smooth_data.append(smooth_joint)

    return np.stack(smooth_data, axis=1)

# 示例
data = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.02514512, 0.00919982, -0.0702962, -0.02074926, 0.06038828, -0.0162559], 
        [0.0129469, 0.00429615, -0.13671287, -0.01749527, 0.05126699, 0.05669749], 
        [0.00309751, 0.06322597, -0.14874462, -0.03767709, -0.01438213, 0.0963004], 
        [0.04726844, 0.06943923, -0.17785491, -0.07357771, 0.02826741, 0.15997988], 
        [0.09205031, 0.06055662, -0.2373945, -0.04310323, -0.02989645, 0.16766052], 
        [0.15905061, 0.01944575, -0.28323652, -0.05285904, -0.00207352, 0.19681656]]

smooth_data = smooth_cubic_interpolation(data, skip=2)

# 可视化第3个关节的原始和拟合轨迹
plt.plot(np.arange(len(data)), np.array(data)[:, 2], 'o-', label='original')
plt.plot(np.arange(len(smooth_data)), smooth_data[:, 2], 'x--', label='smoothed')
plt.legend()
plt.title("Joint 3 trajectory")
plt.show()
