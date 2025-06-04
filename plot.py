import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_cylinder(ax, cylinder, resolution=30):
    start = cylinder.p1
    end = cylinder.p2
    radius = cylinder.r
    v = np.array(end) - np.array(start)
    height = np.linalg.norm(v)
    v = v / height
    # 创建圆柱底面圆环
    theta = np.linspace(0, 2*np.pi, resolution)
    circle = np.array([radius*np.cos(theta), radius*np.sin(theta), np.zeros_like(theta)])
    
    # 构建旋转矩阵，将 z 轴旋转到 v
    z = np.array([0,0,1])
    axis = np.cross(z, v)
    angle = np.arccos(np.dot(z, v))
    if np.linalg.norm(axis) != 0:
        axis = axis / np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)
    else:
        R = np.eye(3)
    
    circle_rot = R @ circle
    bottom = circle_rot + np.array(start).reshape(3,1)
    top = circle_rot + np.array(end).reshape(3,1)
    
    # 构建面
    verts = []
    for i in range(resolution):
        j = (i + 1) % resolution
        verts.append([bottom[:,i], bottom[:,j], top[:,j], top[:,i]])
    
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.6, facecolor='blue'))

def plot_box(ax, box):
    center = box.center
    size = box.size
    # 计算八个顶点
    dx, dy, dz = size / 2
    corners = np.array([
        [-dx, -dy, -dz],
        [ dx, -dy, -dz],
        [ dx,  dy, -dz],
        [-dx,  dy, -dz],
        [-dx, -dy,  dz],
        [ dx, -dy,  dz],
        [ dx,  dy,  dz],
        [-dx,  dy,  dz]
    ])
    corners += center  # 平移到中心

    # 定义每个面由四个顶点组成
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],  # bottom
        [corners[4], corners[5], corners[6], corners[7]],  # top
        [corners[0], corners[1], corners[5], corners[4]],  # front
        [corners[2], corners[3], corners[7], corners[6]],  # back
        [corners[1], corners[2], corners[6], corners[5]],  # right
        [corners[0], corners[3], corners[7], corners[4]]   # left
    ]

    ax.add_collection3d(Poly3DCollection(faces, alpha=0.6, facecolor='orange'))

# if __name__ == '__main__':

#     # 使用
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     plot_cylinder(ax, [0.00823718, 0.19858176, 0.31871104],  [-0.09051886,  0.06896938,  0.111], radiu)
#     plot_cylinder(ax, [0, 0, 0],  [0, 0, 0.2], radius=0.05)
#     plt.show()
#     # pos2 [-0.07181698 -0.17882714  0.3382538 ]
#     # pos3 [-0.20198518 -0.26226086  0.52181592]

#     # pos2 [-0.07181698, -0.17882714,  0.3382538 ]
#     # pos3 [ 0.05835122, -0.09539342,  0.15469167]

#     # pos2 [0.14079814 0.33484936 0.40349543]
#     # pos3 [0.02807838, 0.54644811, 0.41446006]

#     # pos2 [0.00823718, 0.19858176, 0.31871104]
#     # pos3 [-0.09051886,  0.06896938,  0.111]