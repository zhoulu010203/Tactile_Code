'''
6×8阵列式传感器触觉重构算法
高斯混合模型（点接触）
EM算法求解
区域连通检测K
点面（线）识别：单个连通区域响应点数是否大于4
面接触下使用双线性插值呈现
点云生成策略：高斯/对数高斯
'''

from sklearn.mixture import GaussianMixture
from Find_Union import UnionFind
from GMM_Generate import GaussianMixture2D
from PDF_GRID import pdf_grid
from connected_component_labeling import connected_component_labeling, visualize_labeled_regions
from point_cloud_generation import generate_point_cloud, visualize_point_cloud
from PatternRecognition import count_patterns
import numpy as np
import matplotlib.pyplot as plt
import serial
import time

# 点云对数正态分布参数
mu_pointcloud = [0.0, 0.0]
sigma_pointcloud = [[0.05, 0.0], [0.0, 0.05]]
min_points = 20
scale_factor = 1000
spacing = 1  # 网格间距（mm）

# 设置交互模式
plt.ion()

# 创建图形和坐标轴
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 初始化显示一个平面
x = np.linspace(0, 10, 30)
y = np.linspace(0, 10, 30)
X, Y = np.meshgrid(x, y)
Z_plane = np.zeros_like(X)  # 创建平面

# 3D高斯曲面
# 生成网格
grid = np.column_stack([X.ravel(), Y.ravel()])

# 设置坐标轴标签和范围
# ax.set_xlabel("X (mm)")
# ax.set_ylabel("Y (mm)")
# ax.set_zlabel("Probability Density")
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 10)

try:
    ser = serial.Serial('COM3', 115200)  # 修改为你的串口和波特率

    while True:
        # start_time = time.time()
        # # 仿真高斯参数
        # means_sim = [[1.5, 1.5], [4.5, 4.5]]
        # covariances_sim = [[[0.4, 0], [0, 0.4]], [[0.4, 0], [0, 0.4]]]
        # weights_sim = [40, 60]
        # GMM = GaussianMixture2D(means_sim, covariances_sim, weights_sim)  # 生成指定高斯混合函数
        # tactile_image = pdf_grid(GMM, 1, 9, 1, 9, 1)  # 根据GMM生成对应触觉像素点的概率密度值
        # # print(tactile_image)
        # binary_image = (tactile_image * 100 >= 2).astype(int)  # 生成对应二值图像
        # # print(binary_image)

        data = ser.readline().decode().strip()  # 读取一行数据
        values = [float(x) for x in data.split(',')]  # 直接转换为浮点数列表
        tactile_image = np.array(values).reshape(4, 4)
        binary_image = (tactile_image >= 0.23).astype(int)  # 生成对应二值图像

        # 清除之前的图形
        ax.clear()

        # 检查二值图像是否全为0
        if np.all(binary_image == 0):
            # 绘制平面
            ax.plot_surface(X, Y, Z_plane, cmap='viridis', alpha=0.7)
            ax.set_title("No Contact Detected")
        else:
            # 存在点接触，进行高斯曲面拟合
            # 使用连通区域标记函数
            labels, num_components = connected_component_labeling(binary_image)
            count_pattern = count_patterns(binary_image)
            if count_pattern:
                num_components = num_components + count_pattern
            # print(num_components)

            # 生成点云
            PointsCloud, point_counts = generate_point_cloud(
                tactile_image=tactile_image,
                binary_image=binary_image,  # 可选：只在二值图像为1的区域生成点云
                spacing=spacing,
                mu_pointcloud=mu_pointcloud,
                sigma_pointcloud=sigma_pointcloud,
                min_points=min_points,
                scale_factor=scale_factor
            )

            # 如果点云数据为空，绘制平面
            if len(PointsCloud) == 0:
                ax.plot_surface(X, Y, Z_plane, cmap='Blues', alpha=0.7)
                ax.set_title("No Contact Points Generated")
            else:
                # 拟合 GMM
                gmm = GaussianMixture(n_components=num_components, covariance_type='spherical', random_state=0)
                gmm.fit(PointsCloud)

                # 计算 GMM 概率密度
                log_prob = gmm.score_samples(grid)
                Z = np.exp(log_prob).reshape(X.shape)

                # 绘制新的3D曲面
                ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

                # # 原始点云也画出来（作为参考）
                # ax.scatter(PointsCloud[:, 0], PointsCloud[:, 1], np.zeros_like(PointsCloud[:, 0]),
                #            c='red', s=10, alpha=0.5)

                ax.set_title(f"Contact Detected (Components: {num_components})")

        ax.set_zlim(0, 1)  # 保持固定的Z轴范围

        # 更新图形
        plt.draw()
        plt.pause(0.01)  # 短暂暂停以更新图形

        # over_time = time.time()
        # during_time = over_time - start_time
        # print(f'运行时间：{during_time}秒')

except KeyboardInterrupt:
    print("程序被用户中断")
except Exception as e:
    print(f"发生错误: {e}")
# finally:
#     # 关闭串口和图形
#     if 'ser' in locals() and ser.is_open:
#         ser.close()
#     plt.ioff()
#     plt.show()