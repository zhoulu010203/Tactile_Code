'''
6×8阵列式传感器触觉重构算法
高斯混合模型（点接触）
EM算法求解
区域连通检测K
点面（线）识别：单个连通区域响应点数是否大于4
面接触下使用双线性插值呈现
点云生成策略：高斯/对数高斯
包含触觉传感器数据和测力计真值数据存储
'''
from sklearn.mixture import GaussianMixture
from Find_Union import UnionFind
from GMM_Generate import GaussianMixture2D
from PDF_GRID import pdf_grid
from connected_component_labeling import connected_component_labeling, visualize_labeled_regions
from point_cloud_generation import generate_point_cloud, visualize_point_cloud
from PatternRecognition import count_patterns
from Face_contact_test import plot_tactile_simple
import numpy as np
import matplotlib.pyplot as plt
import serial
from read_force import ForceSensorReader
import csv
import datetime
import os

mapping = np.array([
    [0, 2, 4, 6, 1, 3, 5, 7],
    [14, 12, 10, 8, 9, 11, 13, 15],
    [22, 20, 18, 16, 30, 28, 26, 24],
    [23, 21, 19, 17, 31, 29, 27, 25],
    [38, 36, 34, 32, 46, 44, 42, 40],
    [39, 37, 35, 33, 47, 45, 43, 41]
])

# 定义多项式系数
p1 = 1.763
p2 = -0.1261

# 点云对数正态分布参数
mu_pointcloud = [0.0, 0.0]
sigma_pointcloud = [[0.5, 0.0], [0.0, 0.5]]
min_points = 20
scale_factor = 100
spacing = 10

# 设置交互模式
plt.ion()

# 创建主图形和坐标轴（3D点接触可视化）
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')

# 创建面接触可视化窗口
# fig_face = plt.figure(figsize=(6, 6))
ax_face = fig.add_subplot(122)

# 初始化显示一个平面
x = np.linspace(0, 100, 20)
y = np.linspace(0, 70, 14)
X, Y = np.meshgrid(x, y)
Z_plane = np.zeros_like(X)  # 创建平面

ax.plot_surface(X, Y, Z_plane, cmap='viridis', alpha=0.7)
ax.set_title("Ready")
ax.set_zlim(0, 0.005)
ax_face.set_title("Ready")

# 3D高斯曲面生成网格
grid = np.column_stack([X.ravel(), Y.ravel()])

# --- 1. 设置保存文件的名称和表头 ---
# 使用时间戳作为文件名，防止覆盖
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"tactile_data_{timestamp_str}.csv"

# 创建文件并写入表头
# newline='' 是为了防止在Windows下出现空行
f_csv = open(csv_filename, 'w', newline='')
writer = csv.writer(f_csv)
# 表头：时间, 传感器总力, 分配后的力(数组形式)
writer.writerow(["Timestamp", "Force_Sensor", "F_distributed", "Component_Means"])
print(f"数据将保存至: {csv_filename}")

try:
    ser1 = serial.Serial('COM3', 115200)
    ser2 = serial.Serial('COM10', 115200)
    ser3 = serial.Serial('COM9', 115200)
    sensor = ForceSensorReader('COM6', 2400)

    while True:
        # --- 2. 每次循环开始初始化保存变量 ---
        # 如果这一轮循环没有计算出F_distributed（例如无接触），我们存为 "N/A" 或 空列表
        save_f_distributed = 0
        save_component_means = 0

        # 读取三个串口的数据
        data1 = ser1.readline().decode().strip()
        data2 = ser2.readline().decode().strip()
        data3 = ser3.readline().decode().strip()
        force = sensor.read_latest()

        # 转换为浮点数列表
        values1 = [float(x) for x in data1.split(',')]
        values2 = [float(x) for x in data2.split(',')]
        values3 = [float(x) for x in data3.split(',')]

        # 直接将三个列表组合成6×8的图像
        tactile_image = np.array(values1 + values2 + values3).reshape(6, 8)
        flat_image = tactile_image.flatten()
        flat_image_transformed = (p1 * flat_image + p2)
        tactile_image = flat_image_transformed[mapping].reshape(6, 8).round(1)
        binary_image = (tactile_image >= 0.18).astype(int)  # 生成对应二值图像
        count_ones = np.sum(binary_image)
        F_sum = np.sum(tactile_image[tactile_image > 0.15])



        # 检查二值图像是否全为0
        if np.all(binary_image == 0):

            ax.clear()
            ax_face.clear()
            # 绘制平面
            ax.plot_surface(X, Y, Z_plane, cmap='viridis', alpha=0.7)
            ax.set_zlim(0, 0.005)
            ax.set_title("No Contact Detected")
            ax_face.set_title("No Face Contact")

            plt.figure(fig.number)
            plt.draw()
            plt.pause(0.01)
        else:
            ax.clear()
            ax_face.clear()
            # 存在点接触，进行高斯曲面拟合
            # 使用连通区域标记函数
            labels, num_components = connected_component_labeling(binary_image)
            count_pattern = count_patterns(binary_image)
            if count_pattern:
                num_components = num_components + count_pattern

            if count_ones <= 4 * num_components:  # 如果是点接触进行高斯混合模型拟合
                # 生成点云
                PointsCloud, point_counts = generate_point_cloud(
                    tactile_image=tactile_image,
                    binary_image=binary_image,  # 只在二值图像为1的区域生成点云
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
                    component_weights = gmm.weights_
                    component_means = gmm.means_
                    F_distributed = 0.575 * F_sum * gmm.weights_

                    # --- 3. 获取要保存的数据 ---
                    # 将numpy数组转换为列表，以便在csv中显示为 [0.1, 0.2] 的形式
                    save_f_distributed = F_distributed.tolist()
                    save_component_means = component_means.tolist()

                    # --- 打印与校验 ---
                    # print(f"总力 (F_sum): {F_sum}")
                    # print("各组件权重:", component_weights)
                    print("分配后的力 (F_distributed):", F_distributed)
                    print("估计位置 (component_means):", component_means)


                    # 计算 GMM 概率密度
                    log_prob = gmm.score_samples(grid)
                    Z = np.exp(log_prob).reshape(X.shape)

                    # 绘制新的3D曲面
                    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
                    ax.set_title(f"Contact Detected (Components: {num_components})")
                    ax.set_zlim(0, 0.005)  # 保持固定的Z轴范围

                    # # 原始点云也画出来（作为参考）
                    # ax.scatter(PointsCloud[:, 0], PointsCloud[:, 1], np.zeros_like(PointsCloud[:, 0]),
                    #            c='red', s=10, alpha=0.5)

                    plt.figure(fig.number)
                    plt.draw()
                    plt.pause(0.001)

            else:  # 如果是面（线）接触
                # 使用imshow绘制tactile_image，启用插值
                face_imshow = ax_face.imshow(tactile_image, cmap='hot', interpolation='bilinear',
                                             extent=[1, 9, 1, 7], origin='lower')

                ax_face.set_title("Face Contact Visualization")
                ax_face.set_xlabel("X Position")
                ax_face.set_ylabel("Y Position")
                ax.plot_surface(X, Y, Z_plane, cmap='viridis', alpha=0.7)
                ax.set_zlim(0, 1)
                # # 添加网格线以便更好地观察
                # ax_face.grid(True, alpha=0.3)
                plt.figure(fig.number)
                plt.draw()
                plt.pause(0.001)

        # --- 4. 写入 CSV 文件 ---
        current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]  # 精确到毫秒
        writer.writerow([current_time, force, save_f_distributed, save_component_means])
        # 刷新缓冲区，确保即使强制中断，最近的数据也已写入硬盘
        f_csv.flush()

except KeyboardInterrupt:
    print("程序被用户中断")
except Exception as e:
    print(f"发生错误: {e}")
finally:
    # --- 5. 关闭文件 ---
    if 'f_csv' in locals() and not f_csv.closed:
        f_csv.close()
        print("CSV文件已保存并关闭。")
    # 关闭交互模式
    plt.ioff()
    plt.show()
