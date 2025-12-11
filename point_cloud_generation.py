import numpy as np


def generate_point_cloud(tactile_image, binary_image=None, spacing=1,
                         mu_pointcloud=(0.0, 0.0),
                         sigma_pointcloud=((0.01, 0.0), (0.0, 0.01)),
                         min_points=20, scale_factor=1000):
    """
    根据触觉图像生成点云数据

    参数:
        tactile_image: 触觉图像（概率密度图）
        binary_image: 二值图像（可选，用于限制生成区域）
        spacing: 网格间距（mm）
        mu_pointcloud: 正态分布的均值
        sigma_pointcloud: 正态分布的协方差矩阵
        min_points: 每个像素生成点的最小数量阈值
        scale_factor: 概率密度到点数的缩放因子

    返回:
        PointsCloud: 生成的点云数据，形状为 (N, 2)
        point_counts: 每个像素生成的点数矩阵
    """
    rows, cols = tactile_image.shape
    PointsCloud = []
    point_counts = np.zeros_like(tactile_image, dtype=int)

    # 将参数转换为numpy数组
    mu = np.array(mu_pointcloud)
    sigma = np.array(sigma_pointcloud)

    # 计算理论众数用于去中心化（对数高斯分布需要）
    sigma_diag = np.diag(sigma)
    theoretical_mode = np.exp(mu - sigma_diag)

    for i in range(rows):
        for j in range(cols):
            # 如果提供了二值图像，只处理值为1的像素
            if binary_image is not None and binary_image[i, j] == 0:
                continue

            # 计算该像素应生成的点数
            n_points = int(tactile_image[i, j] * scale_factor)
            point_counts[i, j] = n_points

            # 只有当点数超过阈值时才生成点
            if n_points >= min_points:
                center_x = (j + 1) * spacing
                center_y = (rows - i) * spacing

                # 生成高斯分布样本
                samples = np.random.multivariate_normal(mu, sigma, size=n_points)
                # # 随机符号（四象限分布）
                # signs = np.random.choice([-1, 1], size=samples.shape)
                # samples = samples * signs

                # 生成对数高斯分布
                normal_samples = np.random.multivariate_normal(mu, sigma, size=n_points)
                log_samples = np.exp(normal_samples)

                # 关键：减去众数，使其对齐像素中心
                samples = log_samples - theoretical_mode

                # 平移至目标中心点
                samples[:, 0] += center_x
                samples[:, 1] += center_y

                PointsCloud.append(samples)

    # 合并所有点为一个 (N, 2) 的数组
    if PointsCloud:
        PointsCloud = np.concatenate(PointsCloud, axis=0)
    else:
        PointsCloud = np.empty((0, 2))

    return PointsCloud, point_counts


def visualize_point_cloud(point_cloud, tactile_image=None, title="Generated Point Cloud"):
    """
    可视化生成的点云

    参数:
        point_cloud: 点云数据，形状为 (N, 2)
        tactile_image: 原始触觉图像（可选，用于背景）
        title: 图像标题
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    # 如果有触觉图像，先显示为背景
    if tactile_image is not None:
        rows, cols = tactile_image.shape
        extent = [1, cols, 1, rows]  # 注意坐标范围
        im = ax.imshow(tactile_image, extent=extent, cmap='viridis', alpha=0.7, origin='upper')
        plt.colorbar(im, ax=ax, label='Probability Density')

    # 绘制点云
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1],
               c='red', s=5, alpha=0.6, label='Points')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"生成的点云总数: {len(point_cloud)} 个点")


def save_point_cloud(point_cloud, filename):
    """
    将点云保存到文件

    参数:
        point_cloud: 点云数据
        filename: 文件名
    """
    np.savetxt(filename, point_cloud, delimiter=',', header='x,y', comments='')
    print(f"点云已保存到: {filename}")


def load_point_cloud(filename):
    """
    从文件加载点云

    参数:
        filename: 文件名

    返回:
        point_cloud: 加载的点云数据
    """
    point_cloud = np.loadtxt(filename, delimiter=',')
    print(f"从 {filename} 加载了 {len(point_cloud)} 个点")
    return point_cloud