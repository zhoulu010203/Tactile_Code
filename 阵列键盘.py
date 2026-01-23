'''
6×8阵列式传感器触觉重构算法 - TRO期刊风格优化版
'''
from sklearn.mixture import GaussianMixture
from connected_component_labeling import connected_component_labeling
from point_cloud_generation import generate_point_cloud
from PatternRecognition import count_patterns
import numpy as np
import matplotlib.pyplot as plt
import serial
import csv
import datetime
from virtual_keyboard_viz import MatrixKeyboard


# ==========================================
# 【新增】键盘按键监听：按 's' 保存矢量图
# ==========================================
def on_key_press(event):
    if event.key == 's':
        # 生成基础文件名（带时间戳）
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_3d = f"Fig_3D_Reconstruction_{timestamp}.pdf"
        filename_2d = f"Fig_2D_Heatmap_{timestamp}.pdf"
        print(f"\n[系统正在保存]...")

        # 2. 获取渲染器（用于计算子图的精确尺寸）
        renderer = fig.canvas.get_renderer()

        # --- 保存左图 (3D) ---
        # 获取 ax (3D图) 的紧凑包围盒，并转换为“英寸”单位
        extent_3d = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        # 适当向外扩展一点点，防止3D坐标轴标签被切掉 (x=1.1倍宽, y=1.1倍高)
        # extent_3d = extent_3d.expanded(1.1, 1.1)
        fig.savefig(filename_3d, format='pdf', bbox_inches=extent_3d, pad_inches=-1)
        print(f"  -> 已保存左侧3D图: {filename_3d}")

        # --- 保存右图 (2D) ---
        # 获取 ax_face (2D图) 的紧凑包围盒
        extent_2d = ax_face.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        # 2D图通常不需要额外扩展，或者只需微调
        extent_2d = extent_2d.expanded(1.05, 1.05)

        fig.savefig(filename_2d, format='pdf', bbox_inches=extent_2d)
        print(f"  -> 已保存右侧2D图: {filename_2d}")
        print("[保存完成]")


# ==========================================
# TRO 绘图风格设置 (全局)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # TRO 推荐字体
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['axes.linewidth'] = 1.0  # 坐标轴线宽

# ==========================================
# 参数设定 (保持不变)
# ==========================================
mapping = np.array([
    [0, 2, 4, 6, 1, 3, 5, 7],
    [14, 12, 10, 8, 9, 11, 13, 15],
    [22, 20, 18, 16, 30, 28, 26, 24],
    [23, 21, 19, 17, 31, 29, 27, 25],
    [38, 36, 34, 32, 46, 44, 42, 40],
    [39, 37, 35, 33, 47, 45, 43, 41]
])

SHOW_KEYBOARD = False
p1 = 1.763
p2 = -0.1261
mu_pointcloud = [0.0, 0.0]
sigma_pointcloud = [[0.5, 0.0], [0.0, 0.5]]
min_points = 20
scale_factor = 100
spacing = 10

# ==========================================
# 可视化初始化
# ==========================================
plt.ion()
# 初始化键盘，参数对应你的物理尺寸 80mm x 60mm
my_keyboard = MatrixKeyboard(x_range=(0, 80), y_range=(0, 60))
# 调整 figure 尺寸，宽屏更适合展示对比
fig = plt.figure(figsize=(14, 6), facecolor='white')

# --- 左侧：3D 重构图 ---
ax = fig.add_subplot(121, projection='3d')
# 移除3D图默认的灰色背景板，改为白色透明
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)  # 网格线调淡

# --- 右侧：2D 热力图 ---
ax_face = fig.add_subplot(122)

# 初始化坐标网格
x = np.linspace(0, 100, 160)
y = np.linspace(0, 70, 96)
X, Y = np.meshgrid(x, y)
Z_plane = np.zeros_like(X)
grid = np.column_stack([X.ravel(), Y.ravel()])

# 初始绘图状态
surf = ax.plot_surface(X, Y, Z_plane, color='whitesmoke', alpha=0.3)  # 初始平面改用浅灰/白色
ax.set_title("Tactile Reconstruction (3D)", fontweight='bold', pad=10)
ax.set_xlabel("X Axis (mm)")  # 加上单位
ax.set_ylabel("Y Axis (mm)")
ax.set_zlim(0, 0.005)
ax.view_init(elev=90, azim=-90)  # 设置一个比较好看的视角
ax.set_zticks([])

ax_face.set_title("Contact Pressure Distribution (2D)", fontweight='bold', pad=10)
ax_face.set_aspect('equal')  # 保证比例一致

# ==========================================
# 数据记录初始化 (保持不变)
# ==========================================
timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"tactile_data_{timestamp_str}.csv"
f_csv = open(csv_filename, 'w', newline='')
writer = csv.writer(f_csv)
writer.writerow(["Timestamp", "Force_Sensor", "CH1", "CH2", "CH3", "CH4", "F_distributed", "Component_Means"])
print(f"数据将保存至: {csv_filename}")

# 将按键事件绑定到图形窗口
fig.canvas.mpl_connect('key_press_event', on_key_press)

# 串口初始化 (保持不变)
try:
    # ser1 = serial.Serial('COM12', 115200)
    # ser2 = serial.Serial('COM15', 115200)
    # ser3 = serial.Serial('COM14', 115200)

    while True:
        # 数据初始化
        save_f_distributed = 0
        save_component_means = 0

        # # 强制清空这三个主要数据串口的缓存
        # ser1.reset_input_buffer()
        # ser2.reset_input_buffer()
        # ser3.reset_input_buffer()
        #
        # try:
        #     ser1.readline()
        #     ser2.readline()
        #     ser3.readline()
        # except:
        #     pass

        # # 读取传感器数据
        # data1 = ser1.readline().decode().strip()
        # data2 = ser2.readline().decode().strip()
        # data3 = ser3.readline().decode().strip()

        # values1 = [float(x) for x in data1.split(',')]
        # values2 = [float(x) for x in data2.split(',')]
        # values3 = [float(x) for x in data3.split(',')]

        values1 = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        values2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        values3 = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]

        tactile_image = np.array(values1 + values2 + values3).reshape(6, 8)
        flat_image = tactile_image.flatten()
        flat_image_transformed = (p1 * flat_image + p2)
        # tactile_image = flat_image_transformed[mapping].reshape(6, 8).round(1)
        tactile_image = flat_image_transformed.reshape(6, 8).round(1)  # 测试
        # tactile_image[1, 4] = 0.1  # 这个传感器不稳定
        binary_image = (tactile_image >= 0.5).astype(int)
        count_ones = np.sum(binary_image)
        F_sum = np.sum(tactile_image[tactile_image > 0.15])

        # ==========================================
        # 绘图逻辑优化
        # ==========================================
        ax.clear()
        ax_face.clear()

        # 重新设置3D背景 (clear后会被重置，需要重新设定)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 70)
        ax.set_zlim(0, 0.005)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Density")

        # 这里设置为 100:70 的比例，Z轴设为 10 是为了保证有一定的视觉高度(如果旋转看的话)
        # 如果是纯俯视图，Z的值不影响长方形形状，只影响透视深度
        ax.set_box_aspect((100, 70, 10))

        if np.all(binary_image == 0):
            # 无接触状态：绘制浅色网格平面
            ax.plot_surface(X, Y, Z_plane, color='whitesmoke', alpha=0.5)
            ax.text2D(0.5, 0.5, "No Contact", transform=ax.transAxes, ha='center', color='gray')
            ax.set_title("Tactile Reconstruction (IDLE)", fontweight='bold')

            ax_face.text(0.5, 0.5, "No Contact", transform=ax_face.transAxes, ha='center', color='gray')
            ax_face.set_title("No Pressure Detected", fontweight='bold')

        else:
            # 有接触
            labels, num_components = connected_component_labeling(binary_image)
            count_pattern = count_patterns(binary_image)
            if count_pattern:
                num_components = num_components + count_pattern

            if count_ones <= 4 * num_components:
                # === 点接触模式 (GMM) ===
                PointsCloud, point_counts = generate_point_cloud(
                    tactile_image=tactile_image,
                    binary_image=binary_image,
                    spacing=spacing,
                    mu_pointcloud=mu_pointcloud,
                    sigma_pointcloud=sigma_pointcloud,
                    min_points=min_points,
                    scale_factor=scale_factor
                )

                if len(PointsCloud) == 0:
                    ax.plot_surface(X, Y, Z_plane, color='whitesmoke', alpha=0.5)
                else:
                    gmm = GaussianMixture(n_components=num_components, covariance_type='spherical', random_state=0)
                    gmm.fit(PointsCloud)
                    component_weights = gmm.weights_
                    component_means = gmm.means_
                    F_distributed = 0.575 * F_sum * gmm.weights_

                    # 数据保存准备
                    save_f_distributed = F_distributed.tolist()
                    save_component_means = component_means.tolist()
                    print("分配后的力 (F_distributed):", F_distributed)
                    print("估计位置 (component_means):", component_means)

                    # 计算分布并绘图
                    log_prob = gmm.score_samples(grid)
                    Z = np.exp(log_prob).reshape(X.shape)

                    # [TRO Style] 使用 plasma 配色，边缘线设为0，alpha适中
                    # ax.plot_surface(X,可视化优化.py Y, Z, cmap='plasma', edgecolor='none', alpha=1.0, antialiased=False)
                    # ax.set_title(f"Multi-Contact Reconstruction (N={num_components})", fontweight='bold')
                    surf = ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none', alpha=1.0, antialiased=False)
                    ax.set_rasterized(True)

                    # 可以在底部画出等高线投影，增加立体感
                    ax.contourf(X, Y, Z, zdir='z', offset=0, cmap='plasma', alpha=0.3)

                    # ax.axis('off')
                    # ax_face.set_xticks([])
                    # ax_face.set_yticks([])

                    if SHOW_KEYBOARD:
                        detected_keys = my_keyboard.update(ax_face, component_means)

                        if detected_keys:
                            keys_str = ", ".join(detected_keys)
                            print(f">>> 触发按键: {keys_str}")
                            ax_face.set_title(f"Keyboard Input: {keys_str}", fontsize=14, color='red', fontweight='bold')
                        else:
                            ax_face.set_title("Virtual Keyboard (Waiting)", fontsize=12)
                        # =======================================================


            else:
                # === 面/线接触模式 (Heatmap) ===
                ax.plot_surface(X, Y, Z_plane, color='whitesmoke', alpha=0.5)
                ax.set_title("Face Contact Mode", fontweight='bold')

                # [TRO Style] 2D图使用 inferno 配色，更符合现代期刊审美
                im = ax_face.imshow(tactile_image[::-1], cmap='inferno', interpolation='bicubic',
                                    extent=[0, 80, 0, 60], origin='lower')  # 调整extent以匹配物理尺寸比例
                # ax_face.set_title("Pressure Heatmap", fontweight='bold')
                ax_face.set_xlabel("Width (mm)")
                ax_face.set_ylabel("Height (mm)")
                ax_face.set_xticks([])
                ax_face.set_yticks([])
                # 去除不必要的边框 spines
                for spine in ax_face.spines.values():
                    spine.set_visible(False)

        plt.draw()
        plt.pause(0.001)

except KeyboardInterrupt:
    print("程序被用户中断")
except Exception as e:
    print(f"发生错误: {e}")
finally:

    plt.ioff()
    plt.show()
