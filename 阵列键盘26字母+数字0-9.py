'''
6×8阵列式传感器触觉重构算法 - 交互式打字机版 (含双键组合输入0)
'''
from sklearn.mixture import GaussianMixture
from connected_component_labeling import connected_component_labeling
from point_cloud_generation import generate_point_cloud
from PatternRecognition import count_patterns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import serial
import csv
import datetime
import time
from virtual_keyboard_viz import MatrixKeyboard

# ==========================================
# 全局打字机状态变量
# ==========================================
text_buffer = ""  # 存储当前输入的字符串
last_key_time = 0  # 上一次按键生效的时间
last_detected_key = None  # 上一帧检测到的按键
KEY_COOLDOWN = 1.0  # 按键冷却时间(秒)


# ==========================================
# 按键监听
# ==========================================
def on_key_press(event):
    global text_buffer

    if event.key == 's':
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_3d = f"Fig_3D_{timestamp}.pdf"
        filename_2d = f"Fig_Keyboard_{timestamp}.pdf"
        print(f"\n[系统正在保存]...")
        renderer = fig.canvas.get_renderer()
        extent_3d = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(filename_3d, format='pdf', bbox_inches=extent_3d, pad_inches=-1)
        extent_2d = ax_face.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        extent_2d = extent_2d.expanded(1.05, 1.05)
        fig.savefig(filename_2d, format='pdf', bbox_inches=extent_2d)
        print(f"  -> 保存完成")

    elif event.key == 'c':
        text_buffer = ""
        print("[System] Buffer Cleared")

    elif event.key == 'backspace':
        text_buffer = text_buffer[:-1]
        print("[System] Backspace")


# ==========================================
# 绘图风格与参数
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

mapping = np.array([
    [0, 2, 4, 6, 1, 3, 5, 7],
    [14, 12, 10, 8, 9, 11, 13, 15],
    [22, 20, 18, 16, 30, 28, 26, 24],
    [23, 21, 19, 17, 31, 29, 27, 25],
    [38, 36, 34, 32, 46, 44, 42, 40],
    [39, 37, 35, 33, 47, 45, 43, 41]
])

SHOW_KEYBOARD = True
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
my_keyboard = MatrixKeyboard(x_range=(10, 80), y_range=(10, 60))

fig = plt.figure(figsize=(14, 7), facecolor='white')

# 左侧 3D
ax = fig.add_subplot(121, projection='3d')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

# 右侧 2D
ax_face = fig.add_subplot(122)

x = np.linspace(0, 100, 160)
y = np.linspace(0, 70, 96)
X, Y = np.meshgrid(x, y)
Z_plane = np.zeros_like(X)
grid = np.column_stack([X.ravel(), Y.ravel()])

ax.set_xlabel("X Axis (mm)")
ax.set_ylabel("Y Axis (mm)")
ax.set_zlim(0, 0.005)
ax.view_init(elev=90, azim=-90)
ax.set_zticks([])

fig.canvas.mpl_connect('key_press_event', on_key_press)

# 串口初始化
try:
    ser1 = serial.Serial('COM13', 115200)
    ser2 = serial.Serial('COM15', 115200)
    ser3 = serial.Serial('COM12', 115200)

    while True:
        save_f_distributed = 0
        save_component_means = 0

        ser1.reset_input_buffer()
        ser2.reset_input_buffer()
        ser3.reset_input_buffer()

        try:
            ser1.readline()
            ser2.readline()
            ser3.readline()
        except:
            pass

        data1 = ser1.readline().decode().strip()
        data2 = ser2.readline().decode().strip()
        data3 = ser3.readline().decode().strip()

        try:
            values1 = [float(x) for x in data1.split(',')]
            values2 = [float(x) for x in data2.split(',')]
            values3 = [float(x) for x in data3.split(',')]
        except ValueError:
            continue

        tactile_image = np.array(values1 + values2 + values3).reshape(6, 8)

        flat_image = tactile_image.flatten()
        flat_image_transformed = (p1 * flat_image + p2)
        tactile_image = flat_image_transformed[mapping].reshape(6, 8).round(1)
        print(tactile_image)

        binary_image = (tactile_image >= 1.2).astype(int)
        count_ones = np.sum(binary_image)
        F_sum = np.sum(tactile_image[tactile_image > 1.2])

        # ==========================================
        # 绘图逻辑
        # ==========================================
        ax.clear()
        ax_face.clear()

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 70)
        ax.set_zlim(0, 0.005)
        ax.set_box_aspect((100, 70, 10))
        ax.set_title("Tactile Reconstruction", fontweight='bold')

        ax_face.set_xlim(0, 90)
        ax_face.set_ylim(0, 80)
        ax_face.set_aspect('equal')
        ax_face.axis('off')
        ax.set_zticks([])

        # LCD Window
        lcd_rect = patches.Rectangle((10, 65), 70, 12, linewidth=2, edgecolor='black', facecolor='#f0f0f0')
        ax_face.add_patch(lcd_rect)
        ax_face.text(12, 75, "OUTPUT:", fontsize=18, color='gray', fontweight='bold')

        display_text = text_buffer + ("_" if int(time.time() * 2) % 2 == 0 else "")
        ax_face.text(45, 71, display_text, ha='center', va='center', fontsize=24,
                     fontname='Monospace', color='blue', fontweight='bold')

        ax_face.text(45, 5, "Key: 'C' Clear | 'Back' Del | 2 Keys = '0'", ha='center', fontsize=16, color='gray')

        if np.all(binary_image == 0):
            ax.plot_surface(X, Y, Z_plane, color='whitesmoke', alpha=0.5)
            my_keyboard.setup_plot(ax_face)
            last_detected_key = None

        else:
            labels, num_components = connected_component_labeling(binary_image)
            count_pattern = count_patterns(binary_image)
            if count_pattern: num_components += count_pattern

            if count_ones <= 4 * num_components:
                # Point Contact
                PointsCloud, _ = generate_point_cloud(
                    tactile_image=tactile_image, binary_image=binary_image,
                    spacing=spacing, mu_pointcloud=mu_pointcloud,
                    sigma_pointcloud=sigma_pointcloud, min_points=min_points,
                    scale_factor=scale_factor
                )

                if len(PointsCloud) > 0:
                    # 如果检测到的组件数小于2，强制至少设为1防止GMM报错；
                    # 如果确实有两个独立区域，connected_component_labeling 会返回 num_components >= 2
                    n_comp_safe = max(1, num_components)

                    gmm = GaussianMixture(n_components=n_comp_safe, covariance_type='spherical', random_state=0)
                    gmm.fit(PointsCloud)
                    component_means = gmm.means_

                    log_prob = gmm.score_samples(grid)
                    Z = np.exp(log_prob).reshape(X.shape)
                    ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none', alpha=1.0)
                    ax.contourf(X, Y, Z, zdir='z', offset=0, cmap='plasma', alpha=0.3)

                    if SHOW_KEYBOARD:
                        # 返回当前帧检测到的所有按键列表
                        detected_keys = my_keyboard.update(ax_face, component_means)

                        if detected_keys:
                            # ==========================================
                            # 【修改】双键判定逻辑
                            # ==========================================
                            target_input = None

                            # 1. 判断是否按下任意两个(或更多)键
                            if len(detected_keys) >= 2:
                                target_input = '0'
                            # 2. 单键输入
                            else:
                                target_input = detected_keys[0]

                            current_time = time.time()

                            if len(text_buffer) < 20:
                                # 状态变更判定：
                                # 如果输入的内容变了（比如从 'A' 变成了 '0'），或者冷却时间到了
                                if (target_input != last_detected_key) or \
                                        (current_time - last_key_time > KEY_COOLDOWN):
                                    text_buffer += target_input
                                    last_key_time = current_time
                                    last_detected_key = target_input
                                    print(
                                        f"输入: {target_input} | 模式: {'组合键' if len(detected_keys) > 1 else '单键'} | Buffer: {text_buffer}")
                        else:
                            last_detected_key = None

            else:
                # 面接触
                ax.plot_surface(X, Y, Z_plane, color='whitesmoke', alpha=0.5)
                ax_face.imshow(tactile_image[::-1], cmap='inferno', extent=[10, 80, 10, 60], origin='lower')
                my_keyboard.setup_plot(ax_face)

        plt.draw()
        plt.pause(0.001)

except KeyboardInterrupt:
    print("程序结束")
finally:
    plt.ioff()
    plt.show()