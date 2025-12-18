# import serial
# import numpy as np
#
# mapping = np.array([
#     [0, 2, 4, 6, 1, 3, 5, 7],
#     [14, 12, 10, 8, 9, 11, 13, 15],
#     [22, 20, 18, 16, 30, 28, 26, 24],
#     [23, 21, 19, 17, 31, 29, 27, 25],
#     [38, 36, 34, 32, 46, 44, 42, 40],
#     [39, 37, 35, 33, 47, 45, 43, 41]
# ])
#
# ser1 = serial.Serial('COM3', 115200)
# ser2 = serial.Serial('COM10', 115200)
# ser3 = serial.Serial('COM9', 115200)
#
# # 读取三个串口的数据
# data1 = ser1.readline().decode().strip()
# data2 = ser2.readline().decode().strip()
# data3 = ser3.readline().decode().strip()
#
# # 转换为浮点数列表
# values1 = [float(x) for x in data1.split(',')]
# values2 = [float(x) for x in data2.split(',')]
# values3 = [float(x) for x in data3.split(',')]
#
# # 直接将三个列表组合成6×8的图像
# tactile_image = np.array(values1 + values2 + values3).reshape(6, 8)
# flat_image = tactile_image.flatten()
# tactile_image = flat_image[mapping].reshape(6, 8).round(2)
# binary_image = (tactile_image >= 0.18).astype(int)
# save_data = binary_image.flatten()

import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import time

# ================= 配置区域 =================
SAVE_FILENAME = "sensor_data.csv"
THRESHOLD_VALUE = 0.2  # 你代码中设定的阈值


# ===========================================

class SensorDevice:
    def __init__(self):
        # 定义映射矩阵 (来自你的代码)
        self.mapping = np.array([
            [0, 2, 4, 6, 1, 3, 5, 7],
            [14, 12, 10, 8, 9, 11, 13, 15],
            [22, 20, 18, 16, 30, 28, 26, 24],
            [23, 21, 19, 17, 31, 29, 27, 25],
            [38, 36, 34, 32, 46, 44, 42, 40],
            [39, 37, 35, 33, 47, 45, 43, 41]
        ])

        print("正在连接串口...")
        try:
            # 增加 timeout 防止程序卡死
            self.ser1 = serial.Serial('COM3', 115200, timeout=0.1)
            self.ser2 = serial.Serial('COM10', 115200, timeout=0.1)
            self.ser3 = serial.Serial('COM9', 115200, timeout=0.1)
            print("串口连接成功！")
        except serial.SerialException as e:
            print(f"串口连接失败: {e}")
            print("请检查 USB 连接或关闭其他占用串口的软件。")
            exit()

    def read_data(self):
        """
        读取三个串口数据 -> 组合 -> 映射 -> 二值化
        返回: (6, 8) 的二值化 numpy 数组 (int 0或1)
        """
        try:
            # 1. 读取数据
            line1 = self.ser1.readline().decode().strip()
            line2 = self.ser2.readline().decode().strip()
            line3 = self.ser3.readline().decode().strip()

            # 简单的防错检查：如果数据为空，返回全黑图像
            if not line1 or not line2 or not line3:
                return np.zeros((6, 8))

            # 2. 转换为浮点数列表
            values1 = [float(x) for x in line1.split(',')]
            values2 = [float(x) for x in line2.split(',')]
            values3 = [float(x) for x in line3.split(',')]

            # 3. 组合并重塑 (你的逻辑)
            # 注意：需确保总长度为 48，否则 reshape 会报错
            combined_vals = values1 + values2 + values3
            if len(combined_vals) != 48:
                print(f"警告: 数据长度错误 ({len(combined_vals)}), 期望 48")
                return np.zeros((6, 8))

            tactile_image = np.array(combined_vals).reshape(6, 8)

            # 4. 应用映射 (你的逻辑)
            flat_image = tactile_image.flatten()
            mapped_image = flat_image[self.mapping].reshape(6, 8).round(2)

            # 5. 二值化 (你的逻辑)
            binary_image = (mapped_image >= THRESHOLD_VALUE).astype(int)

            return binary_image

        except ValueError:
            # 经常会遇到串口发来半截数据导致转换失败，忽略即可
            return np.zeros((6, 8))
        except Exception as e:
            print(f"读取错误: {e}")
            return np.zeros((6, 8))


def collect_gui():
    device = SensorDevice()
    dataset = []
    labels = []

    label_names = {0: 'Square', 1: 'Rectangle', 2: 'Circle', 3: 'Triangle'}

    # === 创建可视化窗口 ===
    fig, ax = plt.subplots(figsize=(8, 7))
    plt.subplots_adjust(bottom=0.25)  # 底部留白给按钮

    # 初始化图像：cmap='gray' 黑白显示, vmin=0(黑), vmax=1(白)
    img = ax.imshow(np.zeros((6, 8)), cmap='gray', vmin=0, vmax=1)
    ax.set_title("Tactile Sensor Collection\nReal-time Binary Image", fontsize=12)

    # 全局变量，用于存储当前正在显示的这一帧数据
    current_binary_frame = np.zeros((6, 8))

    # === 保存函数 ===
    def save_current_frame(label_idx):
        nonlocal current_binary_frame

        # 将 (6,8) 矩阵拉平为 48维 向量保存
        flat_data = current_binary_frame.flatten()

        dataset.append(flat_data)
        labels.append(label_idx)

        print(f"已保存: {label_names[label_idx]} | 总样本数: {len(dataset)}")
        ax.set_title(f"Last Saved: {label_names[label_idx]} (Count: {len(dataset)})", color='green')
        fig.canvas.draw_idle()

    # === 按钮点击事件 ===
    def btn_square(event):
        save_current_frame(0)

    def btn_rect(event):
        save_current_frame(1)

    def btn_circle(event):
        save_current_frame(2)

    def btn_tri(event):
        save_current_frame(3)

    # === 布局按钮 ===
    # [left, bottom, width, height]
    ax1 = plt.axes([0.1, 0.05, 0.15, 0.075])
    ax2 = plt.axes([0.3, 0.05, 0.15, 0.075])
    ax3 = plt.axes([0.5, 0.05, 0.15, 0.075])
    ax4 = plt.axes([0.7, 0.05, 0.15, 0.075])

    b1 = Button(ax1, 'Square (Q)')
    b1.on_clicked(btn_square)
    b2 = Button(ax2, 'Rect (W)')
    b2.on_clicked(btn_rect)
    b3 = Button(ax3, 'Circle (E)')
    b3.on_clicked(btn_circle)
    b4 = Button(ax4, 'Triangle (R)')
    b4.on_clicked(btn_tri)

    # === 键盘事件支持 ===
    def on_key(event):
        key_map = {'q': 0, 'w': 1, 'e': 2, 'r': 3}
        if event.key in key_map:
            save_current_frame(key_map[event.key])

    fig.canvas.mpl_connect('key_press_event', on_key)

    print("=== 开始采集 ===")
    print("窗口显示的是二值化后的图像。")
    print("请用手按压传感器，看到清晰形状后，点击按钮保存。")

    # === 主循环 ===
    try:
        while plt.fignum_exists(fig.number):
            # 1. 硬件读取
            current_binary_frame = device.read_data()

            # 2. 更新画面
            img.set_data(current_binary_frame)
            plt.pause(0.01)  # 必须有暂停，否则界面会卡死

    except KeyboardInterrupt:
        pass

    # === 退出后写入文件 ===
    if dataset:
        print(f"\n正在保存 {len(dataset)} 条数据到 '{SAVE_FILENAME}' ...")
        full_data = np.column_stack((np.array(dataset), np.array(labels)))
        # 保存为整数格式 (%d)，因为全是 0 和 1
        # np.savetxt(SAVE_FILENAME, full_data, delimiter=",", fmt='%d')
        with open(SAVE_FILENAME, 'ab') as f:
            np.savetxt(f, full_data, delimiter=",", fmt='%d')
        print("文件保存成功！")
    else:
        print("未保存任何数据。")



if __name__ == '__main__':
    collect_gui()