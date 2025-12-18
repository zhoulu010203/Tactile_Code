import serial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # 用于绘制几何图形
import joblib
import time
import sys
import os

# ================= 配置区域 =================
# 1. 硬件与模型配置 (保持不变)
THRESHOLD_VALUE = 0.2
MODEL_FILE = 'shape_model.pkl'

# 2. 可视化配置 【关键修改】
SHOW_SENSOR_HEATMAP = False # True: 显示传感器阵列热力图; False: 显示几何形状动画


# ===========================================

class SensorDevice:
    def __init__(self):
        # 映射矩阵
        self.mapping = np.array([
            [0, 2, 4, 6, 1, 3, 5, 7],
            [14, 12, 10, 8, 9, 11, 13, 15],
            [22, 20, 18, 16, 30, 28, 26, 24],
            [23, 21, 19, 17, 31, 29, 27, 25],
            [38, 36, 34, 32, 46, 44, 42, 40],
            [39, 37, 35, 33, 47, 45, 43, 41]
        ])

        print("正在连接硬件传感器 (COM3, COM10, COM9)...")
        try:
            self.ser1 = serial.Serial('COM3', 115200, timeout=0.1)
            self.ser2 = serial.Serial('COM10', 115200, timeout=0.1)
            self.ser3 = serial.Serial('COM9', 115200, timeout=0.1)
            print("所有串口连接成功！")
        except serial.SerialException as e:
            print(f"串口连接失败: {e}")
            sys.exit(1)

    def read_data(self):
        try:
            line1 = self.ser1.readline().decode().strip()
            line2 = self.ser2.readline().decode().strip()
            line3 = self.ser3.readline().decode().strip()

            if not line1 or not line2 or not line3: return None

            values1 = [float(x) for x in line1.split(',')]
            values2 = [float(x) for x in line2.split(',')]
            values3 = [float(x) for x in line3.split(',')]

            combined_vals = values1 + values2 + values3
            if len(combined_vals) != 48: return None

            tactile_image = np.array(combined_vals).reshape(6, 8)
            flat_image = tactile_image.flatten()
            mapped_image = flat_image[self.mapping].reshape(6, 8)
            binary_image = (mapped_image >= THRESHOLD_VALUE).astype(int)

            return binary_image
        except:
            return None


def draw_geometry(ax, shape_idx, confidence):
    """
    在画布上绘制几何图形
    shape_idx: 0=Square, 1=Rect, 2=Circle, 3=Tri, -1=None
    """
    ax.clear()  # 清除上一帧的图形

    # 设置固定的坐标轴范围，防止图形跳动
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')  # 隐藏坐标轴刻度

    # 颜色逻辑：置信度高显示绿色，低显示橙色
    color = '#32CD32' if confidence > 0.8 else '#FFA500'  # LimeGreen or Orange

    if shape_idx == -1:
        # === 无接触状态 ===
        # 画一个灰色的虚线圆圈表示"等待中"
        circle = mpatches.Circle((0, 0), 0.2, color='lightgray', alpha=0.5)
        ax.add_patch(circle)
        ax.text(0, -0.5, "Waiting for Touch...", ha='center', color='gray', fontsize=12)

    elif shape_idx == 0:
        # === Square (正方形) ===
        # 中心在(0,0)，边长1.4
        rect = mpatches.Rectangle((-0.7, -0.7), 1.4, 1.4, color=color)
        ax.add_patch(rect)
        ax.text(0, 0, "Square", ha='center', va='center', color='white', fontsize=15, fontweight='bold')

    elif shape_idx == 1:
        # === Rectangle (长方形) ===
        # 宽1.8，高1.0
        rect = mpatches.Rectangle((-0.9, -0.5), 1.8, 1.0, color=color)
        ax.add_patch(rect)
        ax.text(0, 0, "Rectangle", ha='center', va='center', color='white', fontsize=15, fontweight='bold')

    elif shape_idx == 2:
        # === Circle (圆形) ===
        # 半径 0.8
        circle = mpatches.Circle((0, 0), 0.8, color=color)
        ax.add_patch(circle)
        ax.text(0, 0, "Circle", ha='center', va='center', color='white', fontsize=15, fontweight='bold')

    elif shape_idx == 3:
        # === Triangle (三角形) ===
        # 定义三个顶点的坐标
        points = [[0, 0.8], [-0.8, -0.6], [0.8, -0.6]]
        tri = mpatches.Polygon(points, closed=True, color=color)
        ax.add_patch(tri)
        ax.text(0, -0.1, "Triangle", ha='center', va='center', color='white', fontsize=15, fontweight='bold')


def realtime_recognition():
    # 1. 加载模型
    if not os.path.exists(MODEL_FILE):
        print(f"错误：找不到模型 {MODEL_FILE}")
        return
    model = joblib.load(MODEL_FILE)
    print("模型加载成功！")

    # 2. 初始化硬件
    device = SensorDevice()

    # 3. 初始化窗口
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))  # 正方形窗口更适合展示形状

    # 如果要显示热力图，保留旧的初始化逻辑
    if SHOW_SENSOR_HEATMAP:
        img = ax.imshow(np.zeros((6, 8)), cmap='gray', vmin=0, vmax=1)
        ax.set_title("Sensor Array View")
    else:
        # 如果显示几何图形，先画个初始状态
        draw_geometry(ax, -1, 0)

    print("\n=== 视觉化识别开始 ===")

    try:
        while plt.fignum_exists(fig.number):
            frame = device.read_data()
            if frame is None:
                plt.pause(0.01)
                continue

            # --- 识别逻辑 ---
            shape_idx = -1
            confidence = 0.0
            info_text = ""

            if np.sum(frame) > 0:  # 有按压
                flat_data = frame.flatten().reshape(1, -1)
                pred_idx = int(model.predict(flat_data)[0])
                probs = model.predict_proba(flat_data)[0]

                shape_idx = pred_idx
                confidence = probs[pred_idx]
                info_text = f"Confidence: {confidence * 100:.1f}%"

            # --- 分支：决定如何显示 ---
            if SHOW_SENSOR_HEATMAP:
                # 方式 A: 原始热力图 (你不需要显示，但代码保留在这里)
                img.set_data(frame)
                title = f"Detected: {['Square', 'Rect', 'Circle', 'Tri'][shape_idx]}" if shape_idx != -1 else "No Touch"
                ax.set_title(f"{title}\n{info_text}")
            else:
                # 方式 B: 几何图形模式 (你要求的模式)
                draw_geometry(ax, shape_idx, confidence)
                # 在窗口顶部显示置信度
                if shape_idx != -1:
                    ax.set_title(f"AI Recognition Result\n{info_text}", fontsize=14)
                else:
                    ax.set_title("AI Recognition Result\nWaiting...", fontsize=14)

            plt.pause(0.02)

    except KeyboardInterrupt:
        print("\n程序已停止。")


if __name__ == '__main__':
    realtime_recognition()