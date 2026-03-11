import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time


class MatrixKeyboard:
    def __init__(self, x_range=(10, 80), y_range=(10, 60), rows=5, cols=7):
        """
        初始化虚拟矩阵键盘
        :param x_range: 物理尺寸 X 范围 (min, max)，单位 mm
        :param y_range: 物理尺寸 Y 范围 (min, max)，单位 mm
        :param rows: 行数
        :param cols: 列数
        """
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.rows = rows
        self.cols = cols

        # 计算单个按键尺寸
        self.key_width = (self.x_max - self.x_min) / self.cols
        self.key_height = (self.y_max - self.y_min) / self.rows

        # 定义字符映射 (35个键位: A-Z + 1-9)
        # 视觉习惯：从左上角开始为A，到右下角为9
        # 物理坐标：Matplotlib origin='lower' (左下角为0,0)
        # 因此我们需要倒序排列行，让第一行字符出现在 Y 轴最大值处
        chars_raw = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"

        self.grid_map = {}  # 用于存储 (row, col) -> char

        # 构建 grid_map
        idx = 0
        # 我们从上往下生成字符 (视觉行)，但在坐标系中是从高 Y 到低 Y
        for r_visual in range(rows):
            for c in range(cols):
                if idx < len(chars_raw):
                    char = chars_raw[idx]
                    # 物理行 r_phys = (rows - 1) - r_visual
                    # 这样 visual_row 0 (A-G) 对应 物理 row 4 (最顶端)
                    r_phys = (rows - 1) - r_visual
                    self.grid_map[(r_phys, c)] = char
                    idx += 1

    def setup_plot(self, ax):
        """
        绘制静态背景（网格线和字符），只需调用一次
        """
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Tactile Virtual Keyboard", fontweight='bold')

        # 绘制背景网格
        for r in range(self.rows + 1):
            y = self.y_min + r * self.key_height
            ax.hlines(y, self.x_min, self.x_max, colors='#bdc3c7', linestyles='-', linewidth=1)

        for c in range(self.cols + 1):
            x = self.x_min + c * self.key_width
            ax.vlines(x, self.y_min, self.y_max, colors='#bdc3c7', linestyles='-', linewidth=1)

        # 绘制字符
        for (r, c), char in self.grid_map.items():
            cx = self.x_min + (c + 0.5) * self.key_width
            cy = self.y_min + (r + 0.5) * self.key_height
            ax.text(cx, cy, char, ha='center', va='center', fontsize=14,
                    color='#2c3e50', fontweight='bold', alpha=0.8)

    def update(self, ax, contact_points):
        """
        更新动态显示（高亮按键）
        :param ax: matplotlib axis 对象
        :param contact_points: 接触点列表 [[x1, y1], [x2, y2], ...]
        :return: 识别到的字符列表
        """
        # 1. 清除旧的高亮块 (保留文本和网格线其实很难只清patch，通常建议clear重画，
        #    但为了性能，这里我们移除所有 'patch' 类型的对象，或者由外部 ax.clear() 控制)
        #    **注意**：为了配合你的主程序逻辑 (ax.clear())，这里假设每次传入的 ax 都是空的或者已经 clear 过的。
        #    所以我们需要重新调用 setup_plot 或者由主程序只画 patches。

        # 为了适应你的主代码风格（每帧 ax.clear()），我们在这里重新绘制所有内容
        # 如果追求极致性能，可以将 static 内容画在背景，但这在 Matplotlib 中较复杂。
        self.setup_plot(ax)

        detected_chars = []

        # 2. 绘制高亮和点
        if contact_points is not None and len(contact_points) > 0:
            for point in contact_points:
                px, py = point[0], point[1]

                # 边界检查
                if not (self.x_min <= px <= self.x_max and self.y_min <= py <= self.y_max):
                    continue

                # 计算网格索引
                c_idx = int((px - self.x_min) / self.key_width)
                r_idx = int((py - self.y_min) / self.key_height)

                # 修正索引防越界
                c_idx = min(c_idx, self.cols - 1)
                r_idx = min(r_idx, self.rows - 1)

                target_char = self.grid_map.get((r_idx, c_idx), "")

                if target_char:
                    detected_chars.append(target_char)

                    # 绘制红色高亮矩形
                    rect_x = self.x_min + c_idx * self.key_width
                    rect_y = self.y_min + r_idx * self.key_height

                    # 红色半透明方块
                    rect = patches.Rectangle(
                        (rect_x, rect_y), self.key_width, self.key_height,
                        linewidth=2, edgecolor='#e74c3c', facecolor='#e74c3c', alpha=0.4
                    )
                    ax.add_patch(rect)

                    # 绘制精确触点（黄色叉号）
                    ax.plot(px, py, marker='x', color='yellow', markersize=10, markeredgewidth=2)

        return list(set(detected_chars))  # 去重返回


# ==========================================
# 独立运行测试代码 (Demo)
# ==========================================
if __name__ == "__main__":
    print("正在启动键盘可视化独立测试...")

    # 初始化
    kb = MatrixKeyboard()

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.ion()  # 开启交互模式

    try:
        step = 0
        while True:
            ax.clear()  # 每一帧清空画布

            # 1. 模拟生成随机接触点 (模拟 component_means)
            # 让点在键盘上移动：生成 1 到 2 个随机点
            import random

            simulated_points = []

            # 模拟一个从左向右移动的点
            moving_x = (step * 2) % 80
            moving_y = 30 + 20 * np.sin(step * 0.1)  # 正弦波轨迹
            simulated_points.append([moving_x, moving_y])

            # 偶尔随机出现第二个点
            if random.random() > 0.7:
                simulated_points.append([random.uniform(0, 80), random.uniform(0, 60)])

            # 2. 更新键盘视图
            # 将模拟的点传入 update 函数
            chars = kb.update(ax, simulated_points)

            # 3. 打印结果
            if chars:
                print(f"Frame {step}: 检测到按键 -> {chars}")
                # 在图上显示检测结果
                ax.text(40, 65, f"Output: {''.join(chars)}",
                        ha='center', fontsize=20, color='blue', fontweight='bold')
            else:
                ax.text(40, 65, "Waiting...", ha='center', fontsize=16, color='gray')

            # 保持比例
            ax.set_aspect('equal')

            plt.draw()
            plt.pause(0.1)  # 模拟 10Hz 刷新率
            step += 1

    except KeyboardInterrupt:
        print("测试结束")
        plt.close()