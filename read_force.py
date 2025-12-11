import serial
import time

# ==========================================
# 1. 配置区域 (来自 Data_caiji.py)
# ==========================================
FORCE_PORT = 'COM6'  # 你原来的配置是 COM6
FORCE_BAUD = 2400  # 你原来的配置是 2400


# ==========================================
# 2. 核心类：力传感器读取器
# ==========================================
class ForceSensorReader:
    def __init__(self, port, baud):
        # 初始化串口
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.buffer = ""
        self.last_valid_force = 0.0  # 存储最新的有效读数

    def read_latest(self):
        """
        读取缓冲区中的所有数据，处理并返回最新的有效力值。
        如果没有新数据，返回上一次的有效值。
        """
        if self.ser.in_waiting > 0:
            try:
                # 读取串口数据
                raw_data = self.ser.read(self.ser.in_waiting)
                data_str = raw_data.decode('utf-8', errors='ignore')

                # 过滤特殊字符
                filtered_data = data_str.replace('\r', '').replace('\n', '')
                self.buffer += filtered_data

                # 处理缓冲区中所有完整的6字符数据包
                # (这也是从 Data_caiji.py 中提取的核心逻辑)
                while len(self.buffer) >= 6:
                    force_str = self.buffer[:6]
                    self.buffer = self.buffer[6:]  # 移出已处理部分

                    try:
                        # 尝试转换，更新最新值
                        val = float(force_str)
                        self.last_valid_force = val
                    except ValueError:
                        pass  # 忽略错误数据
            except Exception as e:
                print(f"Force Sensor Error: {e}")

        return self.last_valid_force

    def close(self):
        if self.ser.is_open:
            self.ser.close()


# ==========================================
# 3. 测试代码 (只有直接运行此文件时才会执行)
# ==========================================
if __name__ == "__main__":
    sensor = None
    try:
        print(f"正在连接力传感器 ({FORCE_PORT})...")
        sensor = ForceSensorReader(FORCE_PORT, FORCE_BAUD)
        print("连接成功！开始读取数据 (按 Ctrl+C 停止):")

        while True:
            # 获取当前力值
            force = sensor.read_latest()
            print(f"Current Force: {force:.3f}")
            time.sleep(0.1)  # 稍微延时，方便观察

    except serial.SerialException:
        print(f"错误：无法打开串口 {FORCE_PORT}。")
        print("请检查串口是否被其他程序（如 Data_caiji.py）占用。")
    except KeyboardInterrupt:
        print("\n程序已停止。")
    finally:
        if sensor:
            sensor.close()
            print("串口已关闭。")