'''四个单独薄膜压力传感器数据和测力计数据（已注释）读取存储代码'''

import serial
import csv
import time

# ================= 配置区域 =================
# 1. ADC 传感器配置 (四个通道电压)
ADC_PORT = 'COM13'  # 修改为 ADC 的串口
ADC_BAUD = 115200

# # 2. 力传感器配置 (Force Sensor)
# FORCE_PORT = 'COM6'  # 修改为 力传感器 的串口
# FORCE_BAUD = 2400

# 3. 文件保存配置
FILENAME = 'combined_data_2Contact.csv'


# ===========================================

# class ForceSensorReader:
#     """
#     专门负责读取力传感器的类
#     保持最新值 (Sample and Hold)
#     """
#
#     def __init__(self, port, baud):
#         self.ser = serial.Serial(port, baud, timeout=0.1)
#         self.buffer = ""
#         self.last_valid_force = 0.0  # 默认值，防止一开始读不到报错
#
#     def read_latest(self):
#         """尝试读取串口数据并更新 self.last_valid_force"""
#         if self.ser.in_waiting > 0:
#             try:
#                 raw_data = self.ser.read(self.ser.in_waiting)
#                 data_str = raw_data.decode('utf-8', errors='ignore')
#                 # 清理换行符
#                 filtered_data = data_str.replace('\r', '').replace('\n', '')
#                 self.buffer += filtered_data
#
#                 # 解析逻辑：按6字符一组 (根据你提供的 Force 协议)
#                 while len(self.buffer) >= 6:
#                     force_str = self.buffer[:6]
#                     self.buffer = self.buffer[6:]  # 移出已处理部分
#                     try:
#                         self.last_valid_force = float(force_str)
#                     except ValueError:
#                         pass
#             except Exception as e:
#                 print(f"Force Sensor Error: {e}")
#
#         return self.last_valid_force
#
#     def close(self):
#         if self.ser.is_open:
#             self.ser.close()


def main():
    print("正在初始化传感器...")

    # 1. 初始化 ADC 串口
    try:
        ser_adc = serial.Serial(ADC_PORT, ADC_BAUD, timeout=1)
        print(f"-> ADC连接成功 ({ADC_PORT})")
    except Exception as e:
        print(f"ADC 串口打开失败: {e}")
        return

    # # 2. 初始化力传感器
    # force_sensor = None
    # try:
    #     force_sensor = ForceSensorReader(FORCE_PORT, FORCE_BAUD)
    #     print(f"-> Force连接成功 ({FORCE_PORT})")
    # except Exception as e:
    #     print(f"力传感器打开失败: {e}")
    #     ser_adc.close()  # 失败则一起关闭
    #     return

    print(f"\n开始采集！数据保存至 {FILENAME}")
    print("数据格式: [CH1, CH2, CH3, CH4, Force]")
    print("按 Ctrl+C 停止...")

    # 3. 准备 CSV 文件
    f = open(FILENAME, 'w', newline='')
    writer = csv.writer(f)
    # # 写入 5 列的表头
    # writer.writerow(['CH1', 'CH2', 'CH3', 'CH4', 'Force'])
    # 【修改】只写入 4 列的表头
    writer.writerow(['CH1', 'CH2', 'CH3', 'CH4'])

    adc_buffer = ""  # ADC 数据的字符串缓存
    adc_row = []  # 暂存解析出的 ADC 数值

    try:
        while True:
            # === 步骤 A: 始终刷新力传感器读数 ===
            # # 无论 ADC 是否准备好，都要不断更新力传感器的“最新值”
            # current_force = force_sensor.read_latest()

            # === 步骤 B: 读取 ADC 数据 ===
            if ser_adc.in_waiting > 0:
                # 读取并拼接到缓存
                raw_chunk = ser_adc.read(ser_adc.in_waiting).decode('utf-8', errors='ignore')
                adc_buffer += raw_chunk

                # 如果缓存中有逗号，说明有数据分割点
                if ',' in adc_buffer:
                    parts = adc_buffer.split(',')

                    # 最后一个部分可能不完整，留回 buffer
                    valid_parts = parts[:-1]
                    adc_buffer = parts[-1]

                    for part in valid_parts:
                        # 清洗数据 (去 /，去空格)
                        val_str = part.replace('/', '').strip()
                        if val_str:
                            adc_row.append(val_str)

                        # === 步骤 C: 凑齐4个ADC数据 -> 写入一行 ===
                        if len(adc_row) == 4:
                            # 1. 把当前的力值加进去，变成第5列
                            # full_row = adc_row + [current_force]
                            full_row = adc_row

                            # 2. 写入 CSV
                            writer.writerow(full_row)

                            # 3. 打印 (可选)
                            print(f"记录: {full_row}")

                            # 4. 清空 ADC 行，准备下一组
                            adc_row = []

            # 短暂休眠，降低 CPU 占用
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n正在停止采集...")
    finally:
        f.close()
        ser_adc.close()
        # if force_sensor:
        #     force_sensor.close()
        print("所有设备已断开，文件已保存。")


if __name__ == "__main__":
    main()