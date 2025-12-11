import time

import serial
import numpy as np

# 定义多项式系数
# p1 = 0.2513
# p2 = -0.7482
# p3 = 2.15
# # p4 = -0.06871
# p4 = -0.20777
p1 = 1.763
p2 = -0.1261

mapping = np.array([
    [0, 2, 4, 6, 1, 3, 5, 7],
    [14, 12, 10, 8, 9, 11, 13, 15],
    [22, 20, 18, 16, 30, 28, 26, 24],
    [23, 21, 19, 17, 31, 29, 27, 25],
    [38, 36, 34, 32, 46, 44, 42, 40],
    [39, 37, 35, 33, 47, 45, 43, 41]
])

ser1 = serial.Serial('COM3', 115200)
ser2 = serial.Serial('COM10', 115200)
ser3 = serial.Serial('COM9', 115200)

try:
    while True:
        # 读取三个串口的数据
        data1 = ser1.readline().decode().strip()
        data2 = ser2.readline().decode().strip()
        data3 = ser3.readline().decode().strip()

        # 转换为浮点数列表
        values1 = [float(x) for x in data1.split(',')]
        values2 = [float(x) for x in data2.split(',')]
        values3 = [float(x) for x in data3.split(',')]

        # 直接将三个列表组合成6×8的图像
        # 使用np.reshape将三个列表直接组合
        image = np.array(values1 + values2 + values3).reshape(6, 8).round(3)
        flat_image = image.flatten()
        # 公式: val(x) = p1*x^3 + p2*x^2 + p3*x + p4
        # 直接对numpy数组运算，会对每个元素生效
        # flat_image_transformed = (p1 * (flat_image ** 3) +
        #                           p2 * (flat_image ** 2) +
        #                           p3 * flat_image +
        #                           p4).round(3)
        flat_image_transformed = (p1 * flat_image + p2)
        # print(flat_image_transformed)
        test = flat_image[mapping].reshape(6, 8)
        print(test)
        tactile_image = flat_image_transformed[mapping].reshape(6, 8).round(1)
        F_sum = np.sum(flat_image_transformed[flat_image_transformed > 0.12])
        # print(F_sum)
        print(tactile_image)
        print("-" * 50)
        # time.sleep(0.5)

except KeyboardInterrupt:
    print("程序被用户中断")
except Exception as e:
    print(f"发生错误: {e}")
finally:
    ser1.close()
    ser2.close()
    ser3.close()
