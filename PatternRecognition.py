import time

def count_patterns(binary_image):
    """
    在二值图像矩阵中统计特定3x3模式的数量

    参数:
    binary_image: 二维列表或数组，包含0和1

    返回:
    int: 找到的模式总数
    """

    # 定义要查找的两种模式
    pattern1 = [
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1]
    ]

    pattern2 = [
        [0, 1, 1],
        [1, 1, 1],
        [1, 1, 0]
    ]

    # 获取图像尺寸
    rows = len(binary_image)
    cols = len(binary_image[0])

    # 如果图像小于3x3，直接返回0
    if rows < 3 or cols < 3:
        return 0

    count = 0

    # 遍历所有可能的3x3子矩阵
    for i in range(rows - 2):
        for j in range(cols - 2):
            # 提取3x3子矩阵
            submatrix = [
                [binary_image[i][j], binary_image[i][j + 1], binary_image[i][j + 2]],
                [binary_image[i + 1][j], binary_image[i + 1][j + 1], binary_image[i + 1][j + 2]],
                [binary_image[i + 2][j], binary_image[i + 2][j + 1], binary_image[i + 2][j + 2]]
            ]

            # 检查是否匹配第一种模式
            match_pattern1 = True
            for x in range(3):
                for y in range(3):
                    if submatrix[x][y] != pattern1[x][y]:
                        match_pattern1 = False
                        break
                if not match_pattern1:
                    break

            # 如果匹配第一种模式，计数加1
            if match_pattern1:
                count += 1
                continue  # 一个位置只能匹配一种模式，所以跳过第二种检查

            # 检查是否匹配第二种模式
            match_pattern2 = True
            for x in range(3):
                for y in range(3):
                    if submatrix[x][y] != pattern2[x][y]:
                        match_pattern2 = False
                        break
                if not match_pattern2:
                    break

            # 如果匹配第二种模式，计数加1
            if match_pattern2:
                count += 1

    return count


# # 测试您的示例
# binary_image = [
#     [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 0, 0, 0, 0, 0],
#     [0, 1, 1, 1, 0, 0, 0, 0, 0],
#     [0, 1, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 0, 0, 0, 0, 0, 0]
# ]
# start_time = time.time()
# result = count_patterns(binary_image)
# end_time = time.time()
# during_time = end_time - start_time
# print(during_time)
# print(result)  # 输出: 1