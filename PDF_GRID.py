import numpy as np

def pdf_grid(GMM, x_start, x_end, y_start, y_end, step):
    """
    生成一个点阵并计算GMM的概率密度矩阵

    参数:
        GMM      : GaussianMixture2D 对象
        x_start  : x坐标起点
        x_end    : x坐标终点
        y_start  : y坐标起点
        y_end    : y坐标终点
        step     : 网格间距

    返回:
        xx, yy   : 网格点坐标矩阵 (meshgrid)
        pdf_mat  : 与坐标对应的概率密度矩阵
    """
    # 生成点阵
    x = np.arange(x_start, x_end + step, step)
    y = np.arange(y_start, y_end + step, step)
    xx, yy = np.meshgrid(x, y)

    # 转成点集 (N,2)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # 计算概率密度
    pdf_values = GMM.pdf(grid_points)

    # 恢复为二维矩阵
    pdf_mat = pdf_values.reshape(xx.shape)
    pdf_mat = np.flipud(pdf_mat)  # 上下翻转矩阵
    # return xx, yy, pdf_mat
    return pdf_mat