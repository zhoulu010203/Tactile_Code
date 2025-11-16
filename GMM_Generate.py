import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

'''
使用时：from GMM_generate import GaussianMixture2D   # 从另一个文件导入类
例：
初始化高斯混合模型
gmm = GaussianMixture2D(means, covs, weights)
# 使用模型计算pdf
X = np.array([[0,0], [1,1], [2,2]])
print(gmm.pdf(X))

# 1. 打印模型参数
print(GMM)
# 2. 查看几个点的概率密度
X = np.array([[0,0], [1,1], [2,2], [3,3]])
print(GMM.pdf(X))
# 3. 直接绘制分布图
GMM.show(xlim=(-2, 6), ylim=(-2, 6), resolution=200)
'''


class GaussianMixture2D:
    def __init__(self, means, covs, weights):
        self.weights = np.array(weights) / np.sum(weights)
        self.components = [
            (w, multivariate_normal(mean=m, cov=c))
            for m, c, w in zip(means, covs, self.weights)
        ]

    def pdf(self, x):
        total = np.zeros(x.shape[0])
        for w, rv in self.components:
            total += w * rv.pdf(x)
        return total

    def __repr__(self):
        return f"GaussianMixture2D(weights={self.weights}, " \
               f"means={[rv.mean for _, rv in self.components]}, " \
               f"covs={[rv.cov for _, rv in self.components]})"

    def show(self, xlim=(-5, 5), ylim=(-5, 5), resolution=100):
        """绘制2D GMM 的概率密度等高线图"""
        x, y = np.meshgrid(np.linspace(xlim[0], xlim[1], resolution),
                           np.linspace(ylim[0], ylim[1], resolution))
        grid = np.column_stack([x.ravel(), y.ravel()])
        z = self.pdf(grid).reshape(x.shape)

        plt.figure(figsize=(6, 5))
        plt.contourf(x, y, z, levels=50, cmap="viridis")
        plt.colorbar(label="PDF value")
        plt.title("2D Gaussian Mixture Model")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
