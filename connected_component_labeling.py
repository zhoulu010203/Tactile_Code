# connected_component_labeling.py

import numpy as np
from Find_Union import UnionFind


def connected_component_labeling(binary_image):
    """
    使用并查集数据结构进行连通区域标记（四连通，只判断上、左）

    参数:
        binary_image: 二值图像，值为0或1

    返回:
        labels: 标记后的图像，背景为0，连通区域用正整数标记
        num_components: 连通区域的数量
    """
    rows, cols = binary_image.shape
    labels = np.zeros_like(binary_image, dtype=int)
    uf = UnionFind()
    current_label = 1

    # 第一遍扫描：临时标记
    for i in range(rows):
        for j in range(cols):
            if binary_image[i, j] == 1:
                neighbors = []
                # 只检查上方和左方（四连通的一半）
                for dx, dy in [(-1, 0), (0, -1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if labels[ni, nj] > 0:
                            neighbors.append(labels[ni, nj])

                if not neighbors:
                    # 如果没有邻居，分配新标签
                    labels[i, j] = current_label
                    uf.parent[current_label] = current_label
                    current_label += 1
                else:
                    # 获取邻居标签的根
                    root_neighbors = []
                    for n in neighbors:
                        root_n = uf.find(n)
                        root_neighbors.append(root_n)

                    min_root = min(root_neighbors)
                    labels[i, j] = min_root

                    # 合并等价标签
                    for root_val in set(root_neighbors):
                        if root_val != min_root:
                            uf.union(min_root, root_val)

    # 第二遍扫描：统一标签
    unique_roots = set()
    for i in range(rows):
        for j in range(cols):
            if labels[i, j] != 0:
                root = uf.find(labels[i, j])
                labels[i, j] = root
                unique_roots.add(root)

    num_components = len(unique_roots)
    return labels, num_components


def visualize_labeled_regions(binary_image, labels):
    """
    可视化标记结果

    参数:
        binary_image: 原始二值图像
        labels: 标记后的图像
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 显示原始二值图像
    ax1.imshow(binary_image, cmap='gray')
    ax1.set_title('Original Binary Image')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # 显示标记结果
    im = ax2.imshow(labels, cmap='nipy_spectral')
    ax2.set_title('Connected Components Labeling')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im, ax=ax2, label='Component Label')

    plt.tight_layout()
    plt.show()

    print(f"检测到 {len(np.unique(labels)) - 1} 个连通区域")  # 减去背景标签0