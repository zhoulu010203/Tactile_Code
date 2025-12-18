import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


def train():
    # 1. 加载数据
    try:
        data = np.loadtxt("sensor_data.csv", delimiter=",")
    except OSError:
        print("错误：找不到 sensor_data.csv")
        return

    # 最后一列是标签，前面是特征
    X = data[:, :-1]
    y = data[:, -1]

    print(f"加载了 {len(X)} 条数据，特征维度 {X.shape[1]}")

    # 2. 划分训练集和测试集 (80% 训练, 20% 验证)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. 定义并训练模型
    # KNN 适合小样本。如果你想要 SVM，可以换成 from sklearn.svm import SVC; model = SVC()
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # 4. 验证效果
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n训练完成！测试集准确率: {acc * 100:.2f}%")

    # 打印详细分类报告
    target_names = ['Square', 'Rect', 'Circle', 'Triangle']
    # 注意：如果你的测试集中某些类别样本太少，这里可能会报错，可以忽略
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 5. 保存模型
    joblib.dump(model, 'shape_model.pkl')
    print("模型已保存为 'shape_model.pkl'")


if __name__ == '__main__':
    train()