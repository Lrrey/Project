# -*- coding: utf-8 -*-
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
import matplotlib.pyplot as plt
import  numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# K 值寻优
def findk(feature):
    # K值寻优
    CHScore = []
    for i in range(2, 11):
        # 模型初始化
        kmodel = KMeans(n_clusters=i, random_state=123)
        # 训练
        kmodel.fit(feature)
        # 计算CH得分
        cs = calinski_harabasz_score(feature, kmodel.labels_)
        CHScore.append(cs)
        del kmodel, cs
    plt.plot(range(2, 11), CHScore)
    plt.grid()
    plt.xlabel('k的个数')
    plt.ylabel('CH评价指标')
    plt.show()

# 绘制雷达图，传入参数1：model_center(聚类中心)，参数2：label(特征名字)


def radarplot(model_center, labels, title="雷达图"):
    """
    绘制雷达图
    :param model_center: 聚类中心的特征值 (numpy array)
    :param labels: 特征名称 (list of str)
    :param title: 图表标题 (str)
    """
    # 特征个数
    n = len(labels)

    # 将特征名称和数据封闭，确保雷达图是一个完整的圆形
    labels = np.concatenate((labels, [labels[0]]))
    model_center = np.concatenate((model_center, model_center[:, [0]]), axis=1)

    # 设置雷达图的角度
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    # 创建画布
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, polar=True)

    # 设置雷达图的标签和角度
    ax.set_thetagrids(angles * 180 / np.pi, labels, fontsize=14)

    # 设置雷达图的范围
    max_val = np.max(model_center)
    min_val = np.min(model_center)
    ax.set_ylim(min_val, max_val)

    # 设置网格线
    ax.grid(True, linestyle='--', alpha=0.6)

    # 定义颜色和标记
    colors = ['r', 'g', 'b', 'm', 'y', 'c', 'orange', 'purple', 'brown']
    markers = ['o', 's', 'x', '*', 'd', '_', '.', '+', '|']

    # 绘制每个聚类中心的雷达图
    for i in range(len(model_center)):
        values = model_center[i]
        ax.plot(angles, values, color=colors[i % len(colors)], marker=markers[i % len(markers)], linewidth=2,
                markersize=8, label=f'客户群{i}')
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.25)

    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)

    # 添加标题
    ax.set_title(title, fontsize=16, pad=20)

    # 显示图表
    plt.show()




