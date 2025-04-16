from Radar import findk, radarplot
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('./temp/task3.csv', encoding='gbk')

# 筛选特征并处理缺失值
vip_feature = data[['消费次数', '最后一次消费距今时长', '消费金额', '入会时长']].dropna()

# 数据标准化
sc = StandardScaler()
feature = sc.fit_transform(vip_feature)

# 使用BIC选择最优聚类数（替代原findk函数）
bic = []
n_components_range = range(2, 11)
for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(feature)
    bic.append(gmm.bic(feature))

# 绘制BIC曲线
plt.plot(n_components_range, bic)
plt.xlabel('Number of Components')
plt.ylabel('BIC Score')
plt.title('BIC for GMM')
plt.grid()
plt.show()

# 选择BIC最低的聚类数
best_n = 7
model = GaussianMixture(n_components=best_n, random_state=0)
model.fit(feature)

# 获取聚类标签（硬聚类）
labels = model.predict(feature)

# 计算聚类中心（取各簇均值）
centers = model.means_

# 添加标签到原始数据
data = data.join(pd.Series(labels, index=vip_feature.index, name='客户类别'))

# 绘制雷达图
radarplot(centers, vip_feature.columns)
plt.show()

# 输出聚类结果
data.to_csv('./temp/task4_gmm.csv', index=None, encoding='gbk')