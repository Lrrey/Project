# 导入数据处理库 pandas，用于数据的读取、处理和分析
import pandas as pd
# 导入模型评估工具，用于评估模型的性能
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score
# 导入随机森林分类器，用于构建分类模型
from sklearn.ensemble import RandomForestClassifier
# 导入数据标准化工具（可选，树模型可不标准化），用于对数据进行标准化处理
from sklearn.preprocessing import StandardScaler
# 导入训练集/验证集划分工具，用于将数据集划分为训练集和验证集
from sklearn.model_selection import train_test_split
# 导入绘图库，用于绘制特征重要性可视化图表
import matplotlib.pyplot as plt

# 读取预处理后的训练集和测试集数据，文件路径为 './temp/train.csv' 和 './temp/test.csv'
train = pd.read_csv('./temp/train.csv')
test = pd.read_csv('./temp/test.csv')

# 样本不平衡处理（与原项目一致）
# 统计训练集中标签为 1 的样本数量
n = sum(train['label'] == 1)
# 从训练集中标签为 0 的样本中随机抽取 n 个样本，随机种子为 123
posdata = train[train['label'] == 0].sample(n, random_state=123)
# 选取训练集中标签为 1 的样本
negdata = train[train['label'] == 1]
# 将抽取的标签为 0 的样本和标签为 1 的样本合并为一个新的数据集
data = pd.concat([posdata, negdata], axis=0)

# 树模型对特征尺度不敏感，可跳过标准化（若需保留，取消注释）
# sc = StandardScaler()  # 初始化数据标准化工具
# df = sc.fit_transform(data.iloc[:, 3:])  # 对数据的第 3 列及以后的列进行标准化处理
# df = pd.DataFrame(df, columns=data.columns[3:])  # 将标准化后的数据转换为 DataFrame 格式
# 直接使用原始特征，选取数据的第 3 列及以后的列作为特征
df = data.iloc[:, 3:]

# 划分训练集和验证集
# 将特征数据 df 和标签数据 data['label'] 按照 8:2 的比例划分为训练集和验证集，随机种子为 2021
xtrain, xval, ytrain, yval = train_test_split(
    df, data['label'], test_size=0.2, random_state=2021
)

# 初始化随机森林分类器
model_rf = RandomForestClassifier(
    n_estimators=200,        # 树的数量，即随机森林中包含的决策树的数量
    max_depth=10,            # 树的最大深度，限制决策树的生长深度
    class_weight='balanced', # 自动调整类别权重，缓解样本不均衡问题
    random_state=2021,
    n_jobs=-1                # 使用全部 CPU 核心加速训练
)

# 训练模型，使用训练集的特征数据 xtrain 和标签数据 ytrain 对随机森林分类器进行训练
model_rf.fit(xtrain, ytrain)

# 模型评价
# 使用训练好的模型对验证集的特征数据 xval 进行预测，得到预测标签
pres = model_rf.predict(xval)
# 使用训练好的模型对验证集的特征数据 xval 进行预测，得到预测概率
presprob = model_rf.predict_proba(xval)[:, 1]

# 打印混淆矩阵，用于评估模型的分类性能
print("混淆矩阵：")
print(confusion_matrix(yval, pres))
# 打印分类报告，包含精确率、召回率、F1 值等评估指标
print("分类报告：")
print(classification_report(yval, pres))
# 打印 AUC 值，用于评估模型的排序能力
print('AUC 值：', roc_auc_score(yval, presprob))
# 打印 F1 值，用于综合评估模型的精确率和召回率
print('F1 值：', f1_score(yval, pres))

# 特征重要性可视化
# 将模型的特征重要性得分与特征名称关联起来，并按重要性得分降序排序
feature_importance = pd.Series(
    model_rf.feature_importances_,
    index=df.columns
).sort_values(ascending=False)
# 绘制特征重要性柱状图，图表大小为 12x6
feature_importance.plot(kind='bar', figsize=(12, 6))
# 设置图表标题
plt.title('Feature Importance (Random Forest)')
# 显示图表
plt.show()

# 预测测试集
# 选取测试集的第 2 列及以后的列作为特征
test_features = test.iloc[:, 2:]
# 使用训练好的模型对测试集的特征数据进行预测，得到预测概率，并将其添加到测试集中
test['prob'] = model_rf.predict_proba(test_features)[:, 1]
# 将测试集中的 user_id、merchant_id 和预测概率保存到 './temp/prediction_rf.csv' 文件中，不保存行索引
test[['user_id', 'merchant_id', 'prob']].to_csv(
    './temp/prediction_rf.csv', index=None
)
