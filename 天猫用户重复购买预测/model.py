import pandas as pd
from sklearn.preprocessing import StandardScaler
import lightgbm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,\
                             roc_auc_score,precision_score,f1_score

# 读取数据
train = pd.read_csv('./temp/train.csv')
test = pd.read_csv('./temp/test.csv')

# 样本不平衡处理
train['label'].value_counts()
# 计算1类样本shul
n = sum(train['label']==1)
# 分别抽取0，1类相同的样本数量
posdata = train[train['label']==0].sample(n, random_state=123)
negdata = train[train['label']==1]
# 两个数据合并
data = pd.concat([posdata, negdata] ,axis=0)
# data['label'].value_counts()

# 标准化
sc = StandardScaler() # 初始化标准差标准化的函数
df = sc.fit_transform(data.iloc[:,3:]) # 执行标准化
df = pd.DataFrame(df, columns=data.columns[3:])

# 模型训练
# 划分训练数据和验证数据
xtrain, xval, ytrain, yval = train_test_split(df, data['label'],
                                              test_size=0.2, random_state=2021)

# 模型构建,模型初始化
model_lgb = lightgbm.LGBMClassifier(
    n_estimators=1000,       # 训练次数
    max_depth=8,             # 树的深度
    num_leaves=25,           # 树的最大叶子树
    colsample_bytree=0.5,    # 特征采样率
    learning_rate=0.1,       # 学习速率
    metric='auc',            # 模型评价指标
    verbose=1,               # 使用整数 1 显示训练信息（0 为不显示）
    early_stopping_rounds=100 # 提前停止轮数
)

# 执行模型训练
model_lgb.fit(
    xtrain, ytrain,
    eval_metric='auc',
    eval_set=[(xtrain, ytrain), (xval, yval)]
)

# 模型评价
print(model_lgb.best_score_)
# 预测标签
pres = model_lgb.predict(xval)
# 预测是否重复购买的概率
presprob = model_lgb.predict_proba(xval)[:, 1]

print(confusion_matrix(yval, pres))
print(classification_report(yval, pres))
print('准确率：', precision_score(yval, pres))
print('F1值：', f1_score(yval, pres))
print('AUC值：', roc_auc_score(yval, presprob))

# 模型训练的过程
lightgbm.plot_metric(model_lgb.evals_result_, metric='auc' )
plt.show()

# 特征重要性
lightgbm.plot_importance(model_lgb)
plt.show()

# 模型预测
# 对test数据进行标准化
testdf = StandardScaler().fit_transform(test.iloc[:,2:])
testdf = pd.DataFrame(testdf, columns=test.columns[2:])

# 调用模型预测
test['prob'] = model_lgb.predict_proba(testdf)[:,1]

# 结果的导出
test[['user_id','merchant_id','prob']].to_csv('./temp/prediction.csv', index=None)
