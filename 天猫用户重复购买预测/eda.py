
import  pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
train = pd.read_csv('./data/train_format1.csv')
test = pd.read_csv('./data/test_format1.csv', usecols=['user_id','merchant_id'])
user_info = pd.read_csv('./data/user_info_format1.csv')
# 数据量太大，直接读取可能会导致内存溢出
# user_log = pd.read_csv('./data/user_log_format1.csv')
def read_local(file_name, chunk_size=500000):
    # 读取为可迭代的TextFileReader对象
    reader = pd.read_csv(file_name, iterator=True, header=0)
    chunks = []
    loop = True
    while loop:
        try:
            # 每一次按500000行获取数据
            chunk = reader.get_chunk(chunk_size)
            chunks.append(chunk)
        except:
            loop = False
            print('数据获取完毕！')
    # 把列表的数据合并为数据框
    df = pd.concat(chunks, ignore_index=True)
    return df
# 通过函数，分块读取
user_log = read_local(file_name='./data/user_log_format1.csv', chunk_size=500000)

# 数据探索和预处理
test.isna().sum()
train.isna().sum()
user_info.isna().sum()
user_log.isna().sum()

# 缺失值填补
user_info['age_range'].fillna(0, inplace=True)
user_info['gender'].fillna(2, inplace=True)
user_log['brand_id'].fillna(0, inplace=True)

# 去重
# test.duplicated().sum()
# test.drop_duplicates(inplace=True)
user_log.drop_duplicates(inplace=True)

# 重命名
user_log.rename(columns={'seller_id':'merchant_id'}, inplace=True)

# 数据保存
user_info.to_csv('./temp/userinfo.csv', index=None)
user_log.to_csv('./temp/userlog.csv', index=None)
del user_log

# 正负样本统计
label = train['label'].value_counts()
# 绘制饼图
plt.pie(label, labels=label.index, autopct='%.2f%%', explode=[0, 0.3])
plt.title('0 VS 1')
plt.show()

# 性别和复购的关系
# 表连接
t_userinfo = train.merge(user_info, on='user_id', how='inner')
# 绘图
sns.countplot(x='gender', hue='label', data=t_userinfo)
plt.title('gender & label')
plt.show()

# 年龄和复购的关系
sns.countplot(x='age_range', hue='label', data=t_userinfo)
plt.title('age & label')
plt.show()
