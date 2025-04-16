
import pandas as pd
import numpy as np
# 读取大文件的函数
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

# 读取数据
train = pd.read_csv('./data/train_format1.csv')
test = pd.read_csv('./data/test_format1.csv', usecols=['user_id','merchant_id'])
user_info = pd.read_csv('./temp/userinfo.csv')
user_log = read_local(file_name='./temp/userlog.csv')

# 把要用到的用户和商家筛选出来
matrix = pd.concat([train, test], axis=0)
# userlog数据太大，也要做筛选
user_log = pd.merge(user_log, matrix[['user_id','merchant_id']],
                    on=['user_id','merchant_id'], how='inner')
# 修改数据格式
# user_log['brand_id'].fillna(0, inplace=True)
user_log['brand_id'] = user_log['brand_id'].astype('int32')

user_log['time_stamp'] = pd.to_datetime(user_log['time_stamp'], format='%m%d')

# matrix添加年龄和性别特征
matrix = pd.merge(matrix, user_info, on='user_id', how='left')
# 数据格式转换
matrix['label'] = matrix['label'].astype(str)
matrix['age_range'] = matrix['age_range'].astype('int8')
matrix['gender'].fillna(2, inplace=True)
matrix['gender'] = matrix['gender'].astype('int8')

# 1.用户相关
# 1.1用户在平台的总交互次数；（用户在平台的所有操作记录都认为是交互）
# user_log['user_id'].value_counts()
# 以id分组
groups = user_log.groupby(by='user_id')
# 计算user_id的数量
temp = groups.size().reset_index()
temp.rename(columns={0:'user_num'}, inplace=True)
# 特征合并
matrix = pd.merge(matrix, temp, on='user_id', how='left')

# 1.2用户最近一次购买距离第一次的时长；
temp = groups['time_stamp'].agg([('F_time','min'),('L_time','max')])
temp.reset_index(inplace=True)
temp['user_days'] = (temp['L_time']-temp['F_time']).dt.days
# 特征拼接
matrix = pd.merge(matrix, temp[['user_id','user_days']], on='user_id', how='left')

# 1.3 用户对不同【商品、品类、品牌、商家】的交互次数；用户交互过多少个【商品、品类、品牌、商家】
merfeature = ['user_id', 'item_id', 'cat_id',  'brand_id', 'merchant_id']
# 函数，计算去重后的数量
xsum = lambda x: len(x.unique())
inmerchant = user_log[merfeature].groupby(by='user_id').agg(xsum)
inmerchant.reset_index(inplace=True)
inmerchant.columns = ['user_id','item_num','cat_num','brand_num','merchant_num']
# 特征拼接
matrix = pd.merge(matrix, inmerchant, on='user_id', how='left')

# 1.4 用户进行【点击、加购物车、购买、收藏】的次数；
temp = pd.pivot_table(user_log[['user_id','action_type']],
               index='user_id', columns='action_type',aggfunc=np.count_nonzero, fill_value=0)
temp.reset_index(inplace=True)
temp.columns = ['user_id','uclick_num','uadd_num','ubuy_num','usave_num']
# 特征合并
matrix = pd.merge(matrix, temp, on='user_id', how='left')

# 1.5 用户的购买率=购买次数/点击次数
matrix['user_buy_rate'] = matrix['ubuy_num']/(matrix['uclick_num']+1)

# 2.商家相关
# 2.1 商家被交互的数量；
groups = user_log.groupby(by='merchant_id')
temp = groups.size().reset_index()
temp.rename(columns={0:'merchant_benum'}, inplace=True)
# 特征拼接
matrix = pd.merge(matrix, temp, on='merchant_id', how='left')

# 2.2 商家被交互的【商品、品类、品牌】的数量；
# 2.2 与商家交互的用户数量；
temp = groups[['user_id', 'item_id','cat_id','brand_id']].agg(xsum)
temp.reset_index(inplace=True)
temp.columns = ['merchant_id','merchant_user','merchant_item','merchant_cat','merchant_brand']
# 特征拼接
matrix = pd.merge(matrix, temp, on='merchant_id', how='left')

# 2.3 商家【被点击、被加购物车、被购买、被收藏】的次数；
temp = pd.pivot_table(user_log[['merchant_id','action_type']],
                      index='merchant_id',columns='action_type',
                      aggfunc=np.count_nonzero, fill_value=0)
temp.reset_index(inplace=True)
temp.columns = ['merchant_id','mclick_num','mbuy_num','madd_num','msave_num']
# 特征拼接
matrix = pd.merge(matrix, temp, on='merchant_id', how='left')

# 2.4 商家的复购次数；
temp = pd.DataFrame(train.loc[train['label']==1, 'merchant_id'].value_counts())
temp.reset_index(inplace=True)
temp.columns = ['merchant_id','merchant_rebuy']
# 特征拼接
matrix = pd.merge(matrix, temp, on='merchant_id', how='left')

# 2.5 商家的被购买率
matrix['merchant_buy_rate'] = matrix['mbuy_num']/(matrix['mclick_num']+1)

# 3. 用户+商家同时交互的特征
# 3.1 用户在商家的交互次数；
groups = user_log.groupby(by=['user_id','merchant_id'])
temp = groups.size().reset_index()
temp.rename(columns={0:'user_merchant'}, inplace=True)
# 特征拼接
matrix = pd.merge(matrix, temp, on=['user_id','merchant_id'], how='left')

# 3.2 用户对商家的【商品、品类、品牌】的交互次数(唯一数量)；
temp = groups[['item_id','cat_id','brand_id']].agg(xsum)
temp.columns = ['user_merchant_item','user_merchant_cat','user_merchant_brand']
temp.reset_index(inplace=True)
# 特征拼接
matrix = pd.merge(matrix, temp, on=['user_id','merchant_id'], how='left')

# 3.3 用户对商家【点击、加购物车、购买、收藏】的次数；
temp = pd.pivot_table(user_log[['user_id','merchant_id','action_type']],
                      index=['user_id','merchant_id'], columns='action_type',
                      aggfunc=np.count_nonzero, fill_value=0)
temp.columns = ['user_merchant_click','user_merchant_add','user_merchant_buy','user_merchant_save']
temp.reset_index(inplace=True)
# 特征拼接
matrix = pd.merge(matrix, temp, on=['user_id','merchant_id'], how='left')

# 3.4 不同用户在不同商家购买率=购买次数/点击次数
temp['user_merchant_buy_rate'] = temp['user_merchant_buy']/(temp['user_merchant_click']+1)
# 特征拼接
matrix = pd.merge(matrix, temp[['user_id','merchant_id','user_merchant_buy_rate']],
                  on=['user_id','merchant_id'], how='left')

# 3.5 用户在该商家的最近一次购买距离第一次的时长；
temp = groups['time_stamp'].agg([('first','min'),('last','max')])
temp['user_merchant_days'] = (temp['last']-temp['first']).dt.days
temp.reset_index(inplace=True)
del temp['first'], temp['last']
# 特征拼接
matrix = pd.merge(matrix, temp, on=['user_id','merchant_id'], how='left')

# 离散型特征age_range，gender特征处理
temp = pd.get_dummies(matrix['age_range'], prefix='age')
# 特征拼接
matrix = pd.concat([matrix, temp], axis=1)
del matrix['age_range']

# 离散型特征gender特征处理
temp = pd.get_dummies(matrix['gender'], prefix='g')
# 特征拼接
matrix = pd.concat([matrix, temp], axis=1)
del matrix['gender']

# 分割训练集和测试集
matrix.fillna(0, inplace=True) # 缺失值填补
train_data =  matrix[matrix['label']!='nan']
test_data = matrix[matrix['label']=='nan']
del test_data['label']

# 数据写出
train_data.to_csv('./temp/train.csv', index=None)
test_data.to_csv('./temp/test.csv', index=None)