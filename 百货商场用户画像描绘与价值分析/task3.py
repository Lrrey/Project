
import numpy as np
import pandas as pd

# 读取数据
data = pd.read_csv('./temp/task2.csv', encoding='gbk',
                   parse_dates=['消费产生的时间','出生日期','登记时间'])
# 性别
data['性别'] = data['性别'].apply(lambda x: '男' if x>0 else '女')

# 入会时间
dates = data['消费产生的时间'].max() # 窗口结束时间
data['入会时长'] = data['消费产生的时间'].apply(lambda x: (dates-x).days)

# 积分等级
jf = data.groupby(by='会员卡号').agg({'此次消费的会员积分':sum})

jf['积分等级'] = pd.cut(jf['此次消费的会员积分'], bins=[-1, 10000, 100000, 100000000],
       labels=['积分低等级','积分中等级','积分高等级'])

# 提取所有会员的基本信息，保存
user = pd.merge(jf, data[['会员卡号','年龄','年龄段','入会时长']], how='left', on='会员卡号')

# 消费次数
cost = data[['会员卡号','单据号']].drop_duplicates()
freq = cost.groupby(by='会员卡号').agg({'单据号':'count'})
freq.columns = ['消费次数']

# 消费频率
freq['消费频率'] = pd.cut(freq['消费次数'], bins=[-1, 6, 30, 10000],
                      labels=['低频消费','中频消费','高频消费'])
freq.reset_index(inplace=True)

# 合并freq
user = pd.merge(user, freq, how='left', on='会员卡号')

# 消费总金额
money = data.groupby(by='会员卡号').agg({'消费金额':'sum'})
money['消费水平'] = pd.cut(money['消费金额'], bins=[-1, 10000, 100000, 100000000],
       labels=['低消费水平','中等消费水平','高消费水平'])
money.reset_index(inplace=True)
# 合并money
user = pd.merge(user, money, how='left', on='会员卡号')

# 计算平均每单的金额
user['平均每单金额'] = user['消费金额']/user['消费次数']
user['价值属性'] = pd.cut(user['平均每单金额'], bins=[-1, 500, 3500, 1000000],
       labels=['单均价值低','单均价值一般','单均价值高'])
# del user['平均每单金额']

# 最近一次消费时间距今时长
dates = data['消费产生的时间'].max() # 窗口结束时间
recent = data.groupby(by='会员卡号').agg({'消费产生的时间':max})
recent['最后一次消费距今时长'] = recent['消费产生的时间'].apply(lambda x: (dates-x).days)
recent.reset_index(inplace=True) # 重置索引
del recent['消费产生的时间']
# 合并recent
user = pd.merge(user, recent, how='left', on='会员卡号')

# 购物偏好
gz = data.groupby(by=['会员卡号','柜组名称']).agg({'柜组名称':'count'})
gz.columns = ['柜组数量']
gz.reset_index(inplace=True)

# 每个会员取数量最多的柜组
gzmax = gz.groupby(by='会员卡号').agg({'柜组数量':max})
gzmax.reset_index(inplace=True)
# 通过合并，找到数量最大的柜组名称
usergz = pd.merge(gzmax, gz, how='inner', on=['会员卡号','柜组数量'])
# 去重，重复的名称，只保留一个
usergz.drop_duplicates(subset='会员卡号', inplace=True)

# 合并usergz
user = pd.merge(user, usergz[['会员卡号','柜组名称']], how='left', on='会员卡号')

# 去重
user.drop_duplicates(subset='会员卡号', inplace=True)

# 数据写出
user.to_csv('./temp/task3.csv', index=None, encoding='gbk')