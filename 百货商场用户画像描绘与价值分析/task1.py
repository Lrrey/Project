
import pandas as pd

# 读取数据
userinfo = pd.read_excel('./data/userinfo.xlsx')
sales = pd.read_csv('./data/sales.csv')

# 数据探索
userinfo.info()
userinfo.describe()
for i in range(userinfo.shape[1]):
    print(userinfo.iloc[:,i].value_counts())

# 日期处理
userinfo['出生日期'] = pd.to_datetime(userinfo['出生日期'], errors='coerce')
userinfo['登记时间'] = pd.to_datetime(userinfo['登记时间'], errors='coerce')
# 空值
userinfo.dropna(inplace=True)
userinfo = userinfo.dropna(subset=['出生日期', '登记时间'])

# 删除登记时间大于出生时间的数据
userinfo = userinfo[ userinfo['出生日期']<userinfo['登记时间'] ]
# 删除大于80岁的人的记录
userinfo = userinfo[ userinfo['出生日期']>pd.to_datetime('1938-01-01') ]
# 重复数据
userinfo.drop_duplicates('会员卡号', keep='first', inplace=True)

# 流水表 处理
sales.info()
sales.describe()

# 日期转换
sales['消费产生的时间'] = pd.to_datetime(sales['消费产生的时间'] )
# 删除销售数量为负的数据
sales = sales[sales['销售数量']>0]
# 删除销售金额为负 或者为0
sales = sales[sales['消费金额']>0]
userinfo['会员卡号'] = userinfo['会员卡号'].astype(str)
sales['会员卡号'] = sales['会员卡号'].astype(str)

# 合并两个表
data = pd.merge(sales, userinfo, how='outer', on='会员卡号')

# 数据写出
data.to_csv('./temp/task1.csv', index=None, encoding='gbk')


