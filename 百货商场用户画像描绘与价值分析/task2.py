
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.available
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = 'SimHei'

# 读取数据
data = pd.read_csv('./temp/task1.csv', encoding='gbk',
                   parse_dates=['消费产生的时间','出生日期','登记时间'])

# 提取会员数据和非会员数据
vip = data[data['会员卡号'].notnull()]
nvip = data[data['会员卡号'].isnull()]
# 删除不在会员信息表出现的会员
vip_data = vip.dropna()

# 计算年龄
vip_data['出生日期'] = pd.to_datetime(vip_data['出生日期'], errors='coerce')   #我的
dates = vip_data['消费产生的时间'].max()
vip_data['年龄'] = dates.year - vip_data['出生日期'].dt.year
age = vip_data['年龄'].value_counts()
plt.bar(x=age.index, height=age)
plt.title('年龄分布')
plt.show()

# 年龄段分布
vip_data['年龄段'] =  pd.cut(vip_data['年龄'], bins=[ 0 ,18, 40, 65, 100],
       labels=['少年','青年','中年','老年'])
age_cut = vip_data['年龄段'].value_counts()
plt.pie(age_cut, labels=age_cut.index, autopct='%.2f%%')
plt.title('会员年龄段分析')
plt.show()

# 不同年龄段的消费情况
vip_money = vip_data[['消费金额','年龄']].groupby(by='年龄').sum()
plt.plot(vip_money)
plt.xlabel('年龄')
plt.ylabel('消费金额')
plt.title('不同年龄段的消费能力')
plt.show()

# 会员的性别占比
sex_count = vip_data['性别'].value_counts()
plt.pie(sex_count, labels=['女性','男性'], autopct='%.2f%%')
plt.title('会员的性别占比')
plt.show()

# 不同性别的消费占比
sex_money = vip_data[['性别','消费金额']].groupby(by='性别').sum()
plt.pie(sex_money['消费金额'], labels=['女性','男性'], autopct='%.2f%%')
plt.title('不同性别的消费占比')
plt.show()

# 会员和非会员订单占比
ordernum = [ len(vip_data['单据号'].unique()), len(nvip['单据号'].unique()) ]
plt.pie(ordernum, labels=['VIP','非VIP'], autopct='%.2f%%')
plt.title('VIP/非VIP订单占比')
plt.show()

# 会员和非会员消费金额占比
ordermoney = [vip_data['消费金额'].sum() , nvip['消费金额'].sum()]
plt.pie(ordermoney, labels=['VIP','非VIP'], autopct='%.2f%%' )
plt.title('VIP/非VIP的消费金额占比')
plt.show()

# 会员年消费趋势
vip_data['年份'] = vip_data['消费产生的时间'].dt.year # 提取年份
vipyear = vip_data.groupby(by='年份').agg({'消费金额':sum, '单据号':np.count_nonzero})
# 绘制散点气泡图
plt.scatter(x=range(4), y=vipyear['消费金额'], s=vipyear['单据号']/100)
plt.title('消费金额和订单数变化')
plt.xticks(range(4), ['2015年','2016年','2017年','2018年'])
plt.ylabel('消费总金额')
plt.show()

# 会员不同月份的消费人数
vip_data['月份'] = vip_data['消费产生的时间'].dt.month
# 按月统计人数
month_count = vip_data['月份'].value_counts().sort_index()
plt.plot(month_count)
plt.xlabel('月份')
plt.ylabel('消费人数')
plt.show()

# 会员不同年份月份的消费额情况
# 按年按月统计总消费金额
yeargro = vip_data.groupby(by=['年份','月份'])['消费金额'].sum()
yeargro = yeargro.reset_index()
# 格式转换，按年分组，按月份为行去转格式
yearpivot = pd.pivot_table(yeargro, index='月份', columns='年份', values='消费金额')
yearpivot.plot.line(title='每年每月的消费情况')
plt.show()

# 会员消费时间分析
vip_data['时刻'] = vip_data['消费产生的时间'].dt.hour
hour_count = vip_data['时刻'].value_counts().sort_index()
plt.plot(hour_count)
plt.xlabel('时刻')
plt.ylabel('消费人数')
plt.title('每个时刻的消费情况')
plt.show()

# 数据保存
vip_data = vip_data.sort_values(by='消费产生的时间')
vip_data.to_csv('./temp/task2.csv', index=None, encoding='gbk')
