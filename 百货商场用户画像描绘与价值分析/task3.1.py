# pip install wordcloud

import pandas as pd
import wordcloud
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('./temp/task3.csv', encoding='gbk')

# 数据筛选，取某个会员为例
samples = data[data['会员卡号']=='fffa7c0b']
del samples['此次消费的会员积分'],samples['平均每单金额']
# 数据处理
samples['年龄'] = samples['年龄'].astype(str)+'岁'
samples['入会时长'] = '入会' + samples['入会时长'].astype(str) + '天'
samples['消费次数'] = '消费' + samples['消费次数'].astype(str) + '次'
samples['消费金额'] = '消费' + samples['消费金额'].astype(int).astype(str) + '元'
samples['最后一次消费距今时长']  = '最后一次距今' + samples['最后一次消费距今时长'].astype(str) + '天'

# 转化，把特征标签全部合并为一句话，用空格分割
texts = ' '.join(list(samples.astype(str).values[0]))

# 绘制词云图
wc = wordcloud.WordCloud(background_color='white',
                    font_path='./data/msyh.ttc')
# 把字符串处理为绘图的形式，词+词频
words = wc.generate(texts)
plt.imshow(words)
plt.axis('off')
plt.show()
plt.savefig('./temp/用户画像.jpg')