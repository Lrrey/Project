import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 读取Excel文件
file_path = '20225477202jiangxianting(3-7).csv'  # 替换为你的文件路径
df = pd.read_csv(file_path)

# 获取最后一列的所有文本数据，并去除NaN值
last_column_name = df.columns[-3]  # 获取最后一列的列名
last_column_data = df[last_column_name].dropna().astype(str)  # 去除NaN值并转换为字符串类型

# 将所有文本合并成一个长字符串
text = ' '.join(last_column_data)  # 直接使用去除NaN值后的数据

# 指定中文字体路径
font_path = 'simhei.ttf'  # 替换为你的中文字体文件路径

# 创建词云对象，指定字体路径
wordcloud = WordCloud(font_path=font_path, width=800, height=400, background_color='white').generate(text)

# 显示词云图像
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # 不显示坐标轴
plt.show()