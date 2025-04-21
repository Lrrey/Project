# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class A20225477202JiangxiantingItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    SongsListID = scrapy.Field()  # 歌单id号
    SongListName = scrapy.Field()  # 歌单名
    AmountOfPlay = scrapy.Field()  # 播放量
    Labels = scrapy.Field()  # 标签名
    Url = scrapy.Field()  # 歌单域名，为下一次详细爬取留备份
    Collection = scrapy.Field()  # 歌单收藏量
    Forwarding = scrapy.Field()  # 转发量
    Comment = scrapy.Field()  # 评论量
    NumberOfSongs = scrapy.Field()  # 歌曲数量
    CreationDate = scrapy.Field()  # 歌单创建日期
    AuthorID = scrapy.Field()  # 作者id
