# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from scrapy.exporters import CsvItemExporter

class A20225477202JiangxiantingPipeline:
    def __init__(self):
        self.MusicListFile = open("wangyiyun(3-7).csv", "wb+")  # 保存为csv格式
        self.MusicListExporter = CsvItemExporter(self.MusicListFile, encoding='utf8')
        self.MusicListExporter.start_exporting()

    def process_item(self, item, spider):
        if spider.name == 'musicinfo':
            self.MusicListExporter.export_item(item)
        return item

    def close_spider(self, spider):
        self.MusicListExporter.finish_exporting()
        self.MusicListFile.close()
