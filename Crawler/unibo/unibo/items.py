# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class PagesList(scrapy.Item):
    pages = scrapy.Field()

class Url(scrapy.Item):
    id = scrapy.Field()
    url = scrapy.Field()
