import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
import os
from urllib.parse import urlparse

class UniboCrawlerSpider(CrawlSpider):
    name = "unibo_crawler"
    jina_prefix = "https://r.jina.ai/"

    # allowed_domains = ["unibo.it", "r.jina.ai"]
    # start_urls = ["https://www.unibo.it/en"]

    allowed_domains = ["corsi.unibo.it", "r.jina.ai"]
    start_urls = ["https://corsi.unibo.it/laurea/IngegneriaScienzeInformatiche"]

    PRINT_ONLY_URL = False
    FOLDER_CHARACTER_LIMIT = 50

    def __init__(self, *args, **kwargs):
        super(UniboCrawlerSpider, self).__init__(*args, **kwargs)
        self.i = 0

    rules = (
        # Rule(LinkExtractor(restrict_css="a"), callback="download_jina", follow=True),
        Rule(LinkExtractor(restrict_css="a", deny_domains=["corsi.unibo.it"]), callback="download_jina", follow=False),
        Rule(LinkExtractor(restrict_css="a", allow_domains=["corsi.unibo.it"]), callback="download_jina", follow=True),
    )

    def cleanMarkdown(self, text):
        marker = "==============="

        start_index = text.find(marker)
        end_index = text.find(marker, start_index + len(marker))
        print(start_index, end_index)
        if start_index == -1 or end_index == -1:
            return text
        
        cleaned_text = text[:start_index] + text[end_index:]
        
        return cleaned_text

    def download_jina(self, response):
        full_link = self.jina_prefix + response.url
        headers = {
            "x-respond-with":"markdown"
        }
        yield scrapy.Request(full_link,
                            callback=self.parse_item_from_request,
                            headers=headers,
                            errback=self.handle_error)

    def parse_item_from_request(self, response):
        self.i+=1
        original_url = response.url.replace("https://r.jina.ai/", "")
        path = "pagine2"
        if self.PRINT_ONLY_URL:
            filename = "links.txt"
            path = os.path.join(path, filename)
        else:
            url = urlparse(original_url)
            path_segments = [segment[:self.FOLDER_CHARACTER_LIMIT] for segment in url.path.split('/')] #Limito a 50 caratteri i nomi delle directory
            filename = str(self.i) + ".txt"
            path = os.path.join(path, *path_segments, filename) 

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, ("a" if self.PRINT_ONLY_URL else "w"), encoding="utf-8") as f:
            f.write(original_url + "\n\n") #Lascio traccia del link della pagina scaricata
            if not self.PRINT_ONLY_URL:
                f.write(self.cleanMarkdown(response.text))
            
        yield {
                'id': self.i,
                "link": original_url
            }

    def handle_error(self, failure):
        self.logger.error(repr(failure))