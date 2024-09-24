import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
import os
from urllib.parse import urlparse

import re

#Jinai SLM
from vllm import SamplingParams
from vllm import LLM

SCRIPT_PATTERN = r'<[ ]*script.*?\/[ ]*script[ ]*>' # (REMOVE <SCRIPT> to </script> and variations)

STYLE_PATTERN = r'<[ ]*style.*?\/[ ]*style[ ]*>' # (REMOVE HTML <STYLE> to </style> and variations)

META_PATTERN = r'<[ ]*meta.*?>' # (REMOVE HTML <META> to </meta> and variations)

COMMENT_PATTERN = r'<[ ]*!--.*?--[ ]*>' # (REMOVE HTML COMMENTS <!-- to --> and variations)

LINK_PATTERN = r'<[ ]*link.*?>'  # (REMOVE HTML LINK <LINK> to </link> and variations), mach any char zero or more times

BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>' # (REPLACE base64 images)

SVG_PATTERN = r'(<svg[^>]*>)(.*?)(<\/svg>)' # (REPLACE <svg> to </svg> and variations)

def create_prompt(text:str, tokenizer) -> str:
   messages = [
    {
        "role": "user",
        "content": text
    },
   ]
   return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
   )

def get_sampling_params():
    top_k = 1 # @param {type:"integer"}
    temperature = 0 # @param {type:"slider", min:0, max:1, step:0.1}
    repetition_penalty = 1.08 # @param {type:"number"}
    presence_penalty = 0.25 # @param {type:"slider", min:0, max:1, step:0.1}
    top_k = 1 # @param {type:"integer"}
    max_tokens = 4096 # @param {type:"integer"}
    sampling_params = SamplingParams(temperature=temperature, top_k=top_k, presence_penalty=presence_penalty, repetition_penalty=repetition_penalty, max_tokens=max_tokens)
    return sampling_params

def replace_svg(html: str, new_content: str = "this is a placeholder") -> str:
    return re.sub(
        SVG_PATTERN,
        lambda match: f"{match.group(1)}{new_content}{match.group(3)}",
        html,
        flags=re.DOTALL,
    )

def replace_base64_images(html: str, new_image_src: str = "#") -> str:
    return re.sub(BASE64_IMG_PATTERN, f'<img src="{new_image_src}"/>', html)

def has_base64_images(text: str) -> bool:
    base64_content_pattern = r'data:image/[^;]+;base64,[^"]+'
    return bool(re.search(base64_content_pattern, text, flags=re.DOTALL))

def has_svg_components(text: str) -> bool:
    return bool(re.search(SVG_PATTERN, text, flags=re.DOTALL))

def clean_html(html: str, clean_svg: bool = False, clean_base64: bool = False):
    html = re.sub(SCRIPT_PATTERN, '', html, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))
    html = re.sub(STYLE_PATTERN, '', html, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))
    html = re.sub(META_PATTERN, '', html, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))
    html = re.sub(COMMENT_PATTERN, '', html, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))
    html = re.sub(LINK_PATTERN, '', html, flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL))

    if clean_svg:
        html = replace_svg(html)

    if clean_base64:
        html = replace_base64_images(html)

    return html

class UniboSimpleCrawlerSpider(CrawlSpider):
    name = 'unibo_simple_crawler'
    jina_prefix = "https://r.jina.ai/"

    FORBIDDEN_PATH_CHARS = ["<", ">", ":", "\"", "/", "\\", "|", "?", "*"]
    allowed_domains = ['corsi.unibo.it', 'unibo.it', "r.jina.ai"]
    # start_urls = ['https://corsi.unibo.it/laurea/IngegneriaScienzeInformatiche']
    # start_urls = ['https://corsi.unibo.it/laurea/SviluppoCooperazioneInternazionale']
    start_urls = ['https://corsi.unibo.it/laurea/matematica']

    rules = (
        Rule(
            # LinkExtractor(allow=(r'https://corsi\.unibo\.it/laurea/IngegneriaScienzeInformatiche.*',)),
            # LinkExtractor(allow=(r'https://corsi\.unibo\.it/laurea/SviluppoCooperazioneInternazionale.*',)),
            LinkExtractor(allow=(r'https://corsi\.unibo\.it/laurea/matematica.*',)),
            callback='parse_course_item',
            follow=True
        ),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.sampling_params = get_sampling_params()
        self.llm = LLM(model='jinaai/reader-lm-1.5b', dtype='float16', max_model_len=48000, gpu_memory_utilization=0.9)
        self.log("LLM model instantiated successfully.")
        
    def download_pdf_with_jina(self, response):
        full_link = self.jina_prefix + response.url
        headers = {
            "x-respond-with":"markdown"
        }
        yield scrapy.Request(full_link,
                            callback=self.save_pdf_markdown,
                            headers=headers)
    
    def save_pdf_markdown(self, response):
        url_path = response.url.replace(self.jina_prefix, '').replace('https://corsi.unibo.it/laurea/', '') + ".txt"

        os.makedirs(os.path.dirname(url_path), exist_ok=True)
        with open(url_path, 'wb') as f:
            f.write(response.body)

    def parse_course_item(self, response):
        pdf_links = response.css('a::attr(href)').re(r'.*\.pdf')
        for link in pdf_links:
            yield response.follow(link, self.download_pdf_with_jina)
        
        url_path = response.url.replace('https://corsi.unibo.it/laurea/', '')
        base_path, extension = os.path.splitext(url_path) 

        if not extension:
            url_path += '/'

        if url_path.endswith('/'):
            url_path += 'index.html'

        # Crea la struttura di directory corrispondente all'URL
        os.makedirs(os.path.dirname(url_path), exist_ok=True)

        base_path, extension = os.path.splitext(url_path)
        counter = 1
        while os.path.exists(url_path):
            url_path = f"{base_path}_{counter}{extension}"
            counter += 1
        
        if counter == 2:
            url_path = url_path.replace("_1", "")

        # Salva il file HTML con il nome del file (unico) scaricato dall'URL
        self.logger.info(f'Saving file {url_path}')
        with open(url_path, 'w', encoding='utf-8') as f:
            raw_html = response.body
            if isinstance(raw_html, bytes):
                raw_html = raw_html.decode('utf-8')
            cleaned_html = clean_html(raw_html, clean_svg=True, clean_base64=True)

            prompt = create_prompt(cleaned_html, self.llm.get_tokenizer())
            results = self.llm.generate(prompt, sampling_params=self.sampling_params)

            f.write(results[0].outputs[0].text)

    # def parse_general_unibo_item_og(self, response):
    #     referer_url = response.request.headers.get('Referer', b'').decode('utf-8')
    #     url_path = response.url
    #     base_path, extension = os.path.splitext(url_path)

    #     if not extension:
    #         url_path += '/'

    #     if url_path.endswith('/'):
    #         url_path += 'index.html'

    #     os.makedirs(os.path.dirname(url_path), exist_ok=True)

    #     #Trova il nome corretto per il salvataggio del file
    #     base_path, extension = os.path.splitext(url_path)
    #     counter = 1
    #     while os.path.exists(url_path):
    #         url_path = f"{base_path}_{counter}{extension}"
    #         counter += 1
        
    #     if counter == 2:
    #         url_path = url_path.replace("_1", "")

    #     # Salva il file HTML con il nome del file (unico) scaricato dall'URL
    #     self.logger.info(f'Saving file {url_path}')
    #     with open(url_path, 'wb') as f:
    #         f.write(response.body)
            
    # def parse_general_unibo_item(self, response):
    #     referer_url = response.request.headers.get('Referer', b'').decode('utf-8')
    #     url_path = response.url
    #     base_path, extension = os.path.splitext(url_path)

    #     if not extension:
    #         url_path += '/'

    #     if url_path.endswith('/'):
    #         url_path += 'index.html'

    #     os.makedirs(os.path.dirname(url_path), exist_ok=True)

    #     #Trova il nome corretto per il salvataggio del file
    #     base_path, extension = os.path.splitext(url_path)
    #     counter = 1
    #     while os.path.exists(url_path):
    #         url_path = f"{base_path}_{counter}{extension}"
    #         counter += 1
        
    #     if counter == 2:
    #         url_path = url_path.replace("_1", "")

    #     # Salva il file HTML con il nome del file (unico) scaricato dall'URL

    #     with open('url_paths.txt', 'a') as f:
    #         f.write(output_line)