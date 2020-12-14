import requests
import pandas as pd
from scrapy.selector import Selector

def get_article_url(url_,list_=[]):
    s = requests.session()
    r = s.get(url_)
    # print(r.text)
    selector = Selector(r)
    article_url = selector.xpath("//li[@class='unit']//a/@href").extract()
    for article in article_url:
        list_.append('https://xiaoxue.hujiang.com'+article)
    next_page_url=get_next_page_url(url_)
    if next_page_url=='No next page':
        return list_
    else:
        return get_article_url(next_page_url,list_)

def get_next_page_url(url_):#url_= https://xiaoxue.hujiang.com/zuowen/yi
    s = requests.session()
    r = s.get(url_)
    # print(r.text)
    selector = Selector(r)
    next_url = selector.xpath("//div[@id='split_page']/a[text()='下一页 >']//@href").extract_first()
    if (next_url):
        return 'https://xiaoxue.hujiang.com'+next_url
    else:
        return 'No next page'
title=[]
content=[]
def parse():
    start_url = "https://xiaoxue.hujiang.com/zuowen/"


    for i in ['yi','er','san','four','five','six']:
        url = start_url + str(i)
        for i in get_article_url(url):
            get_title_content(i)
    df = {'title': title, 'content': content}
    df = pd.DataFrame(df)
    df.to_csv("./article.csv", encoding='utf_8_sig')
def get_title_content(url_):

    title_,content_=get_article_detail(url_)
    title.append(title_)
    content.append(content_)

def get_article_detail(url_):
    s = requests.session()
    r = s.get(url_)
    # print(r.text)
    selector = Selector(r)
    title = selector.xpath("//h1//text()").extract_first().strip()
    next_url = selector.xpath("//div[@id='article']/p//text()").extract()
    content=''
    for i in next_url:
        content+=i.replace('\r\n\t','\n').replace('\xa0',' ')
    return title,content
if __name__ == "__main__":
    parse()