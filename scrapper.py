from bs4 import BeautifulSoup
import spacy,re
import requests
from selenium import webdriver
from selenium.webdriver.firefox.options import Options



nlp = spacy.load("en_core_web_sm")

def worker(query):
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Firefox(options=options)
    driver.get(query)
    return driver.page_source


def scrapper(query):
    data = worker(f"https://www.bing.com/search?q={query}")
    tex = BeautifulSoup(data,features="lxml")
    for script_or_style in tex(["script", "style", "head", "meta", "link","a","header","footer","noscript","title","video","button"]):
        script_or_style.decompose()

    text = re.sub(r'&nbsp;|&amp;|&quot;|&lt;|&gt;', ' ', tex.text)
    text = re.sub(
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|'
        r'January|February|March|April|May|June|July|August|'
        r'September|October|November|December)'
        r'\s+\d{1,2},?\s+\d{4}', '',
        text
    )
    text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '', text)
    text = re.sub(r'[•–—»▶►▪]', ' ', text)
    text = re.sub(r'\s+', ' ', text)


    docs = nlp(text)
    sentences = [sent.text.strip() for sent in docs.sents]
    return sentences
    
    
