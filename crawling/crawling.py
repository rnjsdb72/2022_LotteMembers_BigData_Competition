#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tqdm import tqdm
import pandas as pd
import argparse

import chromedriver_autoinstaller
from selenium.webdriver.common.by import By
from selenium.webdriver import Chrome

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cafe', type=str, default='true')
    parser.add_argument('--blog', type=str, default='true')
    args = parser.parse_args()
    return args

def disc_bool(x):
    if x == 'true':
        return True
    elif x == 'false':
        return False

def crawling_cafe_n_blog(query, cafe=True, blog=True):
    dir_driver = chromedriver_autoinstaller.install()
    browser = Chrome(dir_driver)
    browser.implicitly_wait(15)

    if cafe:
        # 카페 url 탐색
        urls = []
        i = 1
        while True:
            try:
                browser.get(f"https://section.cafe.naver.com/ca-fe/home/search/articles?q={query}&p={i}")
                data = browser.find_elements(By.CSS_SELECTOR, '.ArticleList .article_item .item_subject')
                if i != 1:
                    if titles == list(map(lambda x: x.text, data)):
                        break
                titles = list(map(lambda x: x.text, data))
                urls = urls + list(map(lambda x: x.get_property('href'), data))
                i += 1
            except:
                break

        # 카페 크롤링
        print('Crawling Naver Cafe,,,')
        titles = []
        dates = []
        contexts = []
        urls2 = []
        for url in tqdm(urls):
            browser.get(url)
            browser.switch_to.frame('cafe_main')
            try:
                titles.append(str(browser.find_element(By.CSS_SELECTOR, '.title_text').text))
            except:
                titles.append("no_title")

            try:
                dates.append(str(browser.find_element(By.CSS_SELECTOR, '.article_info .date').text))
            except:
                dates.append("no_date")

            try:
                contexts.append(' '.join(list(map(lambda x: x.text, browser.find_elements(By.CSS_SELECTOR, '.se-module.se-module-text, .ContentRenderer p, .ContentRenderer div')))))
            except:
                contexts.append("no_context")

            try:
                urls2.append(str(url))
            except:
                urls2.append("no_url")

        # 카페 크롤링 내용 데이터프레임화 및 저장
        cafes = ['naver cafe'] * len(titles)
        pd.DataFrame({'url':urls2, 'type':cafes, 'title':titles, 'date':dates, 'context':contexts}).to_csv(f'naver_cafe_{query}.csv', index=False)
        print(f'"naver_cafe_{query}.csv" is Saved!')

    if blog:
        # 블로그 url 탐색
        urls = []
        i = 1
        while True:
            try:
                browser.get(f"https://section.blog.naver.com/Search/Post.naver?pageNo={i}&rangeType=ALL&orderBy=sim&keyword={query}")
                data = browser.find_elements(By.CSS_SELECTOR, '.desc_inner')
                if i != 1:
                    if titles == list(map(lambda x: x.text, data)):
                        break
                titles = list(map(lambda x: x.text, data))
                urls = urls + list(map(lambda x: x.get_property('href'), data))
                i += 1
            except:
                break

        # 블로그 크롤링
        print('Crawling Naver Blog,,,')
        titles = []
        dates = []
        contexts = []
        urls2 = []
        for url in tqdm(urls):
            browser.get(url)
            browser.switch_to.frame('mainFrame')
            try:
                titles.append(str(browser.find_element(By.CSS_SELECTOR, '.pcol1').text))
            except:
                titles.append("no_title")
            try:
                dates.append(str(browser.find_element(By.CSS_SELECTOR, '.blog2_container .se_publishDate.pcol2, .date.fil5.pcol2._postAddDate').text))
            except:
                dates.append("no_date")
                
            try:
                contexts.append(' '.join(list(map(lambda x: x.text, browser.find_elements(By.CSS_SELECTOR, '.se-main-container, #postViewArea, .se_textarea')))))
            except:
                contexts.append("no_contexts")
                
            try:
                urls2.append(str(url))
            except:
                urls2.append("no_url")

        # 블로그 크롤링 내용 데이터 프레임화 및 저장
        blogs = ['naver blog'] * len(titles)
        pd.DataFrame({'url':urls2, 'type':blogs, 'title':titles, 'date':dates, 'context':contexts}).to_csv(f'naver_blog_{query}.csv', index=False)
        print(f'"naver_blog_{query}.csv" is Saved!')
        
if __name__ == '__main__':
    args = arg_parse()
    arg_cafe = disc_bool(args.cafe)
    arg_blog = disc_bool(args.blog)
    query = input('Enter your search term: \t')
    crawling_cafe_n_blog(query, arg_cafe, arg_blog)


# In[ ]:




