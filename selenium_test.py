from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


if __name__=="__main__":
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # driver.get("https://python.org")
    # print(driver.title)

    translate_opt = {'text': 'translate',
                     'images' : 'images',
                     'docs' : 'docs'}
    lang_opt = {'english' : 'en',
                'korean' : 'ko',
                'vietnamese' : 'vi',
                'thai' : 'th',
                'japanese' : 'ja',
                'chinese' : 'zh-CN'}
    
    _from = 'korean'
    _to = 'english'
    _with = 'images'

    url = f"https://translate.google.co.kr/?sl={lang_opt[_from]}&tl={lang_opt[_to]}&op={translate_opt[_with]}"

    driver.get(url)
    
    print(driver.page_source)
    
    driver.close()