from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
from pyvirtualdisplay import Display

# 기본 세팅 방법
# https://github.com/password123456/setup-selenium-with-chrome-driver-on-ubuntu_debian?tab=readme-ov-file#step-6-create-hello_world

# Selenium python document

if __name__=="__main__":
    # 가상 디스플레이 시작
    display = Display(visible=1, size=(1920, 1080), backend="xvfb")
    display.start()
    
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    # options.add_argument('user-data-dir=C:\\User Data')
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("disable-gpu")   # 가속 사용 x
    options.add_argument("lang=ko_KR")    # 가짜 플러그인 탑재
    options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36')
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
    
    _from = 'japanese'
    _to = 'korean'
    _with = 'images'

    url = f"https://translate.google.co.kr/?sl={lang_opt[_from]}&tl={lang_opt[_to]}&op={translate_opt[_with]}"

    driver.get(url)
    
    file_input = driver.find_element(By.XPATH, '//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[4]/c-wiz/div[2]/c-wiz/div/div/div/div[1]/div[2]/div[2]/div[1]/input')
    file_path = '/data/ephemeral/home/kwak/level2-cv-datacentric-cv-16/data/japanese_receipt/img/train/extractor.ja.in_house.appen_000028_page0001.jpg'
    
    file_input.send_keys(file_path)
    
    time.sleep(2)
    download_btn = driver.find_element(By.XPATH, '//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[4]/c-wiz/div[2]/c-wiz/div/div[1]/div[2]/div[2]/button')
    
    download_btn.click()
    
    time.sleep(2)
    
    rm_curr_btn = driver.find_element(By.XPATH, '//*[@id="yDmH0d"]/c-wiz/div/div[2]/c-wiz/div[4]/c-wiz/div[2]/c-wiz/div/div[1]/div[2]/span[3]/button')
    
    rm_curr_btn.click()
    
    time.sleep(2)
    
    driver.close()
    display.stop()