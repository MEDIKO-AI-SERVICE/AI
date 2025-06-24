from selenium import webdriver
from selenium.webdriver.chrome.service import Service
#from webdriver_manager.chrome import ChromeDriverManager


def setup_driver():
    """Headless Chrome WebDriver 설정"""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")  #브라우저 창을 띄우지 않음
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    #service = Service(ChromeDriverManager().install())
    #chromedriver 실행 경로 설정
    service = Service("/usr/local/bin/chromedriver")
    driver = webdriver.Chrome(service=service, options=options)
    return driver