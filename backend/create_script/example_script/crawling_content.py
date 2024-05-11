from selenium import webdriver
from selenium.webdriver.common.by import By
import time

def crawling_content(url):
    driver = webdriver.Chrome(executable_path='/usr/bin/chromedriver') 
    driver.get(url)
    time.sleep(1)

    content = driver.find_element(By.ID, 'content').text
    return content