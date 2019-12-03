from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import time
import os
import numpy as np

FILES = [r'data\ID_0000ca2f6.dcm', r'data\ID_0000ca2f6.dcm', r'data\ID_0000f1657.dcm', r'data\ID_0001dcc25.dcm',
         r'data\ID_0001de0e8.dcm', r'data\ID_0002d438a.dcm', r'data\ID_0004cd66f.dcm', r'data\ID_0005b2d86.dcm',
         r'data\ID_0005db660.dcm', r'data\ID_00008ce3c.dcm', r'data\ID_000012eaf.dcm', r'data\ID_00032d440.dcm',
         r'data\ID_000039fa0.dcm', r'data\ID_00044a417.dcm', r'data\ID_000178e76.dcm', r'data\ID_000259ccf.dcm',
         r'data\ID_0000950d7.dcm', r'data\ID_00005679d.dcm', r'data\ID_00019828f.dcm', r'data\ID_000624786.dcm']


def predict_from_home(driver, change_dir=True):
    proceed_button = driver.find_element_by_class_name('btn')
    time.sleep(3)
    proceed_button.click()
    time.sleep(2)
    upload_button = driver.find_element_by_id('file')
    time.sleep(2)
    if change_dir:
        os.chdir('..')
    current_dir = os.getcwd()
    files = np.random.choice([os.path.join(current_dir, file) for file in FILES], 5)
    files = '\n'.join(files)
    upload_button.send_keys(files)
    time.sleep(2)
    predict_button = driver.find_element_by_id('predict_button')
    time.sleep(2)
    hov = ActionChains(driver).move_to_element(predict_button)
    hov.click()
    hov.perform()
    time.sleep(30)


def go_to_about(driver):
    about_button = driver.find_element_by_xpath('//a[@href="' + '/about' + '"]')
    time.sleep(2)
    about_button.click()
    time.sleep(2)


def go_home(driver):
    home_button = driver.find_element_by_xpath('//a[@href="' + '/' + '"]')
    time.sleep(2)
    home_button.click()
    time.sleep(2)


def crawl_site(driver):
    driver.get('http://127.0.0.1:5000/')
    predict_from_home(driver)
    go_to_about(driver)
    go_home(driver)
    predict_from_home(driver, change_dir=False)
    go_home(driver)
    predict_from_home(driver)
    go_home(driver)


def main():
    start_time = time.time()
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(r'E:\Descarcari\chromedriver_win32\chromedriver.exe', options=options)
    crawl_site(driver)
    print(time.time() - start_time)


if __name__ == '__main__':
    main()
