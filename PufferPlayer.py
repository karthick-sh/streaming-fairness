from selenium import webdriver 
from selenium.common.exceptions import JavascriptException, WebDriverException
from selenium.webdriver.common.action_chains import ActionChains 

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.chrome.options import Options
from pyvirtualdisplay import Display
import time, sys, argparse, glob, random, string

class PufferPlayer:
    def __init__(self, link, output_file, userNum, headless=False):
        self.headless = headless
        # if headless:
        #     self.display = Display(visible=0, size=(1920, 1080))  
        #     self.display.start()

        chrome_options = Options()
        chrome_options.add_argument('--disable-application-cache')
        chrome_options.add_argument('disable-application-cache')
        chrome_options.add_argument('ignore-ssl-errors=yes')
        chrome_options.add_argument('ignore-certificate-errors')
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--ignore-ssl-errors')
        chrome_options.headless = headless

        self.driver = webdriver.Chrome(options=chrome_options) 
        self.driver.create_options()

        self.link = link
        self.throttleAmount = 10
        self.throttle(self.throttleAmount)

        self.numDatapointsCollected = 0
        self.portalUsername = "maestorme{}".format(userNum)
        self.portalPassword = "cqxav7i"
        # self.portalUsername = "maestorme"
        # self.portalPassword = "2xLvvzd2k8aBwNm"

        self.f = open(output_file, "w")
        self.f.write("{},{},{},{},{},{}\n".format(
            "time",
            "resolution",
            "bufferHealth",
            "bitrate",
            "currentTime",
            "rebuffering"
        ))

    def play(self):
        self.driver.execute_script("document.getElementsByTagName('video')[0].play()")

    def pause(self):
        self.driver.execute_script("document.getElementsByTagName('video')[0].pause()")

    def fullscreen(self):
        fullscreenButton = self.driver.find_element_by_css_selector('#full-screen-button')
        fullscreenButton.click()

    def mute(self):
        self.driver.execute_script("document.getElementsByTagName('video')[0].volume = 0")

    def throttle(self, bw):
        self.driver.set_network_conditions(
            offline=False,
            latency=0,
            download_throughput=(bw * 1000 * 1024)//8,  # maximal throughput
            upload_throughput=(bw * 1000 * 1024)//8,  # maximal throughput
        )

    def setOffline(self):
        self.driver.set_network_conditions(
            offline=True,
            latency=0,
            download_throughput=(0.001 * 1000 * 1024)//8,  # maximal throughput
            upload_throughput=(0.001 * 1000 * 1024)//8,  # maximal throughput
        )

    def initVideo(self):
        for trial in range(5):
            try:
                self.driver.get(self.link)
                break
            except WebDriverException:
                time.sleep(5)
                print("RETRYING DRIVER.GET")
                if trial >= 4:
                    self.driver.get(self.link)
                    break

        wait = WebDriverWait(self.driver, 60)

        # Log into Puffer
        username = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[name='username']")))
        password = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input[name='password']")))
        checkbox = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "label.custom-control-label")))
        loginButton = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".btn-lg")))

        username.send_keys(self.portalUsername)
        password.send_keys(self.portalPassword)
        checkbox.click()
        loginButton.click()

        # Wait for video element to load
        self.video = wait.until(EC.element_to_be_clickable((By.TAG_NAME, "video")))

        self.setOffline()

        self.bufferSpan = self.driver.find_element_by_id('video-buf')
        self.resolutionSpan = self.driver.find_element_by_id('video-res')
        self.bitrateSpan = self.driver.find_element_by_id('video-bitrate')

        self.fullscreen()

    def start(self):
        self.throttle(self.throttleAmount)
        self.startTime = time.time()
        self.prevDetails = {
            "currentTime": -1,
            "time": 0
        }


    def collectData(self):
        details = {}

        # Collect required stats
        details["resolution"] = self.resolutionSpan.get_attribute("innerHTML")
        details["bufferHealth"] = self.bufferSpan.get_attribute("innerHTML")
        details["bitrate"] = self.bitrateSpan.get_attribute("innerHTML")
        details["currentTime"] =  self.driver.execute_script("return document.getElementsByTagName('video')[0].currentTime")
        details["time"] = time.time() - self.startTime
        details["rebuffering"] = 0

        if details["resolution"] == "N/A" or details["bufferHealth"] == "N/A" or details['bitrate'] == "N/A":
            return

        rebuffering = abs((details["currentTime"] - self.prevDetails["currentTime"]) - (details["time"] - self.prevDetails["time"]))
        if rebuffering > 0.1 and rebuffering < 2:
            details["rebuffering"] = rebuffering
            details["bitrate"] = (1-(rebuffering / (details["time"] - self.prevDetails["time"]))) * float(details["bitrate"])

        self.prevDetails["currentTime"] = details["currentTime"]
        self.prevDetails["time"] = details["time"]

        self.f.write("{},{},{},{},{},{}\n".format(
            details["time"],
            details["resolution"],
            details["bufferHealth"],
            details["bitrate"],
            details["currentTime"],
            details["rebuffering"]
        ))

        return "{} - {}".format(details["currentTime"], details["resolution"])

    def bufferCleared(self):
        buffer = float(self.bufferSpan.get_attribute("innerHTML"))
        if buffer < 1:
            return True
        return False

    def stopDisplay(self):
        # if self.headless:
        #     self.display.stop()
        return