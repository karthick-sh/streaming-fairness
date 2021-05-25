from selenium import webdriver 
from selenium.common.exceptions import JavascriptException, WebDriverException
from selenium.webdriver.common.action_chains import ActionChains 

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.chrome.options import Options
from pyvirtualdisplay import Display
import time, sys, argparse, glob, random, string
from collections import deque

class PufferPlayer:
    def __init__(self, link, output_file, userNum, headless=False, proxy=None):
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

        if proxy:
            print("Adding proxy! opt file is: ", output_file)
            chrome_options.add_argument('--proxy-server={}'.format(proxy.proxy.proxy))

        self.driver = webdriver.Chrome(options=chrome_options) 
        self.driver.create_options()

        self.link = link
        self.throttleAmount = 3
        self.throttle(self.throttleAmount)

        self.numDatapointsCollected = 0
        self.portalUsername = "maestorme{}".format(userNum)
        self.portalPassword = "cqxav7i"

        self.proxy = proxy
          
        # File for normal motivation stuff
        self.f = open(output_file, "w")
        self.f.write("{},{},{},{},{},{}\n".format(
            "elapsedTime",
            "resolution",
            "bufferHealth",
            "bitrate",
            "currentTime",
            "rebuffering"
        ))
        
        if proxy:
            self.proxy.makeNewHAR(output_file)
            # File for reverse engineering
            self.rf = open(output_file+"-recorded", "w")
            self.rf.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    "time",
                    "resolution",
                    "bitrate",
                    "currentTime",
                    "rebuffering",
                    "buffer-0",
                    "buffer-1",
                    "buffer-2",
                    "buffer-3",
                    "buffer-4",
                    "buffer-5",
                    "buffer-6",
                    "buffer-7",
                    "buffer-8",
                    "buffer-9",
                    "bw-0",
                    "bw-1",
                    "bw-2",
                    "bw-3",
                    "bw-4",
                    "bw-5",
                    "bw-6",
                    "bw-7",
                    "bw-8",
                    "bw-9",
                    "index",
                    "timestamp_start",
                    "timestamp_finish",
                    "n_segment",
                    "t_download_s",
                    "body_size_byte",
                    "bandwidth_mbit",
                    "byte_start",
                    "byte_end",
                    "url",
            ))

            self.df = open(output_file+"-downloaded", "w")
            self.df.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(
                'time',
                'index',
                'timestamp_start',
                'timestamp_finish',
                'n_segment',
                't_download_s',
                'body_size_byte',
                'bandwidth_mbit',
                'byte_start',
                'byte_end',
                'url',
            ))

            self.bwWindow = deque(maxlen=10)
            self.bufWindow = deque(maxlen=10)

            for _ in range(9):
                self.bwWindow.appendleft(0)
                self.bufWindow.appendleft(0)
            self.bwWindow.appendleft(self.throttleAmount)
            self.bufWindow.appendleft(0)


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
        details["elapsedTime"] = time.time() - self.startTime
        details["rebuffering"] = 0

        if details["resolution"] == "N/A" or details["bufferHealth"] == "N/A" or details['bitrate'] == "N/A":
            return "N/A", details

        rebuffering = abs((details["currentTime"] - self.prevDetails["currentTime"]) - (details["elapsedTime"] - self.prevDetails["elapsedTime"]))
        if rebuffering > 0.1 and rebuffering < 2:
            details["rebuffering"] = rebuffering
            details["bitrate"] = (1-(rebuffering / (details["elapsedTime"] - self.prevDetails["elapsedTime"]))) * float(details["bitrate"])

        self.prevDetails["currentTime"] = details["currentTime"]
        self.prevDetails["elapsedTime"] = details["elapsedTime"]

        self.f.write("{},{},{},{},{},{}\n".format(
            details["elapsedTime"],
            details["resolution"],
            details["bufferHealth"],
            details["bitrate"],
            details["currentTime"],
            details["rebuffering"]
        ))

        return "{} - {}".format(details["currentTime"], details["resolution"]), details

    # Use this function for reverse engineering
    def collectProxyData(self):
        _, details = self.collectData()
        if details == {}:
            return "N/A", {}
        self.bufWindow.appendleft(details["bufferHealth"])

        media_requests = self.proxy.getMediaRequests()
        newly_recorded, newly_downloaded = self.proxy.parseMediaRequests(media_requests)

        if len(newly_recorded) != 0:
            # Collect required stats from player
            for entry in newly_recorded:
                self.rf.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    details["time"],
                    details["resolution"],
                    details["bitrate"],
                    details["currentTime"],
                    details["rebuffering"],
                    list(self.bufWindow)[0],
                    list(self.bufWindow)[1],
                    list(self.bufWindow)[2],
                    list(self.bufWindow)[3],
                    list(self.bufWindow)[4],
                    list(self.bufWindow)[5],
                    list(self.bufWindow)[6],
                    list(self.bufWindow)[7],
                    list(self.bufWindow)[8],
                    list(self.bufWindow)[9],
                    list(self.bwWindow)[0],
                    list(self.bwWindow)[1],
                    list(self.bwWindow)[2],
                    list(self.bwWindow)[3],
                    list(self.bwWindow)[4],
                    list(self.bwWindow)[5],
                    list(self.bwWindow)[6],
                    list(self.bwWindow)[7],
                    list(self.bwWindow)[8],
                    list(self.bwWindow)[9],
                    entry['index'],
                    entry['timestamp_start'],
                    entry['timestamp_finish'],
                    entry['n_segment'],
                    entry['t_download_s'],
                    entry['body_size_byte'],
                    entry['bandwidth_mbit'],
                    entry['byte_start'],
                    entry['byte_end'],
                    entry['url'],
                ))

        if len(newly_downloaded) !=  0:
            for entry in newly_downloaded:
                self.df.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    details["time"],
                    entry['index'],
                    entry['timestamp_start'],
                    entry['timestamp_finish'],
                    entry['n_segment'],
                    entry['t_download_s'],
                    entry['body_size_byte'],
                    entry['bandwidth_mbit'],
                    entry['byte_start'],
                    entry['byte_end'],
                    entry['url'],
                ))

        return "{} - {}".format(details["currentTime"], details["resolution"]), details
        
    def bufferCleared(self):
        buffer = float(self.bufferSpan.get_attribute("innerHTML"))
        if buffer < 1:
            return True
        return False

    def stopDisplay(self):
        # if self.headless:
        #     self.display.stop()
        return