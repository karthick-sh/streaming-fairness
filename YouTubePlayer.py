from selenium import webdriver 
from selenium.common.exceptions import JavascriptException, WebDriverException
from selenium.webdriver.common.action_chains import ActionChains 

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.chrome.options import Options
from pyvirtualdisplay import Display
import pandas as pd
from datetime import datetime
import time, sys, argparse, glob, random, string 
from collections import deque

class YouTubePlayer:
    def __init__(self, link, output_file, userNum, headless=False):
        # self.display = Display(visible=0, size=(1920, 1080))
        # self.display.start()

        path_to_extension = r'/home/karthick-sh/Desktop/streaming-fairness/extensions/5.1.2_0'
        chrome_options = Options()
        chrome_options.add_argument('load-extension=' + path_to_extension)
        chrome_options.add_argument('--disable-application-cache')
        chrome_options.add_argument('disable-application-cache')
        chrome_options.add_argument('ignore-ssl-errors=yes')
        chrome_options.add_argument('ignore-certificate-errors')
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--ignore-ssl-errors')
        chrome_options.add_argument("window-size=1920x1080")
        # chrome_options.add_argument('window-size=2880x1680')
        chrome_options.add_argument("--start-maximized")
        # chrome_options.add_argument('--no-sandbox')
        chrome_options.headless = False

        self.driver = webdriver.Chrome(options=chrome_options) 
        self.driver.create_options()

        self.link = link
        self.throttleAmount = 10
        self.throttle(self.throttleAmount)

        self.numDatapointsCollected = 0

        self.bitrateMapping = pd.read_csv("../understanding-video-streaming-in-the-wild/Data/VideoInformation/YouTube_Info/QZUeW8cQQbo_video_info")
        self.widthsToHeights = {
            256: 144,
            426: 240,
            640: 360,
            854: 480,
            1280: 720,
            1920: 1080
        }

        self.f = open(output_file, "w")
        self.f.write("{},{},{},{},{},{}\n".format(
            "time",
            "resolution",
            "bufferHealth",
            "bitrate",
            "currentTime",
            "rebuffering"
        ))

        self.bufferWindow = deque(maxlen=3)
        for _ in range(3):
            self.bufferWindow.appendleft(0)

    def play(self):
        self.driver.execute_script("document.getElementsByTagName('video')[0].play()")

    def pause(self):
        self.driver.execute_script("document.getElementsByTagName('video')[0].pause()")

    def fullscreen(self):
        fullscreenButton = self.driver.find_element_by_css_selector('button.ytp-fullscreen-button.ytp-button')
        fullscreenButton.click()

        # self.driver.execute_script("document.getElementsByTagName('video')[0].requestFullscreen()")
        # self.driver.execute_script("document.getElementsByTagName('video')[0].webkitRequestFullScreen()")

    def mute(self):
        self.driver.execute_script("document.getElementsByTagName('video')[0].volume = 0")

    def throttle(self, bw):
         self.driver.set_network_conditions(
            offline=False,
            latency=0,
            download_throughput=(bw * 1000 * 1024)//8,  # maximal throughput
            upload_throughput=(bw * 1000 * 1024)//8,  # maximal throughput
        )
    
    def initVideo(self):
        self.driver.get(self.link)
        # print("Old viewport size:", self.driver.get_window_size())
        # self.driver.set_window_size(1920, 1080)
        # print("New viewport size:", self.driver.get_window_size())

        wait = WebDriverWait(self.driver, 60)
        # actions = ActionChains(self.driver)
        
        self.driver.switch_to.window(self.driver.window_handles[0])
        # Wait for video element to load
        self.video = wait.until(EC.element_to_be_clickable((By.TAG_NAME, "video")))

        self.throttle(0.001)
        self.mute()
        self.fullscreen()
        # self.pause()
        

        # Set to theatre mode for the first tab, auto sets for other tabs
        # actions.send_keys("t").perform()

    def start(self):
        self.throttle(self.throttleAmount)
        self.play()

        self.startTime = time.time()
        self.prevDetails = {
            "currentTime": 0,
            "time": 0
        }

    def stopDisplay(self):
        return

    def collectData(self, currentBW):
        details = {}

        # Collect required stats
        viewport_h = self.driver.execute_script("return document.getElementsByTagName('video')[0].clientHeight")
        viewport_w = self.driver.execute_script("return document.getElementsByTagName('video')[0].clientWidth")

        video_w = self.driver.execute_script("return document.getElementsByTagName('video')[0].videoWidth")
        video_h = self.driver.execute_script("return document.getElementsByTagName('video')[0].videoHeight")

        details["currentTime"] =  self.driver.execute_script("return document.getElementsByTagName('video')[0].currentTime")
        details["time"] = time.time() - self.startTime
        details["rebuffering"] = 0
        
        if int(video_h) == 0:
            return "ERROR:: t: {}, video:{}x{}, viewport:{}x{}".format(details["currentTime"], video_w, video_h, viewport_w, viewport_h), {}

        if int(video_h) != self.widthsToHeights[int(video_w)]:
            print("Height not same, got {}, expected {}".format(video_h, self.widthsToHeights[int(video_w)])) 
            video_h = self.widthsToHeights[int(video_w)]
        details["resolution"] = "{}x{}".format(video_w, video_h)

        try:
            details["bufferHealth"] = self.driver.execute_script("var vid = document.getElementsByTagName('video')[0]; return vid.buffered.end(0) - vid.currentTime")
        except WebDriverException:
            details["bufferHealth"] = 0

        # Get bitrate from mapping
        bitrateCol = "{}x{}_bitrate".format(video_w, video_h)
        currentTime = float(details["currentTime"])
        prev = None
        for idx, j in self.bitrateMapping.iterrows():
            if idx == 0:
                prev = j
            if convertTSToS(j["delta_t_s"]) > currentTime:
                break
            prev = j
        details["bitrate"] = prev[bitrateCol]/1000

        rebuffering = abs((details["currentTime"] - self.prevDetails["currentTime"]) - (details["time"] - self.prevDetails["time"]))
        if rebuffering > 0.1 and rebuffering < 10:
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

        self.bufferWindow.appendleft(float(details["bufferHealth"]))

        outputs = {
            "buffer": list(self.bufferWindow),
            "bitrate": float(details["bitrate"])
        }
        return "{} - {} on ({}x{})".format(details["currentTime"], details["resolution"], viewport_w, viewport_h), outputs

def convertTSToS(ts):
    date_time = datetime.strptime(ts, "%H:%M:%S")
    a_timedelta = date_time - datetime(1900, 1, 1)

    return a_timedelta.total_seconds()