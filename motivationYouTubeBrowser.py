from selenium import webdriver 
from selenium.common.exceptions import JavascriptException, ElementClickInterceptedException, WebDriverException
from selenium.webdriver.common.action_chains import ActionChains 

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.chrome.options import Options
import time, sys, argparse, glob, random, string, psutil, os

from multiprocessing import Process, Semaphore, Value

from PufferPlayer import PufferPlayer
from BrowserMobProxy import BrowserMobProxy
from TrafficController import TrafficController

class YouTubeController:
    def __init__(self, link, numBrowsers, folderName, experimentName):
        path_to_extension = r'D:\CMU\Fall 2020\Cylab\1.30.6_0'
        chrome_options = Options()
        chrome_options.add_argument('load-extension=' + path_to_extension)
        chrome_options.add_argument('--disable-application-cache')
        chrome_options.add_argument('disable-application-cache')
        chrome_options.add_argument('ignore-ssl-errors=yes')
        chrome_options.add_argument('ignore-certificate-errors')
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--ignore-ssl-errors')

        self.driver = webdriver.Chrome(options=chrome_options) 
        self.driver.create_options()

        self.link = link
        self.throttleAmount = 10
        self.throttle(self.throttleAmount)

        self.numDatapointsCollected = 0

        self.f = open("{}/mot-{}-{}.txt".format(folderName, numBrowsers, experimentName), "w")
        self.f.write("{},{},{},{},{},{}\n".format(
           "viewport_h",
           "viewport_w",
           "video_h",
           "video_w",
           "buffer",
           "currentTime"
        ))

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

        wait = WebDriverWait(self.driver, 60)
        # actions = ActionChains(self.driver)

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

    def collectData(self):
        details = {}

        # Collect required stats
        details["viewport_h"] = self.driver.execute_script("return document.getElementsByTagName('video')[0].clientHeight")
        details["viewport_w"] = self.driver.execute_script("return document.getElementsByTagName('video')[0].clientWidth")
        details["video_h"] = self.driver.execute_script("return document.getElementsByTagName('video')[0].videoHeight")
        details["video_w"] = self.driver.execute_script("return document.getElementsByTagName('video')[0].videoWidth")
        try:
            details["buffer"] = self.driver.execute_script("var vid = document.getElementsByTagName('video')[0]; return vid.buffered.end(0) - vid.currentTime")
        except WebDriverException:
            details["buffer"] = 0

        details["currentTime"] =  self.driver.execute_script("return document.getElementsByTagName('video')[0].currentTime")

        self.f.write("{},{},{},{},{},{}\n".format(
            details["viewport_h"],
            details["viewport_w"],
            details["video_h"],
            details["video_w"],
            details["buffer"],
            details["currentTime"]
        ))

def killWebdrivers():
    for proc in psutil.process_iter():
        if proc.name() == "chromedriver.exe":
            proc.kill()
            print("Killed!")

class Barrier:
    def __init__(self, n):
        self.n       = n
        self.count   = Value('i', 0)
        self.mutex   = Semaphore(1)
        self.barrier = Semaphore(0)

    def wait(self):
        self.mutex.acquire()
        self.count.value += 1
        self.mutex.release()

        if self.count.value == self.n:
            self.barrier.release()

        self.barrier.acquire()
        self.barrier.release()

def incrementTC(tc, barrier):
    barrier.wait()
    start = time.time()
    t = start
    while True:
        if time.time() - t > 1:
            nextBw = tc.getNextBW(time.time() - start + tc.times[0])
            print(tc.bIdx, time.time() - start + tc.times[0], nextBw)
            if nextBw is None:
                break
            tc.throttleNB(nextBw)
            t = time.time()

        time.sleep(0.5)

def pufferPortalThreadOrchestrator(link, abr, traceFile, numDatapointsNeeded, datapointsInterval):
    failedRuns = open("failed-runs.txt", "w")
    for NUM_BROWSERS in [4]:
        for run in range(9, 10):
            print("=======RUN {}-{}=======".format(NUM_BROWSERS, run))
            killWebdrivers()

            tc = None
            if traceFile != "":
                print("USING A TRACE!")
                tc = TrafficController(traceFile)
                tc.setInitialNB(4)
            else:
                print("PRISTINE, setting BW to 4mbit/s", checkNB(4))
                # if not checkNB(4):
                #     setNBLimit(4)
                #     if not checkNB(4):
                #         sys.exit("UNABLE TO SET NB")

            processes = []

            barrier = Barrier(NUM_BROWSERS+1)

            for i in range(NUM_BROWSERS):
                processes.append(Process(
                    target=pufferPortalThread,
                    args=(i+1, tc, run, NUM_BROWSERS, barrier, link, abr, traceFile, numDatapointsNeeded, datapointsInterval)
                ))

            if traceFile != "":
                tc.cleanNB()

            for p in processes:
                p.start()

            tc_process = Process(target=incrementTC, args=(tc,barrier))
            tc_process.start()

            tc_process.join()
            for p in processes:
                p.join()

            # Cleanup
            time.sleep(1)
            for p in processes:
                p.terminate()
            killWebdrivers()

            time.sleep(35)

    failedRuns.close()

def pufferPortalThread(playerNum, tc, run, NUM_BROWSERS, barrier, link, abr, traceFile, numDatapointsNeeded, datapointsInterval):
    if traceFile == "":
        print("PRISTINE")
        folderPart = "pristine"
    else:
        print("TRACE")
        folderPart = "trace"

    if abr == "youtube":
        browser = YouTubeController("https://www.youtube.com/watch?v=QZUeW8cQQbo", NUM_BROWSERS, "motivation-{}-2-21/{}".format(folderPart, abr), "yt-{}-{}".format(run, playerNum))
    else:
        browser = PufferPlayer(NUM_BROWSERS, playerNum, "{}-{}-{}".format(abr, run, playerNum), "motivation-{}-2-21/{}".format(folderPart, abr), link, False)

    try:
        browser.initVideo()
    except ElementClickInterceptedException:
        return False

    # Barrier to wait for all browsers to be ready
    barrier.wait()

    time.sleep(2)

    browser.start()
    numDatapointsCollected = 0
    start = time.time()
    t = start
    while True:
        browser.collectData()
        numDatapointsCollected += 1

        # Exit condition
        if traceFile == "":
            print("numDatapoints:", numDatapointsCollected)
            if numDatapointsCollected >= numDatapointsNeeded:
                break
        else:
            if time.time() - t > 1:
                nextBw = tc.getNextBW(time.time() - start + tc.times[0])
                print("[{}] - {}".format(playerNum, tc.bIdx))
                if nextBw is None:
                    break
                t = time.time()

        time.sleep(datapointsInterval)
    browser.f.close()
    browser.driver.close()
    browser.driver.quit()

    return True

def checkNB(bw):
    bw_bytes_per_sec = int(bw * 125000)
    nb_path = 'D:\\Program Files\\NetBalancer\\nbcmd.exe'
    cmd = '"{}" settings'.format(nb_path)
    lines = os.popen(cmd).read().split("\n")

    isEnabled = False
    isCorrectValue = False
    for line in lines:
        if line[:34] == "System traffic download is limited":
            if line.split(': ')[-1] == "True":
                isEnabled = True
            else:
                print(line)
        if line[:29] == "System traffic download limit":
            if line.split(': ')[-1] == str(bw_bytes_per_sec):
                isCorrectValue = True
            else:
                print(line)
    return isEnabled and isCorrectValue

def setNBLimit(bw):
    bw_bytes_per_sec = int(bw * 125000)
    nb_path = 'D:\\Program Files\\NetBalancer\\nbcmd.exe'
    cmd = '"{}" settings traffic limit true false {} 9990000'.format(nb_path, bw_bytes_per_sec)
    os.system(cmd)

# def pufferPortalMain(link, abr, traceFile, numDatapointsNeeded, datapointsInterval):
#     failedRuns = open("failed-runs.txt", "w")
#     for NUM_BROWSERS in [4]:
#         for run in range(9, 10):
#             print("=======RUN {}-{}=======".format(NUM_BROWSERS, run))
#             killWebdrivers()

#             if traceFile != "":
#                 print("USING A TRACE: ", traceFile)
#                 tc = TrafficController(traceFile)
#                 tc.setInitialNB(4)
#             else:
#                 print("PRISTINE, setting BW to 4mbit/s", checkNB(4))
#                 # if not checkNB(4):
#                 #     setNBLimit(4)
#                 #     if not checkNB(4):
#                 #         sys.exit("UNABLE TO SET NB")

#             browsers = []
#             anyFailed = False
#             for i in range(NUM_BROWSERS):
#                 if traceFile == "":
#                     print("PRISTINE")
#                     folderPart = "pristine"
#                 else:
#                     print("TRACE")
#                     folderPart = "trace"

#                 browsers.append(
#                     YouTubeController("https://www.youtube.com/watch?v=QZUeW8cQQbo", NUM_BROWSERS, "motivation-{}-2-21/{}".format(folderPart, abr), "yt-{}-{}".format(run, i+1))
#                 )
                
#                 # browsers.append(
#                 #     PufferPlayer(NUM_BROWSERS, i+1, "{}-{}-{}".format(abr, run, i+1), "motivation-{}-2-21/{}".format(folderPart, abr), link, False)
#                 # )
#                 try:
#                     browsers[i].initVideo()
#                 except ElementClickInterceptedException:
#                     failedRuns.write("{}-{}\n".format(NUM_BROWSERS, run))
#                     print("==============RUN {}-{} HAS FAILED!!!! ============".format(NUM_BROWSERS, run))
#                     anyFailed = True
#                     break

#             if anyFailed:
#                 continue

#             for i in range(NUM_BROWSERS):
#                 browsers[i].start()

#             numDatapointsCollected = 0
#             start = time.time()
#             t = start
#             while True:
#                 for i in range(NUM_BROWSERS):
#                     browsers[i].collectData()
#                 numDatapointsCollected += 1
#                 # Exit condition
#                 if traceFile == "":
#                     print(numDatapointsCollected)
#                     if numDatapointsCollected >= numDatapointsNeeded:
#                         break
#                 else:
#                     if time.time() - t > 1:
#                         nextBw = tc.getNextBW(time.time() - start + tc.times[0])
#                         print(tc.bIdx, time.time() - start + tc.times[0], nextBw)
#                         if nextBw is None:
#                             break
#                         tc.throttleNB(nextBw)
#                         t = time.time()

#                 time.sleep(datapointsInterval)
        
#             for i in range(NUM_BROWSERS):
#                 browsers[i].f.close()
#                 browsers[i].driver.quit()
#             time.sleep(1)

#             if traceFile != "":
#                 tc.cleanNB()
#     failedRuns.close()

if __name__ == "__main__":
    link = "http://3.91.101.115:8080/player"
    abr = "bola-cubic"
    # link = ""
    # abr = "youtube"
    traceFile = "oboe/trace_394.txt_Oboe"
    numDatapointsNeeded = 60
    datapointsInterval = 0.5

    pufferPortalThreadOrchestrator(link, abr, traceFile, numDatapointsNeeded, datapointsInterval)
    # pufferPortalMain(link, abr, traceFile, numDatapointsNeeded, datapointsInterval)