class YouTubePlayer:
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
