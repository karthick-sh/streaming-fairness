import time
from subprocess import Popen
from selenium import webdriver 
from selenium.webdriver.chrome.options import Options

class MITMProxy:
    def __init__(self, port):
        self.port = port
        self.host = 'localhost'
        self.har_name = 'dump'
        self.proxy = None

    def makeNewHAR(self, har_name):
        self.har_name = har_name
    
    def startProxy(self):
        self.stdout_file = open("{}-proxy_stdout.txt".format(self.har_name), "w")
        self.stderr_file = open("{}-proxy_stderr.txt".format(self.har_name), "w")
        self.proxy = Popen([
            'mitmdump',
            '-s', './har_dump.py',
            '--set', 'hardump=./{}.har'.format(self.har_name),
            '--set', 'listen_port={}'.format(self.port)
        ], stdout=self.stdout_file, stderr=self.stderr_file)
        time.sleep(2)
        print("[MITMProxy] Started proxy at {}:{}".format(self.host, self.port))
        
    def stopProxy(self):
        if self.proxy:
            self.stdout_file.close()
            self.stderr_file.close()
            self.proxy.terminate()

def test():
    proxy = MITMProxy(9001)
    proxy.makeNewHAR("puffer")
    proxy.startProxy()
    # proxy.proxy.proxy_autoconfig_url = pathlib.Path("/home/karthick-sh/Desktop/streaming-fairness/BrowserMobProxy/pac_file_ws.pac").as_uri()

    chrome_options = Options()
    chrome_options.add_argument('--disable-application-cache')
    chrome_options.add_argument('disable-application-cache')
    chrome_options.add_argument('ignore-ssl-errors=yes')
    chrome_options.add_argument('ignore-certificate-errors')
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--ignore-ssl-errors')
    chrome_options.add_argument('--proxy-server=%s:%s' % (proxy.host, proxy.port))

    driver = webdriver.Chrome(options=chrome_options) 
    driver.create_options()
    
    driver.get("http://3.236.180.116:8080/player/")
    # driver.get("https://www.youtube.com/watch?v=QZUeW8cQQbo")
    time.sleep(10)
    # for i in range(10):
    #     media_requests = proxy.getMediaRequests()
    #     proxy.parseMediaRequests(media_requests)
    #     time.sleep(1)

    proxy.stopProxy()

if __name__ == "__main__":
    test()