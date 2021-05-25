from browsermobproxy import Server
import psutil, time, sys
import pandas as pd
import numpy as np
import pathlib

from selenium import webdriver 
from selenium.webdriver.chrome.options import Options

class BrowserMobProxy:
    def __init__(self, port):
        # self.server = Server(path="./BrowserMobProxy/bmp/bin/browsermob-proxy", options={'port': port})
        self.server = Server(path="./BrowserMobProxy/bup/bin/browserup-proxy", options={'port': port})
        self.server.start()
        time.sleep(1)
        self.proxy = self.server.create_proxy()
        time.sleep(1)

        self.already_finished_index = set()
        self.already_recorded_len = 0

    def makeNewHAR(self, name):
        self.proxy.new_har(name)

    def stopProxy(self):
        self.server.stop()

    def obtain_segment_identifier(self, media_request_url):
        query_string = {v.split('=')[0]: v.split('=')[1] for v in media_request_url.split('?')[-1].split('&')}
        assert 'itag' in query_string, 'Wrongly formatted %s' % query_string
        segment_id = 'itag:{itag}_range:{range}'.format(itag=query_string[
            'itag'], range=query_string['range'])
        return segment_id

    def is_well_formed(self, url):
        try :
            query_string = {v.split('=')[0]: v.split('=')[1] for v in url.split('?')[-1].split('&')}
            well_formed = 'itag' in query_string
            return well_formed
        except:
            return False

    def getMediaRequests(self):
        currentHAR = self.proxy.har['log']['entries']

        filtered_har_file = []
        for entry in currentHAR:
            if 'video' in entry['response']['content']['mimeType'] and self.is_well_formed(entry['request']['url']):
                filtered_har_file.append(entry)
        
        return filtered_har_file

    def clean_url(self, media_request_url):
        # quality_level_chosen = self.video_quality_mapper.contained_in_url.map(lambda cnt_url: sum(
        #     [True if c in media_request_url else False for c in cnt_url]) > 0)
        # index = np.where(quality_level_chosen)
        # if len(index[0]) == 1:
        #     replace_str = self.video_quality_mapper.remove_segment_identifier.iloc[index[0][0]]
        #     if replace_str != 'dummy_value':
        #         media_request_url = media_request_url.replace(replace_str, '')
        return media_request_url

    def obtain_byte_start(self, media_request):
        query_string = {v.split('=')[0]: v.split('=')[1] for v in media_request[
            'request']['url'].split('?')[-1].split('&')}
        return float(query_string['range'].split('-')[0])

    def obtain_byte_end(self, media_request):
        query_string = {v.split('=')[0]: v.split('=')[1] for v in media_request[
            'request']['url'].split('?')[-1].split('&')}
        return float(query_string['range'].split('-')[1])

    def parseMediaRequests(self, media_requests):
        # newly_downloaded = [blist.sortedlist([], key=lambda parsed_entry: parsed_entry['timestamp_finish'])]
        newly_downloaded = []
        newly_downloaded_idx = []
        newly_recorded = []

        for index, media_request in enumerate(media_requests):
            
            url = media_request['request']['url']
            url = self.clean_url(url)
            startedDateTime = pd.to_datetime(media_request['startedDateTime']).timestamp()
            t_download_s = media_request['timings']['receive'] * 0.001
            t_download_s = max([t_download_s,
                                0.001])  # Sometimes the granularity of the download measurement is not enough so we set it to the lowest value
            body_size_byte = media_request['response']['bodySize']

            # if body_size_byte < 0:
            #     body_size_byte = float(self.obtain_byte_end(media_request)) - float(self.obtain_byte_start(media_request))

            bandwidth_mbit = (body_size_byte * 8e-6) / t_download_s

            parsed_entry = {
                'url': media_request['request']['url'],
                'timestamp_start': startedDateTime,
                'timestamp_finish': startedDateTime + (float(media_request['time']) / 1000),
                # This is wrong and should be timing - we fix this through the har in postprocessing
                'n_segment': self.obtain_segment_identifier(url),
                't_download_s': t_download_s,
                'body_size_byte': body_size_byte,
                'bandwidth_mbit': bandwidth_mbit,
                'byte_start': self.obtain_byte_start(media_request),
                'byte_end': self.obtain_byte_end(media_request),
                'index': index,
            }

            if body_size_byte != -1 and index not in self.already_finished_index:  # Download is finished
                newly_downloaded.append(parsed_entry)
                newly_downloaded_idx.append(index)
                self.already_finished_index.add(index)
            if index >= self.already_recorded_len:
                self.already_recorded_len += 1
                newly_recorded.append(parsed_entry)

        return newly_recorded, newly_downloaded

def killProxies():
    l = []
    for proc in psutil.process_iter():
        l.append(proc.name())
        if proc.name() == "chromedriver.exe":
            proc.kill()
            print("Killed chromedriver")
        elif proc.name() == "browsermob-proxy":
            print("BMP!!")

def test():
    proxy = BrowserMobProxy(9001)
    # proxy.proxy.proxy_autoconfig_url = pathlib.Path("/home/karthick-sh/Desktop/streaming-fairness/BrowserMobProxy/pac_file_ws.pac").as_uri()

    chrome_options = Options()
    chrome_options.add_argument('--disable-application-cache')
    chrome_options.add_argument('disable-application-cache')
    chrome_options.add_argument('ignore-ssl-errors=yes')
    chrome_options.add_argument('ignore-certificate-errors')
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--ignore-ssl-errors')
    chrome_options.add_argument('--proxy-server={}'.format(proxy.proxy.proxy))

    driver = webdriver.Chrome(options=chrome_options) 
    driver.create_options()

    proxy.makeNewHAR("test1")
    driver.get("http://18.206.136.34:8080/player/")

    time.sleep(1000)
    # for i in range(10):
    #     media_requests = proxy.getMediaRequests()
    #     proxy.parseMediaRequests(media_requests)
    #     time.sleep(1)

    proxy.stopProxy()

if __name__ == "__main__":
    test()