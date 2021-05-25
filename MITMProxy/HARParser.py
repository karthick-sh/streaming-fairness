import json
from haralyzer import HarParser, HarPage
import sys
import pandas as pd

class YouTubeHARParser:
    def __init__(self, har_file):
        with open(har_file, 'r') as hf:
            self.har = json.loads(hf.read())
    
    def is_well_formed(self, url):
        try :
            query_string = {v.split('=')[0]: v.split('=')[1] for v in url.split('?')[-1].split('&')}
            well_formed = 'itag' in query_string
            return well_formed
        except:
            return False

    def getMediaRequests(self):
        currentHAR = self.har['log']['entries']

        filtered_har_file = []
        for entry in currentHAR:
            if 'video' in entry['response']['content']['mimeType']:# and self.is_well_formed(entry['request']['url']):
                filtered_har_file.append(entry)
        
        return filtered_har_file

    def obtain_segment_identifier(self, media_request_url):
        query_string = {v.split('=')[0]: v.split('=')[1] for v in media_request_url.split('?')[-1].split('&')}
        assert 'itag' in query_string, 'Wrongly formatted %s' % query_string
        segment_id = 'itag:{itag}_range:{range}'.format(itag=query_string[
            'itag'], range=query_string['range'])
        return segment_id

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
        parsed_requests = []

        for index, media_request in enumerate(media_requests):
            
            url = media_request['request']['url']
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
            # if body_size_byte != -1 and index not in self.already_finished_index:  # Download is finished
            #     newly_downloaded.append(parsed_entry)
            #     newly_downloaded_idx.append(index)
            #     self.already_finished_index.add(index)
            # if index >= self.already_recorded_len:
            #     self.already_recorded_len += 1
            #     newly_recorded.append(parsed_entry)
            parsed_requests.append(entry)
    
        return parsed_requests

if __name__ == "__main__":
    parser = YouTubeHARParser("test1.har")
    media_requests = parser.getMediaRequests()
    newly_recorded, newly_downloaded = parser.parseMediaRequests(media_requests)