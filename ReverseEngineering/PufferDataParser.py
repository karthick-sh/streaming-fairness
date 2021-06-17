import json
import sys
import pandas as pd
import time
import os
import random
import glob
import numpy as np
from scipy.stats import hmean
from scipy.stats import sem
from collections import defaultdict, deque

random.seed(15689)


def generate_ewma(alpha):
    # Common Sense
    func = lambda value_arr: pd.Series(value_arr).ewm(alpha=alpha, adjust=False).mean().values[-1]
    func.__name__ = 'exponential_weighted_moving_average_alpha_%.2f' % alpha
    return func
def generate_percentile(percentile):
    # Why not use a percentile thingy
    func = lambda value_arr: np.nanpercentile(value_arr, q=percentile)
    func.__name__ = 'percentile_q_%.2f' % percentile
    return func


class PufferDataParser:
    def __init__(self, dataDir, abr, options):
        self.options = options
        self.abr = abr
        self.tput_estimators = [hmean, np.mean]
        self.tput_estimators += [generate_ewma(alpha) for alpha in np.linspace(0.15, 0.95, 5)]
        self.tput_estimators += [generate_percentile(p) for p in np.linspace(0.15, 0.95, 5)]

        # Use below for my custom data collected 
        self.readLogFiles(dataDir)


    def trainTestSplit(self, l, percentTest=0.2):
        # Custom Split
        test_set = []
        train_set = []

        # for traceVideoPair in l:
        #     if "FCC" in traceVideoPair:
        #         test_set.append(traceVideoPair)
        #     else:
        #         train_set.append(traceVideoPair)
        # print(len(test_set), len(train_set), len(l))
        
        # return train_set, test_set

        # Random Split
        numTest = int(len(l)*percentTest)
        random.shuffle(l)

        print(numTest)

        return l[:-numTest], l[-numTest:]

    def filterHAREvents(self, har, bitrateRanges):
        filtered_har = defaultdict(list)
        byteLengthRanges = defaultdict(list)
        for msg in har["log"]["messages"]:
            if msg["type"] == "server-video" and msg["byteOffset"] == 0 and msg["format"].split("-")[0] in bitrateRanges:
                byteLengthRanges[msg["format"].split("-")[0]].append(msg["totalByteLength"])
                filtered_har["server-video"].append(msg)
        for res in byteLengthRanges:
            byteLengthRanges[res] = {
                "lower": min(byteLengthRanges[res]),
                "upper": max(byteLengthRanges[res])
            }
        return filtered_har, byteLengthRanges

    def getBitrateRanges(self, df):
        bitrateRanges = defaultdict(list)
        for idx, row in df.iterrows():
            bitrateRanges[str(row["resolution"])].append(float(row["bitrate"]))
        
        for res in bitrateRanges:
            bitrateRanges[res] = {
                "lower": min(bitrateRanges[res]),
                "upper": max(bitrateRanges[res])
            }

        return bitrateRanges

    def readLogFiles(self, dataDir):
        self.X_train = {}
        self.y_train = {}
        self.X_test_files = {}
        self.y_test_files = {}
        self.bitrateCutoffs = {}

        self.X_train[self.abr] = []
        self.y_train[self.abr] = []
        self.X_test_files[self.abr] = {}
        self.y_test_files[self.abr] = {}
        self.bitrateCutoffs[self.abr] = {}
        
        trainPairs, testPairs = self.trainTestSplit(os.listdir(dataDir))

        for idx, trace in enumerate(glob.glob("{}/*".format(dataDir))):
            trace_name = trace.split("/")[-1]
            player_data_df = trace+"/player_data-{}-0".format(self.abr)
            proxy_har = trace+"/proxy_har.har"

            print(trace_name)

            player_data_df = pd.read_csv(player_data_df).sort_values(by="elapsedTime")
            player_data_df.loc[player_data_df["bitrate"] < 0, "bitrate"] = 0
            bitrateRanges = self.getBitrateRanges(player_data_df)

            with open(proxy_har) as har_file:
                proxy_har = json.loads(har_file.read())
            proxy_har, byteLengthRanges = self.filterHAREvents(proxy_har, bitrateRanges)
            
            featuresForTrace, bitrates = self.getAllFeatures(player_data_df, proxy_har, byteLengthRanges, bitrateRanges)

            if trace_name in testPairs:
                self.X_test_files[self.abr][trace_name] = featuresForTrace
                self.y_test_files[self.abr][trace_name] = bitrates
            else:
                self.X_train[self.abr] += featuresForTrace
                self.y_train[self.abr] += bitrates

            # Build bitrate cutoff for video
            self.bitrateCutoffs[self.abr] = {}
            for res in np.unique(player_data_df["resolution"]):
                vidHeight = int(res.split("x")[1])
                minBR = player_data_df[player_data_df["resolution"] == res]["bitrate"].min()
                maxBR = player_data_df[player_data_df["resolution"] == res]["bitrate"].max()
                avgBR = (minBR+maxBR) / 2 / 1000
                self.bitrateCutoffs[self.abr][vidHeight] = avgBR

        print("Train:", len(self.X_train[self.abr]), len(self.y_train[self.abr]))
        print("Feature Length: ", len(self.X_train[self.abr][0]))
        print("Test Files: ", len(self.X_test_files[self.abr]), len(self.y_test_files[self.abr]))
        # break
    
    def getAllFeatures(self, player_data_df, har, byteLengthRanges, bitrateRanges):
        features = []
        bitrate = []
        for idx, msg in enumerate(har["server-video"]):
            filtered_df = player_data_df[player_data_df["timestamp"] <= msg["currentTimestamp"]]

            feature = []
            # Bandwidth (in a given window size) with estimators
            if "bw" in self.options:
                bw_vals = filtered_df.tail(self.options["bw"]["windowSize"])["currentBW"].values.tolist()
                bw_vals.reverse()
                for _ in range(len(bw_vals), self.options["bw"]["windowSize"]):
                    bw_vals.append(0)
                feature += bw_vals
                # for estimator in self.tput_estimators:
                #     feature.append(estimator(bw_vals))

            self.firstBufferIdx = len(feature)

            # Buffer level (in a given window size)
            if "buf" in self.options:
                buf_vals = filtered_df.tail(self.options["buf"]["windowSize"])["bufferHealth"].values.tolist()
                buf_vals.reverse()
                for _ in range(len(buf_vals), self.options["buf"]["windowSize"]):
                    buf_vals.append(0)
                feature += buf_vals

            # Previous packet mbit
            if "prevMbit" in self.options:
                filtered_prev_mbit = raw_df["bandwidth_mbit"].iloc[max(0,idx-self.options["prevMbit"]["windowSize"]):idx].values.tolist()
                # filtered_prev_mbit.reverse()
                # for _ in range(len(filtered_prev_mbit), self.options["prevMbit"]["windowSize"]):
                #     filtered_prev_mbit.append(0)
                # feature += filtered_prev_mbit

            # Previous packet bitrate
            if "prevBitrate" in self.options:
                br_vals = deque()
                for m in har["server-video"][max(idx-self.options["prevBitrate"]["windowSize"], 0):idx]:
                    m_format = m["format"].split("-")[0]
                    if byteLengthRanges[m_format]["upper"] == byteLengthRanges[m_format]["lower"]:
                        byte_length_ratio = 1
                    else:
                        byte_length_ratio = (m["totalByteLength"] - byteLengthRanges[m_format]["lower"]) / (byteLengthRanges[m_format]["upper"] - byteLengthRanges[m_format]["lower"])

                    extrapolated_br = byte_length_ratio * (bitrateRanges[m_format]["upper"] - bitrateRanges[m_format]["lower"]) +  bitrateRanges[m_format]["lower"]
                    br_vals.appendleft(extrapolated_br / 1000)
                    
                for _ in range(len(br_vals), self.options["prevBitrate"]["windowSize"]):
                    br_vals.append(0)
                feature += list(br_vals)

            # Current bitrate
            m_format = msg["format"].split("-")[0]
            if byteLengthRanges[m_format]["upper"] == byteLengthRanges[m_format]["lower"]:
                byte_length_ratio = 1
            else:
                byte_length_ratio = (msg["totalByteLength"] - byteLengthRanges[m_format]["lower"]) / (byteLengthRanges[m_format]["upper"] - byteLengthRanges[m_format]["lower"])

            extrapolated_br = byte_length_ratio * (bitrateRanges[m_format]["upper"] - bitrateRanges[m_format]["lower"]) +  bitrateRanges[m_format]["lower"]
            bitrate.append(extrapolated_br/1000)
                    
            features.append(feature)
        return features, bitrate

if __name__ == "__main__":
    options = {
        "bw": {"windowSize": 7},
        "buf": {"windowSize": 7},
        "prevBitrate": {"windowSize": 7},
    }
    parser = PufferDataParser("./Data/ReverseEngineering/mpc", "mpc", options)
    # media_requests = parser.getMediaRequests()
    # print(len(media_requests))
    # parsed_requests = parser.parseMediaRequests(media_requests)