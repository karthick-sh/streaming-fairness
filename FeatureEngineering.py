import numpy as np
import pandas as pd
from scipy.stats import hmean

import os
import glob
import sys
import random

random.seed = 42

def generate_ewma(alpha):
    # Common Sense
    func = lambda value_arr: pd.Series(value_arr).ewm(alpha=alpha, adjust=False).mean().values[-1]
    func.__name__ = 'exponential_weighted_moving_average_alpha_%.2f' % alpha
    return func


def generate_double_ewma(alpha, alpha_offset):
    # Taken from the Shaka Player
    if alpha + alpha_offset > 1:
        alpha_offset = 1.0 - alpha
    func = lambda value_arr: np.min([pd.Series(value_arr).ewm(alpha=alpha, adjust=False).mean().values[-1],
                                     pd.Series(value_arr).ewm(alpha=alpha + alpha_offset, adjust=False).mean().values[
                                         -1]])
    func.__name__ = 'exponential_weighted_moving_average_alpha_%.2f_alpha_offset_%.2f' % (alpha, alpha_offset)
    return func


def generate_weighted_moving_average():
    func = lambda value_arr: np.average(value_arr, weights=np.cumsum(np.arange(1, len(value_arr))))
    func.__name__ = 'weighted_moving_average'
    return func


def generate_percentile(percentile):
    # Why not use a percentile thingy
    func = lambda value_arr: np.nanpercentile(value_arr, q=percentile)
    func.__name__ = 'percentile_q_%.2f' % percentile
    return func

class FeatureEngineering:
    def __init__(self, data_dir, max_lookback, max_lookahead):
        self.max_lookback = max_lookback
        self.max_lookahead = max_lookahead

        self.throughput_estimators = [
            hmean,
            np.mean
        ]
        self.throughput_estimators += [generate_ewma(alpha) for alpha in np.linspace(0.15, 0.95, 5)]
        self.throughput_estimators += [generate_percentile(percentile) for percentile in np.linspace(0.15, 0.95, 5)]


        self.get_test_train_split(data_dir)
        # self.read_data_files(data_dir)

    def get_test_train_split(self, data_dir):
        logs = [folder for folder in os.listdir(data_dir)]

        if not os.path.exists("test_traces.txt"):
            with open("test_traces.txt", "w") as test_trace_file:
                for trace in random.sample(logs, int(0.2*len(logs))):
                    test_trace_file.write("{}\n".format(trace))

        with open("test_traces.txt") as test_trace_file:
            self.test_traces = test_trace_file.readlines()
        self.train_traces = [trace for trace in logs if trace not in self.test_traces]

    def get_test_train_data(self, data_dir):
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []

        for folder in glob.glob("{}/*".format(data_dir)):
            df = pd.read_csv("{}/raw_dataframe.csv".format(folder), index_col=0)
            # Drop unnecessary columns
            columns_to_drop = []
            for i in range(1, 11):
                columns_to_drop.append("quality_level_chosen_-{}".format(i))
            columns_to_drop.append("quality_level_chosen")
            columns_to_drop.append("quality_shift")
            df.drop(columns=columns_to_drop, inplace=True)

            # Calculate throughput
            df["throughput_byte"] = df["body_size_byte"]/df["t_download_s"]
        
            # Generate throughput based features
            for tput_estimator in self.throughput_estimators:
                    


if __name__ == "__main__":
    data_dir = "/home/karthick-sh/Desktop/reconstructing-proprietary-video-streaming-algorithms/Data/FeedbackResults/YouTube"
    max_lookback = 7
    max_lookahead = 2
    fe = FeatureEngineering(data_dir, max_lookback, max_lookahead)
