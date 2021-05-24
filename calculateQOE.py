import pandas as pd
import glob, sys, math
from datetime import datetime
import numpy as np
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

LAMBDA = 0.5
ALPHA = 0.1
BETA = 0.1

THETA_BIG_SCREEN =  2.1 * 10**-3
THETA_SM_SCREEN =  3 * 10**-3

def f(b):
    return math.exp(-LAMBDA*b)

def VQ(theta, r):
    return 1 - math.exp(-theta*r)

def calcAverageQOE(df, bitrate_col):
    QoEs = []
    prev = None
    for idx,row in df.iterrows():
        if idx == 0:
            prev = row
            continue

        b = float(row["buffer"])
        r = float(row[bitrate_col]) * 1000

        if r < 0:
            r = 0

        prev_r = prev[bitrate_col]
        if prev_r < 0:
            prev_r = 0

        QoE = VQ(THETA_BIG_SCREEN, r) - BETA * f(b) - ALPHA*(VQ(THETA_BIG_SCREEN, r) - VQ(THETA_BIG_SCREEN, prev_r))
        if QoE < 0:
            QoE = 0
        QoEs.append(QoE)
        prev = row
    return np.average(QoEs)

if __name__ == "__main__":
    dataDir = "Data/atc-data-qoe"
    originalAvgQOEs = []
    predictedAvgQOEs = []
    for abr in glob.glob("{}/*".format(dataDir)):
        abrName = abr.split("/")[-1]
        if abrName not in ["Twitch", "TubiTV"]:
            continue
        for trace in glob.glob("{}/*".format(abr)):
            df = pd.read_csv(trace)
            traceNum = trace.split("/")[-1]
            originalAvgQOEs.append(calcAverageQOE(df, "original_bitrate"))
            predictedAvgQOEs.append(calcAverageQOE(df, "predicted_bitrate"))
            # print("{} - Org: {}, Pred:{}".format(traceNum, calcAverageQOE(df, "original_bitrate"), calcAverageQOE(df, "predicted_bitrate")))

        _, ax = plt.subplots()
        sns.ecdfplot(originalAvgQOEs, ax=ax, label="Original QOE")
        sns.ecdfplot(predictedAvgQOEs, ax=ax, label="Predicted QOE")
        ax.legend()
        # ax.set_linewidth(5)
        ax.set_ylabel("CDF")
        ax.set_xlim(left=0, right=1)
        ax.set_xlabel("Average QOE")
        ax.set_title("Original/Predicted Average QOE for {}".format(abrName))

    plt.show()
