import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.neighbors import NearestNeighbors
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest

import numpy as np
import pandas as pd
from collections import deque
from scipy.stats import hmean

import glob, sys, re

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

class YouTubeDataParser:
    def __init__(self, dataDir, options):
        self.fccTraceIdx = []
        self.genItagToBitrate()
        self.initAllFeatures()
        self.options = options
        self.tput_estimators = [hmean, np.mean]
        self.tput_estimators += [generate_ewma(alpha) for alpha in np.linspace(0.15, 0.95, 5)]
        self.tput_estimators += [generate_percentile(p) for p in np.linspace(0.15, 0.95, 5)]

        # Use below for my custom data collected 
        self.readLogFiles(dataDir)

    def readLogFiles(self, dataDir):
        for logFile in glob.glob("{}/*".format(dataDir)):
            if self.isDownloadLog(logFile):
                continue
            downloadLogFileDf = pd.read_csv(logFile.replace("yt-", "yt-downloaded-"))
            with open(logFile) as lf:
                lf.readline()
                for idx, line in enumerate(lf.readlines()):
                    self.getAllFeatures(line, self.isTestFile(logFile), idx, downloadLogFileDf)
                if self.isTestFile(logFile):
                    if 'http' in logFile:
                        self.fccTraceIdx.append(len(self.X_test_files))
                    self.X_test_files.append(self.X_test)
                    self.y_test_files.append(self.y_test)
                    self.y_test_files_raw.append(self.y_test_raw)
                    self.X_test = []
                    self.y_test = []
                    self.y_test_raw = []

        print("Train:", len(self.X_train), len(self.y_train))
        print("Test Files: ", len(self.X_test_files), len(self.y_test_files))

    def initAllFeatures(self):
        self.X_train = []
        self.y_train = []
        self.y_train_raw = []
        self.X_test_files = []
        self.y_test_files = []
        self.y_test_files_raw = []
        self.X_test = []
        self.y_test = []
        self.y_test_raw = []

        self.video_info_range_mapper = pd.read_csv("../understanding-video-streaming-in-the-wild/Data/VideoInformation/YouTube_Info/QZUeW8cQQbo_video_info_range_mapper", index_col=0)
        self.bitrateCutoffs = {}
        uniqueBitrates = self.video_info_range_mapper.drop_duplicates(subset=["bitrate"])
        for itag in self.itagToBitrate:
            itagValue = float(itag.split("=")[-1])
            # print(uniqueBitrates[uniqueBitrates["itag"] == itagValue])
            # continue
            minBitrateForItag = uniqueBitrates[uniqueBitrates["itag"] == itagValue]["bitrate"].min()
            maxBitrateForItag = uniqueBitrates[uniqueBitrates["itag"] == itagValue]["bitrate"].max()
            avgBitrate = (minBitrateForItag+maxBitrateForItag) / 2
            if not np.isnan(avgBitrate):
                # print("{}, {}, {}, {}".format(itag, self.itagToBitrate[itag], minBitrateForItag, maxBitrateForItag))
                # print(avgBitrate)
                self.bitrateCutoffs[int(self.itagToBitrate[itag])] = avgBitrate / 1000000

        self.N_SEGMENT_IDX = 10
        self.BYTE_END_IDX = 15
        self.BYTE_START_IDX = 14
        self.URL_IDX = 16
        self.TIMESTAMP_IDX = 5

    def getPastNBitrates(self, n, df, currIdx, raw=False):
        pastNBitrates = deque(maxlen=n)
        for k in range(currIdx-1, currIdx-n-1, -1):
            if k < 0:
                break
            if not raw:
                try:
                    pastNBitrates.append(self.itagToBitrate[df["n_segment"].iloc[k].split("_range")[0].replace(":", "=")])
                except IndexError:
                    pastNBitrates.append('0')
            else:
                try:
                    itagURL = df["n_segment"].iloc[k].split("_range")[0].replace(":", "=")
                    itagValue = float(itagURL.split("=")[-1])
                    upperByteRange = float(df["n_segment"].iloc[k].split("_range:")[1].split("-")[-1])
                    itagFiltered = self.video_info_range_mapper[self.video_info_range_mapper["itag"] == itagValue]
                    brFiltered = itagFiltered[itagFiltered["byterange"] >= upperByteRange]
                    bitrateRaw = float(brFiltered[brFiltered["byterange"] == brFiltered["byterange"].min()]["bitrate"])
                    pastNBitrates.append(bitrateRaw / 1000000)
                except (IndexError, TypeError):
                    pastNBitrates.append(0)

        for _ in range(n-len(pastNBitrates)):
            appendVal = '0'
            if raw:
                appendVal = 0
            pastNBitrates.append(appendVal)

        
        return pastNBitrates
    
    def getPastNPacketBandwidth(self, n, df, currIdx):
        pastNPacketBandwidth = deque(maxlen=n)
        for k in range(currIdx-1, currIdx-n-1, -1):
            if k < 0:
                break
            try:
                pastNPacketBandwidth.append(float(df["bandwidth_mbit"].iloc[k]))
            except IndexError:
                pastNPacketBandwidth.append(0)

        for _ in range(n-len(pastNPacketBandwidth)):
            pastNPacketBandwidth.append(0)
        
        return pastNPacketBandwidth

    def getPastNPacketThroughput(self, n, df, currIdx):
        pastNPacketThroughput = deque(maxlen=n)
        for k in range(currIdx-1, currIdx-n-1, -1):
            if k < 0:
                break
            try:
                tput = (float(df["byte_end"].iloc[k]) - float(df["byte_start"].iloc[k])) / 1000000 / float(df["t_download_s"].iloc[k])
                pastNPacketThroughput.append(tput)
            except IndexError:
                pastNPacketThroughput.append(0)

        for _ in range(n-len(pastNPacketThroughput)):
            pastNPacketThroughput.append(0)

        return pastNPacketThroughput

    def getPastNPacketDownload(self, n, df, currIdx):
        pastNPacketDownload = deque(maxlen=n)
        for k in range(currIdx-1, currIdx-n-1, -1):
            if k < 0:
                break
            try:
                pastNPacketDownload.append(float(df["t_download_s"].iloc[k]))
            except IndexError:
                pastNPacketDownload.append(0)

        for _ in range(n-len(pastNPacketDownload)):
            pastNPacketDownload.append(0)
        
        return pastNPacketDownload

    def getAllFeatures(self, line, isTest, idx, downloadLogFileDf):
        bufLine = ""
        bwLine = ""

        matches = re.finditer(r"(\[.*?\])", line)
        for matchNum, match in enumerate(matches):
            if matchNum == 0:
                bufLine = match.group()
            elif matchNum == 1:
                bwLine = match.group()

        line = line.replace(bwLine, "")
        line = line.replace(bufLine, "")
        lineSplits = line.split(",")
        row = []

        # Bandwidth (in a given window size) with estimators
        if "bw" in self.options:
            bw_row = []
            c = 0
            for val in bwLine.split(","):
                bw_row.append(float(val.strip("[] ")))
                c += 1
                if c == self.options["bw"]["windowSize"]:
                    break
            bw_pred = []
            for estimator in self.tput_estimators:
                bw_pred.append(estimator(bw_row))
            row += bw_row
            row += bw_pred

        # Buffer level (in a given window size)
        if "buf" in self.options:
            c = 0
            for val in bufLine.split(","):
                row.append(float(val.strip("[] ")))
                c += 1
                if c == self.options["buf"]["windowSize"]:
                    break

        # Chunk size (Maybe shouldn't be using this since this is related to what we want to predict)
        # if "chunkSize" in self.options:
        #     row.append(float(lineSplits[self.BYTE_END_IDX]) - float(lineSplits[self.BYTE_START_IDX]))

        # Previous packet mbit (to get from downloaded file)
        if "prevMbit" in self.options:
            for val in self.getPastNPacketBandwidth(self.options["prevMbit"]["windowSize"], downloadLogFileDf, idx):
                row.append(val)

        # Previous packet download time
        if "prevDownloadTime" in self.options:
            for val in self.getPastNPacketDownload(self.options["prevDownloadTime"]["windowSize"], downloadLogFileDf, idx):
                row.append(val)

        # Previous packet bitrate
        if "prevBitrate" in self.options:
            for val in self.getPastNBitrates(self.options["prevBitrate"]["windowSize"], downloadLogFileDf, idx, raw=True):
                row.append(val)

        # Previous packet throughput with estimators
        if "prevThroughput" in self.options:
            tputs = list(self.getPastNPacketThroughput(self.options["prevThroughput"]["windowSize"], downloadLogFileDf, idx))
            row += tputs
            for estimator in self.tput_estimators:
                row.append(estimator(tputs))

        # Current bitrate
        itagNSegment = lineSplits[self.N_SEGMENT_IDX].split("_range")[0].replace(":", "=")
        itagURL = re.search(r"&itag=[0-9]*&", lineSplits[self.URL_IDX]).group(0).strip("&")

        if itagNSegment != itagURL:
            sys.exit("ITAG N SEG != URL")

        bitrate = self.itagToBitrate[itagURL]

        # Get raw bitrate value
        itagValue = float(itagURL.split("=")[-1])
        timestamp = float(lineSplits[self.TIMESTAMP_IDX])
        itagFiltered = self.video_info_range_mapper[self.video_info_range_mapper["itag"] == itagValue]
        tsFiltered = itagFiltered[itagFiltered["time_s"] >= timestamp]
        bitrateRaw = float(tsFiltered[tsFiltered["time_s"] == tsFiltered["time_s"].min()]["bitrate"]) / 1000000

        if isTest:
            self.X_test.append(row)
            self.y_test.append(bitrate)
            self.y_test_raw.append(bitrateRaw)
        else:
            self.X_train.append(row)
            self.y_train.append(bitrate)
            self.y_train_raw.append(bitrateRaw)
        
    def fixLogFileLines(self, dataDir):
        for logFile in glob.glob("{}/*".format(dataDir)):
            if self.isDownloadLog(logFile):
                continue
            
            fname = logFile.split("/")[-1]
            print(fname)
            with open(logFile) as lf:
                header = lf.readline().strip()
                header += ",url\n"
                with open("YouTube with BMP/run-1-all-fixed/"+fname, "w") as of:
                    of.write(header)
                    for line in lf.readlines():
                        of.write(line)

    def isDownloadLog(self, log):
        return log.split("/")[-1][:4] == "yt-d"
    
    def isTestFile(self, log):
        return log.split("/")[-1].split("-")[1] in [
            '1001','1003','1058','2124','0749',
            'trace_79', 'trace_230', 'trace_244', 'trace_333', 'trace_394',
            'trace_1000279_http',
        ]

    def genItagToBitrate(self):
        self.itagToBitrate = {}
        qfile = pd.read_csv("../understanding-video-streaming-in-the-wild/Data/VideoInformation/YouTube_Info/QZUeW8cQQbo_video_quality_mapper")
        
        for idx, row in qfile.iterrows():
            for itag in row['contained_in_url'].strip("[]").split(","):
                self.itagToBitrate[itag.strip().strip("'")] = row['resolution'].split("x")[1]
    

class TrainModel:
    def __init__(self, data):
        self.X_train = data.X_train
        self.y_train = data.y_train
        self.y_train_raw = data.y_train_raw

        assert(len(self.y_train) == len(self.y_train_raw))

        self.X_test_files = data.X_test_files
        self.y_test_files = data.y_test_files
        self.y_test_files_raw = data.y_test_files_raw

        self.fccTraceIdx = data.fccTraceIdx
        self.bitrateCutoffs = data.bitrateCutoffs

    def classificationSuite(self):
        accuracies = {}

        # clfs = [
        #     {
        #         "algorithm": "RandomForest",
        #         "clf": RandomForestClassifier(criterion="entropy", n_estimators=100),
        #     },
        #     {
        #         "algorithm": "SVM",
        #         "clf": SVC(C=10),
        #     },
        #     {
        #         "algorithm": "DecisionTree",
        #         "clf": DecisionTreeClassifier(),
        #     }
        # ]

        # for clf in clfs:
        #     print("Training", clf["algorithm"])
        #     clf["clf"].fit(self.X_train, self.y_train)
        #     accuracies[clf["algorithm"]] = self.testClassifier(clf["clf"])

        # print("Training 1-NearestNeighbor")
        # accuracies["1-NN"] = self.testNearestNeighbor(1)
        # print("Training 5-NearestNeighbor")
        # accuracies["5-NN"] = self.testNearestNeighbor(5)

        print("Training PolynomialRegression")
        accuracies["LinReg"] = self.testLinearRegression(degree=2)

        print()
        return accuracies

    def testLinearRegression(self, degree):
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(self.X_train)
        poly.fit(X_train_poly, self.y_train_raw)
        print("Done with poly.fit")
        reg = LinearRegression()
        reg.fit(X_train_poly, self.y_train_raw)
        print("Done with reg.fit")

        accuracies = []
        rmses = []
 
        for fidx, X_test in enumerate(self.X_test_files):
            y_test_raw = self.y_test_files_raw[fidx]
            y_test = self.y_test_files[fidx]
            X_test_poly = poly.fit_transform(X_test)
            y_pred = reg.predict(X_test_poly)
            
            accuracies.append(self.accuracyWithCutoff(y_test_raw, y_pred, X_test, fidx))
            rmses.append(self.rmse(y_test_raw, y_pred))

        print("Average Test RMSE: ", np.mean(rmses))
        print("Average Test Accuracy: ", np.mean(accuracies))
        y_train_pred = reg.predict(X_train_poly)
        print("Average Train RMSE: ", self.rmse(self.y_train_raw, y_train_pred))

        
        return accuracies

    def accuracyWithCutoff(self, org, pred, x_test, fidx):
        correct = 0
        with open("Data/run-2-3-mixed-test/{}.txt".format(fidx), "w") as of:
            of.write("buffer,original_bitrate,predicted_bitrate\n")
            for i in range(len(org)):
                predBitrate = max(self.bitrateCutoffs)
                for k in sorted(self.bitrateCutoffs):
                    if pred[i] <= self.bitrateCutoffs[k]:
                        predBitrate = k
                        break
                orgBitrate = max(self.bitrateCutoffs)
                for k in sorted(self.bitrateCutoffs):
                    if org[i] <= self.bitrateCutoffs[k]:
                        orgBitrate = k
                        break
                # print(org[i], pred[i], predBitrate, orgBitrate)
                # print(x_test)
                of.write("{},{},{}\n".format(x_test[i][19], org[i], pred[i]))
                if orgBitrate == predBitrate:
                    correct += 1

            
        
        # Save details to file
        return correct/len(org)

    def rmse(self, org, pred):
        rmse = 0
        for i in range(len(org)):
            rmse += (org[i]-pred[i])**2
        return (rmse/len(org))**0.5

    def testNearestNeighbor(self, k):
        clf = NearestNeighbors()
        clf.fit(self.X_train)

        accuracies = []
        for idx, X_test in enumerate(self.X_test_files):
            y_test = self.y_test_files[idx]

            total = 0
            predictions = []
            actual = []
            correct = 0

            for idx, x in enumerate(X_test):
                nidx = clf.kneighbors(np.array(x).reshape(1, -1))
                
                real = y_test[idx]
                pred = [self.y_train[p] for p in nidx[1][0]]

                actual.append(real[0])
                if k == 1:
                    predictions.append(pred[0])

                    if real == pred[0]:
                        correct += 1
                elif k == 5:
                    mostFreq = most_frequent(pred)
                    predictions.append(mostFreq)
                    if real == mostFreq:
                        correct += 1
                else:
                    sys.exit("PANIC! Wrong k value in NN")

                total += 1
            accuracies.append(correct/total)

            # # Plot the two arrays
            # plt.figure()
            # plt.title("Predictions with NN (k={})".format(k))
            # plt.plot(predictions, "r", actual, "b")
            # plt.legend(["predictions", "actual"])
            # plt.xlabel("Time (s)")
            # plt.ylabel("Bitrate")
            # plt.show()

        return accuracies

    def testClassifier(self, clf):
        accuracies = []
        for fidx, X_test in enumerate(self.X_test_files):
            y_test = self.y_test_files[fidx]
            predictions = []
            actual = []
            correct = 0

            for idx, x in enumerate(X_test):
                predictions.append(int(clf.predict(np.array(x).reshape(1, -1))))
                actual.append(int(y_test[idx]))

                if predictions[idx] == actual[idx]:
                    correct += 1
            
            # if fidx in self.fccTraceIdx:
            #     # Plot the two arrays
            #     plt.figure()
            #     plt.title("Predictions with {}".format(clf))
            #     plt.plot(predictions, "r", actual, "b")
            #     plt.legend(["predictions", "actual"])
            #     plt.xlabel("Time (s)")
            #     plt.ylabel("Bitrate (144p|240p|360p|480p|720p|1080p)")
            #     plt.show()


            #     print("Accuracy for {} is {}".format(clf, correct/len(predictions)))
            accuracies.append(correct/len(predictions))
        # print()
        # print("Mean accuracy for {} is {}".format(clf, np.mean(accuracies)))
        return accuracies

    def trainRandomForest(self):
        print("======Random Forest======")
        clf = RandomForestClassifier(criterion="entropy", n_estimators=100)
        # X_train, X_test, y_train, y_test = train_test_split(self.X.to_numpy(), self.Y.to_numpy(), test_size=0.20, random_state=42)

        print(self.X.head())
        X_train = self.X.to_numpy()
        y_train = self.Y.to_numpy()
        print(X_train)
        clf.fit(X_train, y_train)
        # total = 0
        # correct = 0
        # for idx, x in enumerate(X_test):
        #     pred = clf.predict(x.reshape(1, -1))
        #     if pred[0] == y_test[idx]:
        #         correct += 1
        #     total += 1
        # print(correct/total)
        # scores = cross_val_score(clf, X_train, y_train, cv=5)
        # print(np.mean(scores), scores)

        columns = [col for col in self.X.columns]

        importances = clf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                    axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(X_train.shape[1]):
            print("%d. feature %d [%s] (%f)" % (f + 1, indices[f], columns[indices[f]], importances[indices[f]]))

        # Plot the impurity-based feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X_train.shape[1]), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(X_train.shape[1]), indices)
        plt.xlim([-1, X_train.shape[1]])
        plt.show()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def most_frequent(l): 
    counter = 0
    num = l[0] 
      
    for i in l: 
        curr_frequency = l.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
  
    return num 

def main():
    featureGroups =[
        {
            "bw": {"windowSize": 7},
            "buf": {"windowSize": 7},
            "prevDownloadTime": {"windowSize": 7},
            "prevBitrate": {"windowSize": 7},
            "prevMbit": {"windowSize": 7},
            "prevThroughput": {"windowSize": 7}
        },
    ]

    featureGroupAccuracies = []
    xAxisLabels = []
    for idx, fg in enumerate(featureGroups):
        print("==========Feature Group {}==========".format(idx))
        data = YouTubeDataParser("./Data/run-2-3-mixed", fg)
        suite = TrainModel(data)
        accuracies = suite.classificationSuite()

        for key in accuracies:
            arr = np.array(accuracies[key], dtype=np.float64)
            print("{}: {} {} {}".format(key, np.mean(arr), np.min(arr), np.max(arr)))
            featureGroupAccuracies.append(accuracies[key])
            xAxisLabels.append("FG-{}[{}]".format(idx, key))

    _, ax1 = plt.subplots()
    bplot = ax1.boxplot(featureGroupAccuracies,patch_artist=True,showmeans=True)
    ax1.set_xticklabels(xAxisLabels)
    ax1.set_title('YouTube ABR Modelling')
    ax1.set_xlabel('Feature Groups')
    ax1.set_ylabel('Test Accuracy')

    # Add Colors
    COLORS = [
        {
            "color": "pink",
            "algorithm": "RandomForest",
        },
        {
            "color": 'mediumorchid',
            "algorithm": "SVM",
        },
        {
            "color": 'lightgreen',
            "algorithm": "DecisionTree",
        },
        {
            "color": 'lemonchiffon',
            "algorithm": "1-NN",
        },
        {
            "color": 'paleturquoise',
            "algorithm": "5-NN",
        },
        {
            "color": "red",
            "algorithm": "LinReg"
        }
    ]
    cidx = 0

    boxColors = []
    for _ in featureGroupAccuracies:
        boxColors.append(COLORS[cidx]["color"])
        cidx += 1
        cidx %= len(COLORS)

    for patch, color in zip(bplot['boxes'], boxColors):
        patch.set_facecolor(color)
    for meanPoint in bplot['means']:
        meanPoint.set_markerfacecolor("k")
        meanPoint.set_markeredgecolor("k")

    # Add legend
    legend_elements = [Patch(facecolor=c["color"], label=c["algorithm"]) for c in COLORS]
    ax1.legend(handles=legend_elements, loc='best')

    plt.show()

if __name__ == "__main__":
    main()
    
