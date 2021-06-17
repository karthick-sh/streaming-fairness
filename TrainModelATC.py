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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import IsolationForest
import pickle

import numpy as np
import pandas as pd
from collections import deque
from scipy.stats import hmean
from scipy.stats import sem

from sklearn.metrics import f1_score

import glob, sys, re, random, os

random.seed(42)

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

class ATCDataParser:
    def __init__(self, dataDir, options):
        self.options = options
        self.tput_estimators = [hmean, np.mean]
        self.tput_estimators += [generate_ewma(alpha) for alpha in np.linspace(0.15, 0.95, 5)]
        self.tput_estimators += [generate_percentile(p) for p in np.linspace(0.15, 0.95, 5)]

        # Use below for my custom data collected 
        self.readLogFiles(dataDir)

    def trainTestSplit(self, l, percentTest=0.2):
        # Custom Split
        test_set = []
        train_set = []

        for traceVideoPair in l:
            if "FCC" in traceVideoPair:
                test_set.append(traceVideoPair)
            else:
                train_set.append(traceVideoPair)
        print(len(test_set), len(train_set), len(l))
        
        return train_set, test_set

        # Random Split
        # numTest = int(len(l)*percentTest)
        # random.shuffle(l)

        # return l[:-numTest], l[-numTest:]

    def readLogFiles(self, dataDir):
        self.X_train = {}
        self.y_train = {}
        self.X_test_files = {}
        self.y_test_files = {}
        self.bitrateCutoffs = {}
        
        for videoProvider in glob.glob("{}/FeedbackResults/*".format(dataDir)):
            videoProviderName = videoProvider.split("/")[-1]
            if videoProviderName not in ["YouTube"]:
                continue
            print(videoProviderName)

            # Train test split
            trainPairs, testPairs = self.trainTestSplit(os.listdir(videoProvider))

            self.X_train[videoProviderName] = []
            self.y_train[videoProviderName] = []
            self.X_test_files[videoProviderName] = {}
            self.y_test_files[videoProviderName] = {}
            self.bitrateCutoffs[videoProviderName] = {}

            for traceVideoPair in glob.glob("{}/*".format(videoProvider)):
                raw_df = traceVideoPair + "/raw_dataframe.csv"
                local_client_state_logger = traceVideoPair + "/local_client_state_logger.csv"
                throttle_logging = traceVideoPair + "/throttle_logging.tc"
                if not os.path.exists(raw_df) or not os.path.exists(local_client_state_logger) or not os.path.exists(throttle_logging):
                    continue

                raw_df = pd.read_csv(raw_df, index_col=0).sort_values(by="timestamp_start")
                local_client_state_logger = pd.read_csv(local_client_state_logger, index_col=0).sort_values(by="timestamp_s")
                throttle_logging = pd.read_csv(throttle_logging, sep="\t", names=["timestamp_s", "bw"]).sort_values(by="timestamp_s")

                local_client_state_logger['buffer_level'] = local_client_state_logger['buffered_until'] - local_client_state_logger[
                    'played_until']
                raw_df["throughput"] = (raw_df["byte_end"] - raw_df["byte_start"]) / 1000000 / raw_df["t_download_s"] 
                
                featuresForPair, bitrates = self.getAllFeatures(raw_df, local_client_state_logger, throttle_logging)
                if traceVideoPair.split("/")[-1] in testPairs:
                    self.X_test_files[videoProviderName][traceVideoPair.split("/")[-1]] = featuresForPair
                    self.y_test_files[videoProviderName][traceVideoPair.split("/")[-1]] = bitrates
                else:
                    self.X_train[videoProviderName] += featuresForPair
                    self.y_train[videoProviderName] += bitrates

                # Build bitrate cutoff for video
                videoName = self.getVideoName(traceVideoPair.split("/")[-1])
                if videoName not in self.bitrateCutoffs[videoProviderName]:
                    self.bitrateCutoffs[videoProviderName][videoName] = {}
                    video_info_df = pd.read_csv("{}/Video_Info/{}_Info/{}_video_info".format(dataDir, videoProviderName, videoName))
                    for col in video_info_df:
                        if "bitrate" in col:
                            videoHeight = int(col.split("_")[0].split("x")[1])
                            minBR = video_info_df[col].min()
                            maxBR = video_info_df[col].max()
                            avgBR = (minBR+maxBR) / 2 / 1000000
                            self.bitrateCutoffs[videoProviderName][videoName][videoHeight] = avgBR

            print("Train:", len(self.X_train[videoProviderName]), len(self.y_train[videoProviderName]))
            print("Test Files: ", len(self.X_test_files[videoProviderName]), len(self.y_test_files[videoProviderName]))
            print("Num videos: ", len(self.bitrateCutoffs[videoProviderName]))
            # break

    def getAllFeatures(self, raw_df, local_client_state, tc_log):
        features = []
        bitrate = []
        for idx, row in raw_df.iterrows():
            feature = []
            # Bandwidth (in a given window size) with estimators
            if "bw" in self.options:
                filtered_tc = tc_log[tc_log["timestamp_s"] <= row["timestamp_start"]].tail(self.options["bw"]["windowSize"])["bw"].values.tolist()
                filtered_tc.reverse()
                for _ in range(len(filtered_tc), self.options["bw"]["windowSize"]):
                    filtered_tc.append(0)
                feature += filtered_tc
                # for estimator in self.tput_estimators:
                #     feature.append(estimator(filtered_tc))

            self.firstBufferIdx = len(feature)

            # Buffer level (in a given window size)
            if "buf" in self.options:
                filtered_lcs = local_client_state[
                    local_client_state["timestamp_s"] <= row["timestamp_start"]
                ].tail(self.options["buf"]["windowSize"])["buffer_level"].values.tolist()
                filtered_lcs.reverse()
                for _ in range(len(filtered_lcs), self.options["buf"]["windowSize"]):
                    filtered_lcs.append(0)
                feature += filtered_lcs

            # Previous packet mbit
            if "prevMbit" in self.options:
                filtered_prev_mbit = raw_df["bandwidth_mbit"].iloc[max(0,idx-self.options["prevMbit"]["windowSize"]):idx].values.tolist()
                filtered_prev_mbit.reverse()
                for _ in range(len(filtered_prev_mbit), self.options["prevMbit"]["windowSize"]):
                    filtered_prev_mbit.append(0)
                feature += filtered_prev_mbit

            # Previous packet download time
            if "prevDownloadTime" in self.options:
                filtered_prev_dt = raw_df["t_download_s"].iloc[max(0,idx-self.options["prevDownloadTime"]["windowSize"]):idx].values.tolist()
                filtered_prev_dt.reverse()
                for _ in range(len(filtered_prev_dt), self.options["prevDownloadTime"]["windowSize"]):
                    filtered_prev_dt.append(0)
                feature += filtered_prev_dt


            self.firstBitrateIdx = len(feature)
            # Previous packet bitrate
            if "prevBitrate" in self.options:
                filtered_prev_bitrate = raw_df["bitrate_level"].iloc[max(0,idx-self.options["prevBitrate"]["windowSize"]):idx].values.tolist()
                filtered_prev_bitrate.reverse()
                for _ in range(len(filtered_prev_bitrate), self.options["prevBitrate"]["windowSize"]):
                    filtered_prev_bitrate.append(0)
                feature += [br/1000000 for br in filtered_prev_bitrate]

            # Previous packet throughput with estimators
            if "prevThroughput" in self.options:
                filtered_prev_tput = raw_df["bitrate_level"].iloc[max(0,idx-self.options["prevThroughput"]["windowSize"]):idx].values.tolist()
                filtered_prev_tput.reverse()
                for _ in range(len(filtered_prev_tput), self.options["prevThroughput"]["windowSize"]):
                    filtered_prev_tput.append(0)
                feature += filtered_prev_tput
                for estimator in self.tput_estimators:
                    feature.append(estimator(filtered_prev_tput))
            
            bitrate.append(float(row["bitrate_level"]) / 1000000)
            features.append(feature)
        return features, bitrate

    def getTraceName(self, traceVideoPair):
        return traceVideoPair.split("_file_id_")[1]
    
    def getVideoName(self, traceVideoPair):
        return traceVideoPair.split("_file_id_")[0].split("video_")[1]


class TrainModel:
    def __init__(self, data, qoeOutputDir):
        self.X_train = data.X_train
        self.y_train = data.y_train

        self.X_test_files = data.X_test_files
        self.y_test_files = data.y_test_files

        self.bitrateCutoffs = data.bitrateCutoffs
        self.qoeOutputDir = qoeOutputDir

        self.firstBufferIdx = data.firstBufferIdx
        self.firstBitrateIdx = data.firstBitrateIdx

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

        for abr in self.X_train:
            print("Training PolynomialRegression for", abr)
            accuracies[abr] = {
                "LinReg": self.testLinearRegression(degree=3, abr=abr),
                "PrevBitrate": self.testSimpleClassifier(abr)
            }

        print()
        return accuracies

    def testSimpleClassifier(self, abr):
        accuracies = []
        rmses = []
        f1s = []
 
        for traceVideoPair in self.X_test_files[abr]:
            X_test = self.X_test_files[abr][traceVideoPair]
            y_test = self.y_test_files[abr][traceVideoPair]
            y_pred = [x_test_f[-1] for x_test_f in X_test]
            
            accuracies.append(self.accuracyWithCutoff(y_test, y_pred, X_test, traceVideoPair, abr))
            rmses.append(self.rmse(y_test, y_pred))
            f1s.append(self.f1score(y_test, y_pred, traceVideoPair, abr))

        print("Average Test RMSE: ", np.mean(rmses))
        print("Average Test Accuracy: ", np.mean(accuracies))
        print("Average Test F1 Score: ", np.mean(f1s))
    
        return accuracies

    def testLinearRegression(self, degree, abr):
        poly = PolynomialFeatures(degree=degree)

        X_train_poly = poly.fit_transform(self.X_train[abr])
        poly.fit(X_train_poly, self.y_train[abr])
        print("Done with poly.fit")
        reg = Ridge()
        reg.fit(X_train_poly, self.y_train[abr])

        with open("./Data/Models/{}.pickle".format(abr), "wb") as of:
            pickle.dump(reg, of)
        print("Done with reg.fit")

        accuracies = []
        rmses = []
        f1s = []
 
        for traceVideoPair in self.X_test_files[abr]:
            X_test = self.X_test_files[abr][traceVideoPair]
            y_test = self.y_test_files[abr][traceVideoPair]
            X_test_poly = poly.fit_transform(X_test)
            y_pred = reg.predict(X_test_poly)
            
            accuracies.append(self.accuracyWithCutoff(y_test, y_pred, X_test, traceVideoPair, abr))
            rmses.append(self.rmse(y_test, y_pred))
            f1s.append(self.f1score(y_test, y_pred, traceVideoPair, abr))

        print("Average Test RMSE: ", np.mean(rmses))
        print("Average Test Accuracy: ", np.mean(accuracies))
        print("Average Test F1 Score: ", np.mean(f1s))
        y_train_pred = reg.predict(X_train_poly)
        print("Average Train RMSE: ", self.rmse(self.y_train[abr], y_train_pred))

        return f1s

    def f1score(self, org, pred, traceVideoPair, abr):
        y_true = []
        y_pred = []
        videoName = self.getVideoName(traceVideoPair)
        for i in range(len(org)):
            predBitrate = max(self.bitrateCutoffs[abr][videoName])
            for k in sorted(self.bitrateCutoffs[abr][videoName]):
                if pred[i] <= self.bitrateCutoffs[abr][videoName][k]:
                    predBitrate = k
                    break
            orgBitrate = max(self.bitrateCutoffs[abr][videoName])
            for k in sorted(self.bitrateCutoffs[abr][videoName]):
                if org[i] <= self.bitrateCutoffs[abr][videoName][k]:
                    orgBitrate = k
                    break
            y_true.append(orgBitrate)
            y_pred.append(predBitrate)

        return f1_score(y_true, y_pred, average='micro')

    def accuracyWithCutoff(self, org, pred, x_test, traceVideoPair, abr):
        correct = 0
        videoName = self.getVideoName(traceVideoPair)
        with open("{}/{}/{}.txt".format(self.qoeOutputDir, abr, traceVideoPair), "w") as of:
            of.write("buffer,original_bitrate,predicted_bitrate\n")
            for i in range(len(org)):
                predBitrate = max(self.bitrateCutoffs[abr][videoName])
                for k in sorted(self.bitrateCutoffs[abr][videoName]):
                    if pred[i] <= self.bitrateCutoffs[abr][videoName][k]:
                        predBitrate = k
                        break
                orgBitrate = max(self.bitrateCutoffs[abr][videoName])
                for k in sorted(self.bitrateCutoffs[abr][videoName]):
                    if org[i] <= self.bitrateCutoffs[abr][videoName][k]:
                        orgBitrate = k
                        break
                # print(org[i], pred[i], predBitrate, orgBitrate)
                # print(x_test)
                of.write("{},{},{}\n".format(x_test[i][self.firstBufferIdx], org[i], pred[i]))
                if orgBitrate == predBitrate:
                    correct += 1

        return correct/len(org)

    def getVideoName(self, traceVideoPair):
        return traceVideoPair.split("_file_id_")[0].split("video_")[1]

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
            "bw": {"windowSize": 3},
            "buf": {"windowSize": 3},
            # "prevDownloadTime": {"windowSize": 7},
            "prevBitrate": {"windowSize": 1},
            # "prevMbit": {"windowSize": 7},
            # "prevThroughput": {"windowSize": 7}
        },
    ]

    featureGroupAccuracies = []
    xAxisLabels = []
    for idx, fg in enumerate(featureGroups):
        print("==========Feature Group {}==========".format(idx))
        data = ATCDataParser("../reconstructing-proprietary-video-streaming-algorithms/Data", fg)
        suite = TrainModel(data, "./Data/atc-data-qoe")
        accuracies = suite.classificationSuite()

        for abr in accuracies:
            for mlModel in accuracies[abr]:
                arr = np.array(accuracies[abr][mlModel], dtype=np.float64)
                print("{}-{}: {} {} {} {}".format(abr, mlModel, np.mean(arr), np.min(arr), np.max(arr), sem(arr)))
                # featureGroupAccuracies.append(accuracies[key])
                # xAxisLabels.append("FG-{}[{}]".format(idx, key))

    sys.exit()
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
    
