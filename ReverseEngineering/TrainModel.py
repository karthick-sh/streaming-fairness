from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.metrics import f1_score
from scipy.stats import sem
import pickle

from PufferDataParser import PufferDataParser

class TrainModel:
    def __init__(self, data, qoeOutputDir):
        self.X_train = data.X_train
        self.y_train = data.y_train

        self.X_test_files = data.X_test_files
        self.y_test_files = data.y_test_files
        self.bitrateCutoffs = data.bitrateCutoffs
        self.firstBufferIdx = data.firstBufferIdx

        self.qoeOutputDir = qoeOutputDir

    def classificationSuite(self):
        accuracies = {}

        for abr in self.X_train:
            print("Training PolynomialRegression for", abr)
            accuracies[abr] = {
                "LinReg": self.testLinearRegression(degree=3, abr=abr),
                "PrevBitrate": self.testSimpleClassifier(abr)
            }

        return accuracies

    def testSimpleClassifier(self, abr):
        accuracies = []
        rmses = []
        f1s = []
 
        for trace in self.X_test_files[abr]:
            X_test = self.X_test_files[abr][trace]
            y_test = self.y_test_files[abr][trace]
            y_pred = [x_test_f[-1] for x_test_f in X_test]
            
            accuracies.append(self.accuracyWithCutoff(y_test, y_pred, X_test, trace, abr))
            rmses.append(self.rmse(y_test, y_pred))
            f1s.append(self.f1score(y_test, y_pred, trace, abr))

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
 
        for trace in self.X_test_files[abr]:
            print("test trace: ", trace)
            X_test = self.X_test_files[abr][trace]
            y_test = self.y_test_files[abr][trace]
            print(X_test)
            X_test_poly = poly.fit_transform(X_test)
            y_pred = reg.predict(X_test_poly)
            
            accuracies.append(self.accuracyWithCutoff(y_test, y_pred, X_test, trace, abr))
            rmses.append(self.rmse(y_test, y_pred))
            f1s.append(self.f1score(y_test, y_pred, trace, abr))

        print("Average Test RMSE: ", np.mean(rmses))
        print("Average Test Accuracy: ", np.mean(accuracies))
        print("Average Test F1 Score: ", np.mean(f1s))
        y_train_pred = reg.predict(X_train_poly)
        print("Average Train RMSE: ", self.rmse(self.y_train[abr], y_train_pred))
        print("Coefficients: ", len(reg.coef_))

        return f1s

    def f1score(self, org, pred, traceVideoPair, abr):
        y_true = []
        y_pred = []
        for i in range(len(org)):
            predBitrate = max(self.bitrateCutoffs[abr])
            for k in sorted(self.bitrateCutoffs[abr]):
                if pred[i] <= self.bitrateCutoffs[abr][k]:
                    predBitrate = k
                    break
            orgBitrate = max(self.bitrateCutoffs[abr])
            for k in sorted(self.bitrateCutoffs[abr]):
                if org[i] <= self.bitrateCutoffs[abr][k]:
                    orgBitrate = k
                    break
            y_true.append(orgBitrate)
            y_pred.append(predBitrate)

        return f1_score(y_true, y_pred, average='micro')

    def accuracyWithCutoff(self, org, pred, x_test, traceVideoPair, abr):
        correct = 0
        with open("{}/{}/{}.txt".format(self.qoeOutputDir, abr, traceVideoPair), "w") as of:
            of.write("buffer,original_bitrate,predicted_bitrate\n")
            for i in range(len(org)):
                predBitrate = max(self.bitrateCutoffs[abr])
                for k in sorted(self.bitrateCutoffs[abr]):
                    if pred[i] <= self.bitrateCutoffs[abr][k]:
                        predBitrate = k
                        break
                orgBitrate = max(self.bitrateCutoffs[abr])
                for k in sorted(self.bitrateCutoffs[abr]):
                    if org[i] <= self.bitrateCutoffs[abr][k]:
                        orgBitrate = k
                        break
                # print(org[i], pred[i], predBitrate, orgBitrate)
                # print(x_test)
                of.write("{},{},{}\n".format(x_test[i][self.firstBufferIdx], org[i], pred[i]))
                if orgBitrate == predBitrate:
                    correct += 1

        return correct/len(org)

    def rmse(self, org, pred):
        rmse = 0
        for i in range(len(org)):
            rmse += (org[i]-pred[i])**2
        return (rmse/len(org))**0.5

    
def main():
    featureGroup = {
        "bw": {"windowSize": 3},
        "buf": {"windowSize": 3},
        # "prevDownloadTime": {"windowSize": 7},
        "prevBitrate": {"windowSize": 1},
        # "prevMbit": {"windowSize": 7},
        # "prevThroughput": {"windowSize": 7}
    }

    featureGroupAccuracies = []
    xAxisLabels = []
 
    data = PufferDataParser("./Data/ReverseEngineering/pensieve", "pensieve", featureGroup)
    suite = TrainModel(data, "./Data/puffer-data-qoe")
    accuracies = suite.classificationSuite()

    for abr in accuracies:
        for mlModel in accuracies[abr]:
            arr = np.array(accuracies[abr][mlModel], dtype=np.float64)
            print("{}-{}: {} {} {} {}".format(abr, mlModel, np.mean(arr), np.min(arr), np.max(arr), sem(arr)))
            # featureGroupAccuracies.append(accuracies[key])
            # xAxisLabels.append("FG-{}[{}]".format(idx, key))


if __name__ == "__main__":
    main()
    
