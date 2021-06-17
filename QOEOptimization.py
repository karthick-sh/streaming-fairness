import time
import sys
import os
import math
import numpy as np
from scipy.optimize import NonlinearConstraint, minimize
from sklearn.preprocessing import PolynomialFeatures
import pickle
from collections import deque

N = 4
LAMBDA = 0.5
ALPHA = 0.1
BETA = 0.1
THETA =  2.1 * 10**-3
deltaT = 1

# Hyperparameters
GAMMA = 1

def f(b):
    return math.exp(-LAMBDA*b)

def VQ(theta, r):
    return 1 - np.exp(-theta*r)

class QOEOptimization:
    def __init__(self, abrList, Tp, windowLength=3):
        self.w = [500 for _ in range(4*Tp)]
        self.abrList = abrList
        self.initMLModels()
        self.Tp = Tp
        self.windowLength = windowLength

    def initMLModels(self):
        self.abrModels = {}
        for abr in self.abrList:
            if abr not in self.abrModels:
                with open("./Data/Models/{}.pickle".format(abr), "rb") as mf:
                    self.abrModels[abr] = pickle.load(mf)
 
    def playerABR(self, x, prev_bw, b, prev_r, abr):
        poly = PolynomialFeatures(degree=3)
        X = [x] + [pbw*1000 for pbw in prev_bw] + b + [prev_r]
        X = np.array(X).reshape(1, -1)
        x_poly = poly.fit_transform(X)

        y_pred = self.abrModels[abr].predict(x_poly)

        return max(y_pred, 0)

    def ABR(self, bw_list, buf_list, r, abr):
        poly = PolynomialFeatures(degree=3)
        X = list(bw_list) + list(buf_list) + [r]
        X = np.array(X).reshape(1, -1)
        x_poly = poly.fit_transform(X)

        y_pred = self.abrModels[abr].predict(x_poly)

        return min(max(y_pred[0], 0), 1000000)

    def QoE(self, bw_list, buf_list, prev_r, abr):
        return VQ(THETA, self.ABR(bw_list, buf_list, prev_r, abr)) - BETA * f(buf_list[0]) - abs(ALPHA*(VQ(THETA, self.ABR(bw_list, buf_list, prev_r, abr)) - VQ(THETA, prev_r)))
    
    def pairwiseSum(self, QoEs):
        pairwiseError = 0
        for i in QoEs:
            for j in QoEs:
                pairwiseError += abs(i-j)
        return pairwiseError

    def tpSumGen(self, t):
        def tpSum(x):
            x_vals = [x[self.Tp*n+t] for n in range(N)]
            return sum(x_vals)
        return tpSum 
    
    # Should return pairwise sum
    def costFunction(self, x, inputs):
        # Build bandwidths deque
        bandwidths = {k: deque(maxlen=self.windowLength) for k in range(1, 5)}
        for k in range(1, 5):
            for bw in inputs[k]["bandwidth"]:
                bandwidths[k].append(bw*1000)

        # Build bitrates dict
        bitrates = {k: inputs[k]["bitrate"] for k in range(1, 5)}

        # Build buffer deque 
        buffers = {k: deque(maxlen=self.windowLength) for k in range(1, 5)}
        for k in range(1, 5):
            for buf in inputs[k]["buffer"]:
                buffers[k].append(buf)

        # Calculate pairwise sum for prediction window
        totalPairwiseSum = 0
        totalSocialWelfare = 0
        for t in range(self.Tp):
            QoEs = []
            for n in range(N):
                QoEs.append(self.QoE(
                    bandwidths[n+1],
                    buffers[n+1],
                    bitrates[n+1],
                    self.abrList[n]))

                # Update inputs for next timestamp
                next_abr_for_buf = self.ABR(bandwidths[n+1], buffers[n+1], bitrates[n+1], self.abrList[n])
                if next_abr_for_buf != 0:
                    new_buf = buffers[n+1][0] + (bandwidths[n+1][0] / next_abr_for_buf) * deltaT
                else:
                    new_buf = max(0, buffers[n+1][0] - deltaT)
                # print("next aBR: ", next_abr_for_buf)
                # print("New buf: ", new_buf, buffers[n+1][0])
                # print("New bw: ", x[N*n+t])
                buffers[n+1].appendleft(new_buf)
                bandwidths[n+1].appendleft(x[self.Tp*n + t])
                bitrates[n+1] = self.ABR(bandwidths[n+1], buffers[n+1], bitrates[n+1], self.abrList[n])

            totalPairwiseSum += self.pairwiseSum(QoEs)
            totalSocialWelfare += sum(QoEs)

        totalSocialWelfare = -totalSocialWelfare / (N * self.Tp)
        return GAMMA*totalSocialWelfare + (1 - GAMMA)*totalPairwiseSum

    def runOneIteration(self, inputs):
        for k in ["W", 1, 2, 3, 4]:
            if k not in inputs:
                print("{} not in inputs".format(k))
                return []

        for i in range(1, 5):
            for k in ["bandwidth", "bitrate", "buffer"]:
                if k not in inputs[i]:
                    print("{} not in inputs[{}]".format(k, i))
                    return []
            if inputs[i]["bitrate"] < 0:
                inputs[i]["bitrate"] = 0
            inputs[i]["buffer"] = [max(buf, 0) for buf in inputs[i]["buffer"]]

        # Constraints
        nlc1 = [NonlinearConstraint(self.tpSumGen(t), int(inputs["W"][t]*1000), int(inputs["W"][t]*1000)) for t in range(self.Tp)]
        bounds = [(0, None) for _ in range(N*self.Tp)]

        # Objective Function
        sol = minimize(self.costFunction, self.w, args=(inputs), method='SLSQP', bounds=bounds, constraints=nlc1)
    
        self.w = [int(solx) for solx in sol.x]
        print("Optimizedzz: ", [self.w[self.Tp*n] for n in range(N)])
        print()

        return [self.w[self.Tp*n]/1000 for n in range(N)][::-1]


if __name__ == "__main__":
    qoe = QOEOptimization(["YouTube", "YouTube", "mpc", "mpc"], 3)
    inputs = {
        "W": [0.94457142, 1.11073432, 0.72179543],
        1: {
            "bandwidth": [0.1, 0.1, 0.1],
            "bitrate": 3000,
            "buffer": [4, 5, 6]
        },
        2: {
            "bandwidth": [0.1, 0.1, 0.1],
            "bitrate": 3000,
            "buffer": [4, 5, 6]
        },
        3: {
            "bandwidth": [0.5, 0.5, 0.5],
            "bitrate": 300,
            "buffer": [4, 5, 6]
        },
        4: {
            "bandwidth": [0.5, 0.5, 0.5],
            "bitrate": 300,
            "buffer": [4, 5, 6]
        }
    }
    print(qoe.runOneIteration(inputs))