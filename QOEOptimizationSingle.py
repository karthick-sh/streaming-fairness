import time
import sys
import os
import math
import numpy as np
from scipy.optimize import NonlinearConstraint, minimize
from sklearn.preprocessing import PolynomialFeatures
import pickle

N = 4
deltaT = 1
LAMBDA = 0.5
ALPHA = 0.1
BETA = 0.1
THETA =  2.1 * 10**-3

def f(b):
    return math.exp(-LAMBDA*b)

def VQ(theta, r):
    return 1 - np.exp(-theta*r)

class QOEOptimization:
    def __init__(self, abrList, Tp):
        self.w = [500, 500, 500, 500]
        self.abrList = abrList
        self.initMLModels()
        self.Tp = Tp

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

    def QoE(self, x, prev_bw, prev_r, b, abr):
        return VQ(THETA, self.playerABR(x, prev_bw, b, prev_r, abr)) - BETA * f(b[0]) - abs(ALPHA*(VQ(THETA, self.playerABR(x, prev_bw, b, prev_r, abr)) - VQ(THETA, prev_r)))
    
    def pairwiseSum(self, QoEs):
        pairwiseError = 0
        for i in QoEs:
            for j in QoEs:
                pairwiseError += abs(i-j)
        return pairwiseError
    
    # Should return pairwise sum
    def costFunction(self, x, inputs):
        QoEs = [self.QoE(x[i], inputs[i+1]["bandwidth"], inputs[i+1]["bitrate"], inputs[i+1]["buffer"], self.abrList[i]) for i in range(N)]
        # print(QoEs, self.pairwiseSum(QoEs))
        return self.pairwiseSum(QoEs)
    
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

        W = inputs["W"]*1000
        print("W = ", W)
    
        # Constraints
        nlc1 = NonlinearConstraint(sum, W, W)
        bounds = [(0, None) for _ in range(N)]
    
        # Objective Function
        sol = minimize(self.costFunction, self.w, args=(inputs), method='SLSQP', bounds=bounds, constraints=nlc1)
    
        self.w = [int(solx) for solx in sol.x]
        print("Optimized: ", self.w)
        print()

        return [solx/1000 for solx in self.w]

    
 

