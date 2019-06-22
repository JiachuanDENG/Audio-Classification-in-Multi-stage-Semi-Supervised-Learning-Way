from numpy import array
from numpy import argmax
import numpy as np
class One_hot_encoder(object):
    def __init__(self,labels):
        self.labels = labels
        self.labels2int()
        self.labels2onehot()
    def labels2int(self):
        self.labels2intDict = {}
        idx = 0
        for l in set(self.labels):
            self.labels2intDict[l] = idx
            idx += 1

    def labels2onehot(self,):
        self.labels2onehotDict = {}
        labellen = len(self.labels2intDict)
        for l in self.labels2intDict:
            onehot = np.zeros(labellen)
            onehot[self.labels2intDict[l]] = 1
            self.labels2onehotDict[l] = onehot
    def get_numlabels(self):
        return len(self.labels2intDict)
    def label2onehot(self,label):
        return self.labels2onehotDict[label]
