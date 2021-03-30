
import sys
import pdb
import numpy as np
sys.path.append("../src")
import probabilisticModels

class NaiveBayes:
    def train(self, x_by_class, model_class=probabilisticModels.Exponential):
        self._models = {}
        for class_label in x_by_class:
            x = x_by_class[class_label]
            self._models[class_label] = model_class()
            self._models[class_label].train(x=x)

    def classify(self, x):
        maxLogLikeValue = -np.Inf
        maxLogLikeClass = None
        for class_label, model in self._models.items():
            logLike = model.logLikelihood(x=x)
            # pdb.set_trace()
            if logLike>maxLogLikeValue:
                maxLogLikeValue = logLike
                maxLogLikeClass = class_label
        return maxLogLikeClass
