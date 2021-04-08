
import sys
import pdb
import numpy as np
sys.path.append("../src")
import probabilisticModels

class NaiveBayes:
    def train(self, x, y, model_class=probabilisticModels.Exponential):
        self._models = {}
        unique_y = np.unique(y)
        for class_label in unique_y:
            # x_in_class_indices = np.where(y==class_label)[0]
            x_in_class = np.array([x[i] for i, y_item in enumerate(y) if y_item==class_label])
            self._models[class_label] = model_class()
            self._models[class_label].train(x=x_in_class)
        # pdb.set_trace()

    def predict(self, x):
        y = [None]*len(x)
        for i, x_item in enumerate(x):
            maxProbValue = -np.Inf
            maxProbClass = None
            for class_label, model in self._models.items():
                prob = model.probability(x=x_item)
                # pdb.set_trace()
                if prob>maxProbValue:
                    maxProbValue = prob
                    maxProbClass = class_label
            y[i] = maxProbClass
        return y
