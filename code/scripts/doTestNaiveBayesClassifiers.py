
import sys
import pdb
import numpy as np

sys.path.append("../src")
import classifiers
import probabilisticModels

def generateTrainData():
    ISIs_short = np.random.wald(mean=1.0, scale=1.0, size=1000)
    ISIs_medium = np.random.wald(mean=1.2, scale=1.0, size=1000)
    ISIs_long = np.random.wald(mean=1.3, scale=1.0, size=1000)
    answer = {"short": ISIs_short, "medium": ISIs_medium, "long": ISIs_long}
    return answer

def generateTestData():
    ISIs_long = np.random.wald(mean=1.2, scale=1.0, size=1000)
    return ISIs_long

def main(argv):
    train_ISIs = generateTrainData()
    test_ISIs = generateTestData()
    classifier = classifiers.NaiveBayes()
    classifier.train(x_by_class=train_ISIs,
                     # model_class=probabilisticModels.Exponential)
                     model_class=probabilisticModels.InverseGaussian)
    for model_label, model in classifier._models.items():
        print("{:s}: mu={:f}, lambda={:f}".format(model_label, model._mu, model._lambda))
    classified_behavior = classifier.classify(x=test_ISIs)
    aString = "True test behavior={:s}. Classified test behavior={:s}".format("medium", classified_behavior)
    print(aString)

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
