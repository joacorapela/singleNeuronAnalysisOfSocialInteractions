
import pdb
import abc
import numpy as np

class ProbabilisticModel(abc.ABC):
    @abc.abstractmethod
    def train(self, x):
        pass

    @abc.abstractmethod
    def logLikelihood(self, x):
        pass

class Exponential(ProbabilisticModel):
    def train(self, x):
        self._lambda = 1/x.mean()

    def logLikelihood(self, x):
        N = len(x)
        answer = N*np.log(self._lambda)*(1-x.mean())
        return answer

class InverseGaussian(ProbabilisticModel):
    def train(self, x):
        self._mu = x.mean()
        self._lambda = 1/(1/x-1/self._mu).mean()

    def logLikelihood(self, x):
        N = len(x)
        answer = N/2*np.log(self._lambda/(2*np.pi))
        answer -= 3.0/2.0*np.log(x).sum()
        answer -= self._lambda*np.divide((x-self._mu)**2, 2*x*self._mu**2).sum()
        return answer
