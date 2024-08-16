import numpy as np
import torch
from torch.distributions.distribution import Distribution
from tqdm.autonotebook import tqdm
from model.model import Model
from typing import Union
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod


class Flow(ABC):

    def __init__(self, N:int, nStep:int, dt:float, model : Model, q0 : Distribution = torch.distributions.Uniform(-2, 2), pre_compute : bool = False):
        """
        Args:
            N: int Number of particles
            nStep: int Number of steps
            dt: float Time step
            q0: Distribution Initial distribution
        """

        self.N = N
        self.nStep = nStep
        self.dt = dt

        self.model = model
        self.q0 = q0

        self.x = None
        self.logq = None

        if pre_compute:
            self.logQ()
        
    def getFlow(self, numpy : bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            numpy: bool If True, return a numpy array
        """

        if self.x is None:
            self.flow()

        return self.x.detach().numpy() if numpy else self.x.copy()
    

    def getlogQ(self, numpy : bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            numpy: bool If True, return a numpy array
        """

        if self.logq is None:
            self.logQ()

        return self.logq.detach().numpy() if numpy else self.logq.copy()
    
        
    @abstractmethod
    def flow(self) -> None:
        pass

    @abstractmethod
    def logQ(self):
        pass

    @staticmethod
    def processFlow(x : torch.Tensor, numpy : bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            x: torch.Tensor
            numpy: bool If True, return a numpy array
        """

        if numpy:
            return x.detach().numpy()

        return x.copy()
    

    def monteCarlo(self, alpha : list[int]) -> np.ndarray:

        if self.x is None:
            self.flow()

        x = self.x
        moments = np.empty((len(alpha), self.nStep, self.model.n))
        
        for i in range(self.nStep):
            for j, a in enumerate(alpha):
                mean_time = torch.mean(x[:i+1]**a, axis = 0)
                moments[j, i, :] = torch.mean(mean_time, axis = 0).detach().numpy()
        
        return moments
    
    def importanceSampling(self, alpha : list[int]) -> np.ndarray:

        if self.logq is None:
            self.logQ()

        x = self.x

        moments = np.empty((len(alpha), self.nStep, self.model.n))

        logP = self.model.model_XY.log_prob(x)
        logQ = self.logq

        for i in range(self.nStep):
            logW = logP[i] - logQ[i]
            W = torch.exp(logW - torch.logsumexp(logW, dim = 0))
        
            for j, a in enumerate(alpha):
                moments[j, i, :] = torch.sum(W[:, np.newaxis] * x[i]**a, axis = 0).detach().numpy() 

        return moments

    def adaptiveImportanceSampling(self, alpha : list[int]) -> np.ndarray:

        if self.logq is None:
            self.logQ()

        x = self.x
        w = torch.empty((self.nStep, self.N), dtype=torch.float32)
        moments = np.empty((len(alpha), self.nStep, self.model.n))

        logP = self.model.logP_XY(x)
        logQ = self.logq


        for i in range(self.nStep):
            logW = logP[i] - logQ[i]
            w[i] = torch.exp(logW - torch.logsumexp(logW, dim = 0))

            W = w[:i+1].reshape(-1)
            W = (W / W.sum()).reshape(-1, 1)
            
            for j, a in enumerate(alpha):
                moments[j, i, :] = torch.sum(W * (x[:i+1].reshape(-1, self.model.n))**a, axis = 0).detach().numpy()

        return moments