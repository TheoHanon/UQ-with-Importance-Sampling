import torch
from model import Model
from flow import Flow
import numpy as np


class ImportanceSampling:

    def __init__(self, x_test : torch.Tensor, flow : Flow,  model : Model):

        """
        Args:
            x_test: torch.Tensor, size (1, n) : Test point
            w: torch.Tensor, size (nStep, N, d) : Particles
            logQ: torch.Tensor, size (nStep, N) : Log of the proposal distribution
            model: Model : Model
        """

        self.x_test = x_test
        self.model = model

        self.w = flow.getFlow(numpy = False)
        self.logq = flow.getlogQ(numpy = False) 
        

    def monteCarlo(self, alpha : list[int]) -> np.ndarray:

        moments = np.empty((len(alpha), self.w.shape[0], self.x_test.shape[0], self.model.m))
        for i in range(self.w.shape[0]):
            y = self.model.gw(self.x_test, self.w[i])
            for j, a in enumerate(alpha):
                moments[j, i, :] = torch.mean(y**a, axis = 0).detach().numpy()
        
        return moments
    

    def iSWeights(self) -> np.ndarray:
        
        W = torch.empty(self.w.shape[0], self.w.shape[1], dtype=torch.float32)

        for i in range(self.w.shape[0]):
            logPpos = self.model.logP(self.w[i])
            logQ = self.logq[i]
            logW = logPpos - logQ
            W[i] = torch.exp(logW - torch.logsumexp(logW, dim = 0))

            print(torch.min(W[i]), torch.max(W[i]))
        
        return W.detach().numpy()
    
    def importanceSampling(self, alpha : list[int]) -> np.ndarray:

        moments = np.empty((len(alpha), self.w.shape[0], self.x_test.shape[0], self.model.m))

        for i in range(self.w.shape[0]):

            y = self.model.gw(self.x_test, self.w[i])
            logPpos = self.model.logP(self.w[i])
            logQ = self.logq[i]

            logW = logPpos - logQ
            W = torch.exp(logW - torch.logsumexp(logW, dim = 0))
            print(torch.min(W), torch.max(W))

            for j, a in enumerate(alpha):

                moments[j, i, ...] = torch.sum(W[:, np.newaxis, np.newaxis] * y**a, axis = 0).detach().numpy() 

        return moments

    def adaptiveImportanceSampling(self, alpha : list[int]) -> np.ndarray:

        weights = torch.empty(self.w.shape[:-1], dtype=torch.float32)
        y = torch.empty((self.w.shape[0], self.w.shape[1], self.x_test.shape[0], self.model.m), dtype=torch.float32)
        moments = np.empty((len(alpha), self.w.shape[0], self.x_test.shape[0], self.model.m))

        for i in range(self.w.shape[0]):
            # test = self.model.gw(self.x_test, self.w[i]).squeeze(1)
            y[i] = self.model.gw(self.x_test, self.w[i])
            logP = self.model.logP(self.w[i])
            logQ = self.logq[i]

            logW = logP - logQ
            weights[i] = torch.exp(logW - torch.logsumexp(logW, dim = 0))
            # print(torch.min(weights[i]), torch.max(weights[i]))
            W = weights[:i+1].reshape(-1)
            W = (W / W.sum()).reshape(-1, 1)
            
            for j, a in enumerate(alpha):
                # print(torch.sum(y[:i+1].reshape(-1, self.x_test.shape[0], self.model.n), axis = 0).shape)
                moments[j, i, ...] = torch.sum(W[:, np.newaxis] * (y[:i+1].reshape(-1, self.x_test.shape[0], self.model.n))**a, axis = 0).detach().numpy()

        return moments