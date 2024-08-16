import torch
from torch import nn
import numpy as np
from typing import Union, Tuple
from robust_uniform import RobustUniform

class Model:
    def __init__(self, Y_train: Union[np.ndarray, torch.Tensor], X_train : Union[np.ndarray, torch.Tensor], gw : nn.Module, prior_bounds : Union[Union[np.ndarray, torch.Tensor], Tuple], scale_likelihood: float = 1.0):

        """
        Args:
        X_train, Y_train : Union[np.ndarray, torch.Tensor], size (M, n), (M, m) : Training sample
        gw : nn.Module :  Neural network
        prior_bounds : Union[Union[np.ndarray, torch.Tensor], Tuple], size (d, 2) or tuple :  Bounds on the uniform prior distribution

        """
        
        if isinstance(Y_train, np.ndarray):
            Y_train = torch.tensor(Y_train, dtype=torch.float64)

        if isinstance(X_train, np.ndarray):
            X_train = torch.tensor(X_train, dtype=torch.float64)


        self.m = Y_train.shape[1] # size of y
        self.n = X_train.shape[1] # size of x
        self.M = X_train.shape[0] # number of training samples

        self.d = gw.get_params_count() # number of parameters in the neural network

        if isinstance(prior_bounds, np.ndarray):
            if prior_bounds.shape[0] != self.d :
                raise ValueError("The number of bounds must be equal to the number of parameters in the neural network")
            prior_bounds = torch.tensor(prior_bounds, dtype=torch.float64)

        if isinstance(prior_bounds, tuple):
            prior_bounds = torch.tile(torch.tensor(prior_bounds, dtype=torch.float64), (self.d, 1))

        self.X_train = X_train
        self.Y_train = Y_train
        self.gw = gw

        self.priorW = RobustUniform(prior_bounds[:, 0], prior_bounds[:, 1], val_max = 1e2)   
        self.likelihood = torch.distributions.Normal(loc=0.0, scale=scale_likelihood)   


    def logP(self, w: torch.Tensor) -> torch.Tensor:       
        prior = self.priorW.log_prob(w).sum()
        logP_XY =  self.likelihood.log_prob(torch.sqrt(((self.Y_train - self.gw(self.X_train, W = w))**2).sum(dim = 2).mean(dim = 1))) + prior
        return logP_XY

    def grad_logP(self, w: torch.Tensor) -> torch.Tensor:
        """
        Args:
        w : torch.Tensor, size (N, d) : should be differentiable
        """
        
        logP_XY = self.logP(w)
        grad_w = torch.autograd.grad(logP_XY, w, create_graph=True, retain_graph = True, grad_outputs = torch.ones_like(logP_XY))[0]
    
        return grad_w
    

    def laplace_logP(self, w: torch.Tensor, grad_logP : torch.Tensor) -> torch.Tensor:
        """
        Args:

        w : torch.Tensor, size (N, d) : should be differentiable
        grad_logP : torch.Tensor, size (N, d) : should be differentiable

        """
        
        div = 0.0

        for i in range(self.n):
            grad_i = grad_logP[..., i]
            div += torch.autograd.grad(grad_i, w, grad_outputs = torch.ones_like(grad_i))[0][..., i:i+1]
        
        return div.squeeze()
    


    