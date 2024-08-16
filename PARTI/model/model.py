import torch
import numpy as np
from typing import Union
from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, y: Union[np.ndarray, torch.Tensor], A: Union[np.ndarray, torch.Tensor], sigma_X: Union[np.ndarray, torch.Tensor]):
        
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32)
        if isinstance(A, np.ndarray):
            A = torch.tensor(A, dtype=torch.float32)
        if isinstance(sigma_X, np.ndarray):
            sigma_X = torch.tensor(sigma_X, dtype=torch.float32)

        self.m = y.shape[0]
        self.n = A.shape[1]

        self.y = y
        self.A = A

        self.sigma_X = sigma_X
        self.sigma_X_inv = torch.inverse(sigma_X)

        self.model_XY = None

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def compute_moment(self):
        pass

    def logP_XY(self, x: Union[np.ndarray, torch.Tensor], numpy: bool = False) -> Union[np.ndarray, torch.Tensor]:

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        elif isinstance(x, torch.Tensor):
            x = x.clone().detach()
        else:
            raise ValueError("x must be a numpy array or a torch tensor")

        logP_XY = self.model_XY.log_prob(x)

        if numpy:
            return logP_XY.detach().numpy()
        
        return logP_XY

    def grad_logP_XY(self, x: Union[np.ndarray, torch.Tensor], numpy: bool = False) -> Union[np.ndarray, torch.Tensor]:

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
        elif isinstance(x, torch.Tensor):
            x = x.clone().detach().requires_grad_(True)
        else:
            raise ValueError("x must be a numpy array or a torch tensor")
        
        logP_XY = self.model_XY.log_prob(x)
        logP_XY.backward(torch.ones_like(logP_XY))

        grad = x.grad
        if numpy:
            return grad.detach().numpy()
        
        return grad
    

    def laplace_logP_XY(self, x: Union[np.ndarray, torch.Tensor], numpy: bool = False) -> Union[np.ndarray, torch.Tensor]:

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
        elif isinstance(x, torch.Tensor):
            x = x.clone().detach().requires_grad_(True)
        else:
            raise ValueError("x must be a numpy array or a torch tensor")
        

        logP_XY = self.model_XY.log_prob(x)
        grad = torch.autograd.grad(logP_XY, x, create_graph=True, grad_outputs = torch.ones_like(logP_XY))[0]

        div = 0.0

        for i in range(self.n):
            div += torch.autograd.grad(grad[..., i], x, create_graph=True, grad_outputs = torch.ones_like(grad[..., i]))[0][..., i:i+1]
        if numpy:
            return div.detach().numpy()
        
        return div
    
