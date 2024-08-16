import numpy as np
import torch
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily
from typing import Union
from model.model import Model


class GaussianMixture(Model):

    def __init__(self, y: Union[np.ndarray, torch.Tensor], A: Union[np.ndarray, torch.Tensor], weights: Union[np.ndarray, torch.Tensor], mu_YX: Union[np.ndarray, torch.Tensor], sigma_YX: Union[np.ndarray, torch.Tensor], sigma_X: Union[np.ndarray, torch.Tensor]):
        """
        Args:
            y: Union[np.ndarray, torch.Tensor] Measurement vector
            A: Union[np.ndarray, torch.Tensor] Measurement matrix
            weights: Union[np.ndarray, torch.Tensor] Weights of the mixture components
            mu_YX: Union[np.ndarray, torch.Tensor] Mean of the Gaussian components
            sigma_YX: Union[np.ndarray, torch.Tensor] Covariance of the Gaussian components
            sigma_X: Union[np.ndarray, torch.Tensor] Covariance of the state
        """

        super().__init__(y, A, sigma_X)

        if isinstance(weights, np.ndarray):
            weights = torch.tensor(weights, dtype=torch.float32)
        if isinstance(mu_YX, np.ndarray):
            mu_YX = torch.tensor(mu_YX, dtype=torch.float32)
        if isinstance(sigma_YX, np.ndarray):
            sigma_YX = torch.tensor(sigma_YX, dtype=torch.float32)

        self.nMixt = len(weights)
        self.weights = weights

        self.mu_YX = mu_YX
        self.sigma_YX = sigma_YX
        self.sigma_YX_inv = torch.stack([torch.inverse(sigma_YX[i]) for i in range(self.nMixt)])

        self.sigma_XY_inv = torch.stack([self.sigma_X_inv + self.A.T @ self.sigma_YX_inv[i] @ self.A for i in range(self.nMixt)])
        self.sigma_XY = torch.stack([torch.inverse(self.sigma_XY_inv[i]) for i in range(self.nMixt)])
        self.mu_XY = torch.stack([self.sigma_XY[i] @ self.A.T @ self.sigma_YX_inv[i] @ (self.y - self.mu_YX[i]) for i in range(self.nMixt)])
        
        self.model_XY = self.create_model()

    def create_model(self) -> MixtureSameFamily:
        """
        Create a Mixture of Gaussians distribution
        """

        component_distribution = MultivariateNormal(loc=self.mu_XY, covariance_matrix=self.sigma_XY)
        mixture_distribution = Categorical(probs=self.weights)

        return MixtureSameFamily(mixture_distribution=mixture_distribution, component_distribution=component_distribution)


    def compute_moment(self, alpha: int) -> np.ndarray:
        """
        Args:
            alpha: int Moment to compute
        """

        if alpha == 1:

            m1 = torch.sum(self.mu_XY * self.weights[:, None], axis = 0)
            return m1.detach().numpy()
        
        elif alpha == 2:

            m2 = torch.zeros((self.n, self.n))
            for i in range(self.nMixt):
                m2 += self.weights[i] * (self.sigma_XY[i] + torch.outer(self.mu_XY[i], self.mu_XY[i]))

            return m2.detach().numpy()
        else:
            raise ValueError("Only moments 1 and 2 are implemented")
            
