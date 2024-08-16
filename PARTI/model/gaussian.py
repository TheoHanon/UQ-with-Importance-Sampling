import numpy as np
import torch
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily
from typing import Union
from model.model import Model

class Gaussian(Model):

    def __init__(self, y: Union[np.ndarray, torch.Tensor], A: Union[np.ndarray, torch.Tensor], mu_YX: Union[np.ndarray, torch.Tensor], sigma_YX: Union[np.ndarray, torch.Tensor], sigma_X: Union[np.ndarray, torch.Tensor]):
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

        if isinstance(mu_YX, np.ndarray):
            mu_YX = torch.tensor(mu_YX, dtype=torch.float32)
        if isinstance(sigma_YX, np.ndarray):
            sigma_YX = torch.tensor(sigma_YX, dtype=torch.float32)


        self.mu_YX = mu_YX
        self.sigma_YX = sigma_YX
        self.sigma_YX_inv = torch.inverse(self.sigma_YX)

        self.sigma_XY_inv = self.sigma_X_inv + self.A.T @ self.sigma_YX_inv @ self.A 
        self.sigma_XY = torch.inverse(self.sigma_XY_inv)
        self.mu_XY = self.sigma_XY @ self.A.T @ self.sigma_YX_inv @ self.y
        
        self.model_XY = self.create_model()

    def create_model(self) -> MultivariateNormal:
        """
        Create a Mixture of Gaussians distribution
        """

        return MultivariateNormal(loc=self.mu_XY, covariance_matrix=self.sigma_XY)


    def compute_moment(self, alpha: int) -> np.ndarray:
        """
        Args:
            alpha: int Moment to compute
        """

        if alpha == 1:

            m1 = self.mu_XY

            return m1.detach().numpy()
        
        elif alpha == 2:

            m2 = self.sigma_XY + torch.outer(self.mu_XY, self.mu_XY)

            return m2.detach().numpy()
        else:
            raise ValueError("Only moments 1 and 2 are implemented")