import torch
from tqdm.autonotebook import tqdm
from flow.flow import Flow
from model.model import Model
from torch.distributions.distribution import Distribution
import numpy as np
from matplotlib import pyplot as plt
from typing import Union


class SplitStepFlow(Flow):

    def __init__(self, N:int, nStep:int, dt:float, model : Model, q0 : Distribution = torch.distributions.Uniform(-2, 2), k : float = 1 , sigma0 : float = 1, phi_n : float = 0.015, alpha_star : float = 0.574, lam : int = 10, burn_in : int = 500, pre_compute : bool = False):
        """
        Args:
            N: int Number of particles
            nStep: int Number of steps
            dt: float Time step
            q0: Distribution Initial distribution
            k: float Diffusion coefficient

            model: Model Model to use
            k: float Diffusion coefficient
            phi_n: float Constant lr
            alpha_star: float Target acceptance rate
            lam: int Damping parameter
            burn_in: int Burn-in period
        """

        self.k = k
        self.z = None
        self.sigma0 = sigma0

        self.phi_n = phi_n
        self.alpha_star = alpha_star
        self.lam = lam
        self.burn_in = burn_in

        super().__init__(N, nStep, dt, model, q0, pre_compute)




    def flow(self):
        """
        Args:

        """

        x = torch.empty((self.nStep, self.N, self.model.n), dtype=torch.float32)
        z = torch.empty((self.nStep, self.N, self.model.n), dtype=torch.float32)
       
        x[0] = self.q0.sample((self.N, self.model.n))
        z[0] = x[0]

        sigma_s = self.sigma0 * torch.ones(self.N)
        sigma_R = self.sigma0 * torch.ones(self.N)
        R = torch.tile(torch.eye(self.model.n), (self.N, 1, 1))

        D_square_root = np.sqrt(self.k)  * torch.eye(self.model.n)

        normalDistribution = torch.distributions.MultivariateNormal(torch.zeros(self.model.n), torch.eye(self.model.n))


        with tqdm(total=self.nStep - 1, desc="SplitStep flow") as pbar :

            for i in range(self.nStep - 1):

                R_mean = torch.mean(R, dim = 0)
                sigma_R_mean = torch.mean(sigma_R)

                gradlogP_x = self.model.grad_logP_XY(x[i])

                xsi = torch.sqrt(sigma_R_mean) * torch.einsum('jk,ik->ij', R_mean, normalDistribution.sample((self.N,)))
           
                y = x[i] + (sigma_R_mean / 2) * torch.einsum("jk, ik -> ij", R_mean, torch.einsum("jk, ik -> ij", R_mean.T, gradlogP_x)) + xsi
    
                gradlogP_y = self.model.grad_logP_XY(y)

                h_yx = self.h(y, x[i], gradlogP_x, sigma_R_mean, torch.matmul(R_mean, R_mean.T))
                h_xy = self.h(x[i], y, gradlogP_y, sigma_R_mean, torch.matmul(R_mean, R_mean.T))

                logP_y = self.model.logP_XY(y)
                logP_x = self.model.logP_XY(x[i])

                alpha = torch.minimum(torch.ones(self.N), torch.exp(logP_y + h_xy - logP_x - h_yx))

                if i < self.burn_in:

                    sigma_s *= (1 + self.phi_n * (alpha - self.alpha_star))
                    sigma_R = sigma_s

                elif i == self.burn_in :

                    s_delta = torch.sqrt(alpha)[:, None] * (gradlogP_y - gradlogP_x)
                    inner_s = torch.einsum("ij, ij -> i", s_delta, s_delta)
                    r = 1 / (1 + torch.sqrt(self.lam / (self.lam + inner_s)))

                    R = 1/np.sqrt(self.lam) * (torch.eye(self.model.n) - (r / (self.lam + inner_s))[:, None, None] * np.einsum("ij, il -> ijl", s_delta, s_delta))

                    sigma_s = sigma_s * (1 + self.phi_n * (alpha - self.alpha_star))
                    sigma_R = sigma_s / (1/self.model.n * torch.einsum("ijk, ikj -> i", R, R))

                else :

                    s_delta = torch.sqrt(alpha)[:, None] * (gradlogP_y - gradlogP_x)

                    phi = torch.einsum("ijk, ij -> ik", R.transpose(1, 2), s_delta)
                    inner_phi = torch.einsum("ij, ij -> i", phi, phi)
                    r = 1 / (1 + torch.sqrt(1 / (1 + inner_phi)))

                    R = R - (r / (1 + inner_phi))[:, None, None] * torch.einsum("ij, il -> ijl", torch.einsum("ijk, ik -> ij", R, phi), phi)
                    
                    sigma_s *= (1 + self.phi_n *(alpha - self.alpha_star))
                    sigma_R = sigma_s / (1/self.model.n * torch.einsum("ijk, ikj -> i", R, R))


                u = torch.rand(self.N)
                mask = u < alpha

                x[i+1][mask] = y[mask]
                x[i+1][~mask] = x[i][~mask]
                
                eta = np.sqrt(2*self.dt) * D_square_root @ normalDistribution.sample((self.N,)).T

                z[i+1] = x[i+1] + eta.transpose(0,1)

                pbar.update(1)

        self.x = x
        self.z = z

        return 
    

    def logQ(self) -> None:

        if self.x is None:
            self.flow()

        zt = self.z
        xt = self.x
        logQ = torch.empty((self.nStep, self.N), dtype=torch.float32)
        
        logQ[0] = self.q0.log_prob(zt[0]).sum(dim = 1)
        
        D_inv = (1/self.k) * torch.eye(self.model.n)

        for i in range(1, self.nStep):
            diff = zt[i, :, None, :] - xt[i, None, :, :]
            exponents = -.5 * torch.einsum("ijl, lk, ijk -> ij", diff, D_inv/(2*self.dt), diff)
            logQ[i] = torch.logsumexp(exponents, dim = 1) - self.model.n/2 * np.log(4*np.pi*self.dt) - self.model.n/2 * np.log(self.k) - np.log(self.N)

        self.logq = logQ

        return 

    @staticmethod
    def h(x: torch.Tensor, y: torch.Tensor, grad_logP_XY: torch.Tensor, sigma: torch.Tensor, A:torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.einsum("ij, ij -> i", y - x - (sigma / 4) * torch.einsum("kj, ik -> ij", A, grad_logP_XY), grad_logP_XY)
    
    def getFlow(self, numpy : bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            numpy: bool If True, return a numpy array
        """

        if self.z is None:
            self.flow()

        return self.z.detach().numpy() if numpy else self.z.copy()
        

    def importanceSampling(self, alpha : list[int]) -> np.ndarray:
    
        if self.logq is None:
            self.logQ()

        z = self.z
        moments = np.empty((len(alpha), self.nStep, self.model.n))

        logP = self.model.model_XY.log_prob(z)
        logQ = self.logq


        for i in range(self.nStep):
            logW = logP[i] - logQ[i]

            W = torch.exp(logW - torch.logsumexp(logW, dim = 0))

            for j, a in enumerate(alpha):
                moments[j, i, :] = torch.sum(W[:, np.newaxis] * z[i]**a, axis = 0).detach().numpy()

        return moments


    def adaptiveImportanceSampling(self, alpha : list[int]) -> np.ndarray:

        if self.logq is None:
            self.logQ()

        z = self.z
        w = torch.empty((self.nStep, self.N), dtype=torch.float32)
        moments = np.empty((len(alpha), self.nStep, self.model.n))

        logP = self.model.logP_XY(z)
        logQ = self.logq


        for i in range(self.nStep):
            logW = logP[i] - logQ[i]
            w[i] = torch.exp(logW - torch.logsumexp(logW, dim = 0))

            W = w[:i+1].reshape(-1)
            W = (W / W.sum()).reshape(-1, 1)

            for j, a in enumerate(alpha):
                m = torch.sum(W * (z[:i+1].reshape(-1, self.model.n))**a, axis = 0).detach().numpy()
                
                moments[j, i, :] = m

        return moments