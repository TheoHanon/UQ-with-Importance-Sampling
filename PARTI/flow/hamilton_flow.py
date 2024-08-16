import torch
from tqdm.autonotebook import tqdm
from flow.flow import Flow
from model.model import Model
from torch.distributions.distribution import Distribution
from torch.distributions import MultivariateNormal
import numpy as np
from matplotlib import pyplot as plt
from typing import Union, Tuple


class HamiltonFlow(Flow):

    def __init__(self, N:int, nStep:int, dt:float, model : Model, q0 : Distribution = torch.distributions.Uniform(-2, 2), pre_compute : bool = False, M_inv : torch.tensor = None, epsilon : float = 1e-3):
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

        super().__init__(N, nStep, dt, model, q0, pre_compute)

        if M_inv is None:
            self.M_inv = torch.eye(self.model.n)
        else:
            self.M_inv = M_inv

        self.epsilon = epsilon


    def flow(self):
        
        x = torch.empty((self.nStep, self.N, self.model.n), dtype=torch.float32)
        x[0] = self.q0.sample((self.N, self.model.n))

        momentum = MultivariateNormal(loc = torch.zeros(self.model.n), precision_matrix = self.M_inv)

        with tqdm(total=self.nStep, desc="Hamilton flow") as pbar:

            for i in range(self.nStep - 1):
                p0 = momentum.sample((self.N,))
                x_new, p_new = self.leapfrog(x[i], p0)

                Ux_new = -self.model.logP_XY(x_new)
                Ux_old = -self.model.logP_XY(x[i])

                Kq_new = momentum.log_prob(p_new)
                Kq_old = momentum.log_prob(p0)

                alpha = torch.min(torch.tensor(1.0), torch.exp(Ux_old - Ux_new + Kq_old - Kq_new))

                u = torch.rand(self.N)
                mask = u < alpha

                x[i+1] = torch.where(mask[:, None], x_new, x[i])

                pbar.update(1)
            

        self.x = x

        return 

    def logQ(self) -> None:
            return


    def leapfrog(self, q: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: torch.Tensor Position
            p: torch.Tensor Momentum
            dVdq: torch.Tensor Gradient of the potential energy
            t_end: float End time
            dt: float Time step
        """

        q = q.clone()
        p = p.clone()

        p -= 0.5 * self.epsilon *(-self.model.grad_logP_XY(q))

        for _ in range(int(self.dt / self.epsilon)):
            q += self.epsilon * torch.einsum("ij, kj -> ki", self.M_inv, p)
            p -= self.epsilon * (-self.model.grad_logP_XY(q))

        q += self.epsilon * torch.einsum("ij, kj -> ki",self.M_inv, p)
        p -= 0.5 * self.epsilon * (-self.model.grad_logP_XY(q))

        return q, -p
    



class SplitHamiltonFlow(Flow):

    def __init__(self, N:int, nStep:int, dt:float, model : Model, q0 : Distribution = torch.distributions.Uniform(-2, 2), pre_compute : bool = False, M_inv : torch.tensor = None, epsilon : float = 1e-3, k : float = 1):
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
                
        if M_inv is None:
            self.M_inv = torch.eye(model.n)
        else:
            self.M_inv = M_inv

        self.epsilon = epsilon
        self.z = None
        self.logq = None
        self.k = k

        super().__init__(N, nStep, dt, model, q0, pre_compute)


        
    def flow(self):
        
        x = torch.empty((self.nStep, self.N, self.model.n), dtype=torch.float32)
        z = torch.empty((self.nStep, self.N, self.model.n), dtype=torch.float32)
        x[0] = self.q0.sample((self.N, self.model.n))
        z[0] = x[0]

        momentum = MultivariateNormal(loc = torch.zeros(self.model.n), precision_matrix = self.M_inv)
        normalDistribution = MultivariateNormal(torch.zeros(self.model.n), torch.eye(self.model.n))
        D_square_root = np.sqrt(self.k) * torch.eye(self.model.n)

        with tqdm(total=self.nStep, desc="Hamilton flow") as pbar:

            for i in range(self.nStep - 1):
                p0 = momentum.sample((self.N,))
                x_new, p_new = self.leapfrog(x[i], p0)

                Ux_new = -self.model.logP_XY(x_new)
                Ux_old = -self.model.logP_XY(x[i])

                Kq_new = momentum.log_prob(p_new)
                Kq_old = momentum.log_prob(p0)

                alpha = torch.min(torch.tensor(1.0), torch.exp(Ux_old - Ux_new + Kq_old - Kq_new))

                u = torch.rand(self.N)
                mask = u < alpha

                x[i+1] = torch.where(mask[:, None], x_new, x[i])

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
            exponents = -0.5 * torch.einsum("ijl, lk, ijk -> ij", diff, D_inv/(2*self.dt), diff)
            logQ[i] = torch.logsumexp(exponents, dim = 1) - self.model.n/2 * np.log(4*np.pi*self.dt) - self.model.n/2 * np.log(self.k) - np.log(self.N)

        self.logq = logQ

        return 
    

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
    


    def getFlow(self, numpy : bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            numpy: bool If True, return a numpy array
        """

        if self.z is None:
            self.flow()

        return self.z.detach().numpy() if numpy else self.z.copy()
    


    def leapfrog(self, q: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: torch.Tensor Position
            p: torch.Tensor Momentum
            dVdq: torch.Tensor Gradient of the potential energy
            t_end: float End time
            dt: float Time step
        """

        q = q.clone()
        p = p.clone()

        p -= 0.5 * self.epsilon *(-self.model.grad_logP_XY(q))

        for _ in range(int(self.dt / self.epsilon)):
            q += self.epsilon * torch.einsum("ij, kj -> ki", self.M_inv, p)
            p -= self.epsilon * (-self.model.grad_logP_XY(q))

        q += self.epsilon * torch.einsum("ij, kj -> ki",self.M_inv, p)
        p -= 0.5 * self.epsilon * (-self.model.grad_logP_XY(q))

        return q, -p



 