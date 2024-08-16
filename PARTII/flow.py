import numpy as np
import torch
from torch.distributions.distribution import Distribution
from tqdm.autonotebook import tqdm
from model import Model
from typing import Union, Tuple
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

        self._w = None
        self._logq = None

        if pre_compute:
            self.flow()
        
    def getFlow(self, numpy : bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            numpy: bool If True, return a numpy array
        """

        if self._w is None:
            self.flow()

        return self._w.detach().numpy() if numpy else self._w.detach()
    

    def getlogQ(self, numpy : bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        Args:
            numpy: bool If True, return a numpy array
        """

        if self._logq is None:
            self.flow()

        return self._logq.detach().numpy() if numpy else self._logq.detach()
    
        
    @abstractmethod
    def flow(self) -> None:
        pass


class GradientFlow(Flow):

    def flow(self) -> None:

        w = torch.empty((self.nStep, self.N, self.model.d), dtype=torch.float64)
        logQ = torch.empty((self.nStep, self.N), dtype=torch.float64)

        w[0] = self.q0.sample((self.N, self.model.d))
        logQ[0] = self.q0.log_prob(w[0]).sum(dim=1)


        w_compute = w[0].detach().clone().requires_grad_(True)
        gradV = self.model.grad_logP(w_compute)
        laplaceV_new = self.model.laplace_logP(w_compute, gradV).detach()
    
        with tqdm(total=self.nStep, desc="Gradient flow") as pbar:
            for i in range(1, self.nStep):
                
                w_compute = w[i - 1].detach().clone().requires_grad_(True)                
                gradV = self.model.grad_logP(w_compute)

                laplaceV_old = laplaceV_new
                laplaceV_new = self.model.laplace_logP(w_compute, gradV).detach()
                gradV = gradV.detach()

                w[i] = w[i - 1] + self.dt * gradV
                logQ[i] = logQ[i - 1] - self.dt * (laplaceV_new + laplaceV_old) / 2

                pbar.update(1)


        self._w = w
        self._logq = logQ   

        return 
    


class MixFlow(Flow):

    def __init__(self, N:int, nStep:int, dt:float, model : Model, q0 : Distribution = torch.distributions.Uniform(-2, 2), pre_compute : bool = False, k : float = 1e-4):
        """
        Args:
            N: int Number of particles
            nStep: int Number of steps
            dt: float Time step
            q0: Distribution Initial distribution
            alpha: float Mixing parameter
        """
        self.k = k
        super(MixFlow, self).__init__(N, nStep, dt, model, q0, pre_compute)

        


    def flow(self):
        """
        Args:
            model: Model Model to use
            numpy: bool If True, return a numpy array
            k: float Diffusion coefficient
        """

        w = torch.empty((self.nStep, self.N, self.model.d), dtype=torch.float64)
        logQ = torch.empty((self.nStep, self.N), dtype=torch.float64)

        w[0] = self.q0.sample((self.N, self.model.d))
        logQ[0] = self.q0.log_prob(w[0]).sum(dim=1)

        w_compute = w[0].detach().clone().requires_grad_(True)
        gradV = self.model.grad_logP(w_compute)
        laplaceV_new = self.model.laplace_logP(w_compute, gradV).detach()

        D_square_root = np.sqrt(self.k) * torch.eye(self.model.d, dtype = torch.float64)
        D_inv = 1/self.k * torch.eye(self.model.d, dtype = torch.float64)

        normalDistribution = torch.distributions.MultivariateNormal(torch.zeros(self.model.d), torch.eye(self.model.d))

        with tqdm(total=self.nStep, desc="Mix flow") as pbar:
            for i in range(1, self.nStep):
                if i % 2 == 1 :
                    w_compute = w[i - 1].detach().clone().requires_grad_(True)
                    gradV = self.model.grad_logP(w_compute)
                    
                    laplaceV_old = laplaceV_new
                    laplaceV_new = self.model.laplace_logP(w_compute, gradV).detach()
                    gradV = gradV.detach()

                    w[i] = w[i-1] + self.dt * gradV
                    logQ[i] = logQ[i-1] - self.dt * (laplaceV_new + laplaceV_old) / 2
     
                else :

                    xsi = np.sqrt(2*self.dt) * D_square_root @ (normalDistribution.sample((self.N,)).T).double()
                    w[i] = w[i - 1] + xsi.transpose(0,1)

                    diff = w[i, :, None, :] - w[i-1, None, :, :]
                    exponents = -0.5 * torch.einsum("ijl, lk, ijk -> ij", diff, D_inv/(2*self.dt), diff) 

                    logQ[i] = torch.logsumexp(exponents, dim = 1) - self.model.d/2 * np.log(4*np.pi*self.dt) - self.model.d/2 * np.log(self.k) - np.log(self.N)

                pbar.update(1)

        self._w = w
        self._logq = logQ

        return
    



class SplitStepFlow(Flow):

    def __init__(self, N:int, nStep:int, dt:float, model : Model, q0 : Distribution = torch.distributions.Uniform(-2, 2), k : float = 1e-3 , sigma0 : float = 1, phi_n : float = 0.015, alpha_star : float = 0.574, lam : int = 10, burn_in : int = 500, pre_compute : bool = False):
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
        self.sigma0 = sigma0

        self.phi_n = phi_n
        self.alpha_star = alpha_star
        self.lam = lam
        self.burn_in = burn_in

        self._z = None
        
        super().__init__(N, nStep, dt, model, q0, pre_compute)



    @staticmethod
    def h(x: torch.Tensor, y: torch.Tensor, grad_logP: torch.Tensor, sigma: torch.Tensor, A:torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.einsum("ij, ij -> i", y - x - (sigma[:, None] / 4) * torch.einsum("ikj, ik -> ij", A, grad_logP), grad_logP)
    
    def flow(self):
        """
        Args:

        """

        w = torch.empty((self.nStep, self.N, self.model.d), dtype=torch.float64)
        z = torch.empty((self.nStep, self.N, self.model.d), dtype=torch.float64)
        logQ = torch.empty((self.nStep, self.N), dtype=torch.float64)
       
        w[0] = self.q0.sample((self.N, self.model.d))
        z[0] = w[0]
        logQ[0] = self.q0.log_prob(z[0]).sum(dim = 1)

        sigma_s = self.sigma0 * torch.ones(self.N, dtype = torch.float64)
        sigma_R = self.sigma0 * torch.ones(self.N, dtype = torch.float64)
        R = torch.tile(torch.eye(self.model.d, dtype = torch.float64), (self.N, 1, 1))

        D_square_root = np.sqrt(self.k)  * torch.eye(self.model.d, dtype = torch.float64)
        D_inv = (1/self.k) * torch.eye(self.model.d, dtype = torch.float64)
        
        normalDistribution = torch.distributions.MultivariateNormal(torch.zeros(self.model.d), torch.eye(self.model.d))

        with tqdm(total=self.nStep - 1, desc="SplitStep flow") as pbar :

            for i in range(self.nStep - 1):
                w_compute = w[i].detach().clone().requires_grad_(True)
                gradlogP_x = self.model.grad_logP(w_compute).detach()

            
                xsi = torch.sqrt(sigma_R[:, None]) * torch.einsum('ijk,ik->ij', R, normalDistribution.sample((self.N,)).double())
           
                y = w[i] + (sigma_R[:, None] / 2) * torch.einsum("ijk, ikj, ik -> ij", R, R.transpose(1, 2), gradlogP_x) + xsi

                y_compute = y.detach().clone().requires_grad_(True)
                gradlogP_y = self.model.grad_logP(y_compute).detach()

                h_yx = self.h(y, w[i], gradlogP_x, sigma_R, torch.matmul(R, R.transpose(1, 2)))
                h_xy = self.h(w[i], y, gradlogP_y, sigma_R, torch.matmul(R, R.transpose(1, 2)))

                logP_y = self.model.logP(y)
                logP_x = self.model.logP(w[i])

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
                    sigma_R = sigma_s / (1/self.model.d * torch.einsum("ijk, ikj -> i", R, R))

                else :

                    s_delta = torch.sqrt(alpha)[:, None] * (gradlogP_y - gradlogP_x)

                    phi = torch.einsum("ijk, ij -> ik", R.transpose(1, 2), s_delta)
                    inner_phi = torch.einsum("ij, ij -> i", phi, phi)
                    r = 1 / (1 + torch.sqrt(1 / (1 + inner_phi)))

                    R = R - (r / (1 + inner_phi))[:, None, None] * torch.einsum("ij, il -> ijl", torch.einsum("ijk, ik -> ij", R, phi), phi)
                    
                    sigma_s *= (1 + self.phi_n *(alpha - self.alpha_star))
                    sigma_R = sigma_s / (1/self.model.d * torch.einsum("ijk, ikj -> i", R, R))

    
                u = torch.rand(self.N)
                mask = u < alpha

                w[i+1][mask] = y[mask]
                w[i+1][~mask] = w[i][~mask]
                
                eta = np.sqrt(2*self.dt) * D_square_root @ (normalDistribution.sample((self.N,)).T).double()

                z[i+1] = w[i+1] + eta.transpose(0,1)

                diff = z[i+1, :, None, :] - w[i+1, None, :, :]
                exponent = -0.5 * torch.einsum("ijl, lk, ijk -> ij", diff, D_inv/(2*self.dt), diff)
                logQ[i] = torch.logsumexp(exponent, dim = 1) - self.model.d/2 * np.log(4*np.pi*self.dt) - self.model.d/2 * np.log(self.k) - np.log(self.N)


                pbar.update(1)

        self._w = w
        self._z = z
        self._logq = logQ

        return 
    

    def getFlow(self, numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:

        if self._z is None:
            self.flow()

        return self._z.detach().numpy() if numpy else self._z.detach()




class HamiltonFlow(Flow):

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
            self.M_inv = torch.eye(model.d)
        else:
            self.M_inv = M_inv

        self.epsilon = epsilon
        self.k = k

        self._z = None
    

        super().__init__(N, nStep, dt, model, q0, pre_compute)

        
    def flow(self):
        
        w = torch.empty((self.nStep, self.N, self.model.d), dtype=torch.float64)
        z = torch.empty((self.nStep, self.N, self.model.d), dtype=torch.float64)
        logQ = torch.empty((self.nStep, self.N), dtype=torch.float64)


        w[0] = self.q0.sample((self.N, self.model.d)).double()
        z[0] = w[0]
        logQ[0] = self.q0.log_prob(z[0]).sum(dim = 1).double()

        momentum = torch.distributions.MultivariateNormal(loc = torch.zeros(self.model.d), precision_matrix = self.M_inv)
        normalDistribution = torch.distributions.MultivariateNormal(torch.zeros(self.model.d), torch.eye(self.model.d))

        D_square_root = np.sqrt(self.k) * torch.eye(self.model.d, dtype = torch.float64)
        D_inv = (1/self.k) * torch.eye(self.model.d, dtype = torch.float64)

        with tqdm(total=self.nStep, desc="Hamilton flow") as pbar:

            for i in range(self.nStep - 1):
                p0 = momentum.sample((self.N,)).double().requires_grad_(True)
                w_compute = w[i].detach().clone().requires_grad_(True)

                w_new, p_new = self.leapfrog(w_compute, p0)
                w_new = w_new.detach()

                Ux_new = -self.model.logP(w_new)
                Ux_old = -self.model.logP(w[i])

                Kq_new = momentum.log_prob(p_new)
                Kq_old = momentum.log_prob(p0)

                alpha = torch.min(torch.tensor(1.0), torch.exp(Ux_old - Ux_new + Kq_old - Kq_new))

                u = torch.rand(self.N)
                mask = u < alpha

                w[i+1] = torch.where(mask[:, None], w_new, w[i])

                eta = np.sqrt(2*self.dt) * D_square_root @ (normalDistribution.sample((self.N,)).T).double()

                z[i+1] = w[i+1] + eta.transpose(0,1)

                diff = z[i+1, :, None, :] - w[i+1, None, :, :]
                exponents = -0.5 * torch.einsum("ijl, lk, ijk -> ij", diff, D_inv/(2*self.dt), diff)
                logQ[i] = torch.logsumexp(exponents, dim = 1) - self.model.d/2 * np.log(4*np.pi*self.dt) - self.model.d/2 * np.log(self.k) - np.log(self.N)

                pbar.update(1)
            
        self._logq = logQ
        self._w = w
        self._z = z

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

        p -= 0.5 * self.epsilon *(-self.model.grad_logP(q).detach())

        for _ in range(int(self.dt / self.epsilon)):
            q += self.epsilon * torch.einsum("ij, kj -> ki", self.M_inv.double(), p)
            p -= self.epsilon * (-self.model.grad_logP(q).detach())

        q += self.epsilon * torch.einsum("ij, kj -> ki",self.M_inv.double(), p)
        p -= 0.5 * self.epsilon * (-self.model.grad_logP(q).detach())

        return q, -p



