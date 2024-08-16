import torch
from tqdm.autonotebook import tqdm
from flow.flow import Flow
from model.model import Model
from torch.distributions.distribution import Distribution
import numpy as np

class MixFlow(Flow):

    def __init__(self, N:int, nStep:int, dt:float, model : Model, q0 : Distribution = torch.distributions.Uniform(-2, 2), k : float = 1, pre_compute : bool = False):
        """
        Args:
            N: int Number of particles
            nStep: int Number of steps
            dt: float Time step
            q0: Distribution Initial distribution
            k: float Diffusion coefficient
        """
        self.k = k
        super().__init__(N, nStep, dt, model, q0, pre_compute)
        


    def flow(self):
        """
        Args:
            model: Model Model to use
            numpy: bool If True, return a numpy array
            k: float Diffusion coefficient
        """

        x = torch.empty((self.nStep, self.N, self.model.n), dtype=torch.float32)
        x[0] = self.q0.sample((self.N, self.model.n))

        D_square_root = np.sqrt(self.k) * torch.eye(self.model.n)
        normalDistribution = torch.distributions.MultivariateNormal(torch.zeros(self.model.n), torch.eye(self.model.n))

        with tqdm(total=self.nStep, desc="Mix flow") as pbar:
            for i in range(self.nStep - 1):
                if i % 2 == 0 :
                    gradVx = self.model.grad_logP_XY(x[i])
                    x[i+1] = x[i] + self.dt * gradVx

                else :
                    xsi = np.sqrt(2*self.dt) * D_square_root @ normalDistribution.sample((self.N,)).T
                    x[i+1] = x[i] + xsi.transpose(0,1)

                pbar.update(1)

        self.x = x

        return
    

    def logQ(self) -> None:

        if self.x is None:
            self.flow()

        xt = self.x
        logQ = torch.empty((self.nStep, self.N), dtype=torch.float32)
        logQ[0] = self.q0.log_prob(xt[0]).sum(dim = 1)

        D_inv = (1/self.k) * torch.eye(self.model.n)

        for i in range(1, self.nStep):

            if i % 2 == 0 :
                logQ[i] = logQ[i-1] - self.dt * (self.model.laplace_logP_XY(xt[i-1]).squeeze() + self.model.laplace_logP_XY(xt[i]).squeeze()) / 2
            else :
                diff = xt[i, :, None, :] - xt[i-1, None, :, :]
                exponents = -.5 * torch.einsum("ijl, lk, ijk -> ij", diff, D_inv/(2*self.dt), diff) 
                logQ[i] = torch.logsumexp(exponents, dim = 1) - self.model.n/2 * np.log(4*self.dt*np.pi) - self.model.n/2 * np.log(self.k) - np.log(self.N)


        self.logq = logQ

        return 