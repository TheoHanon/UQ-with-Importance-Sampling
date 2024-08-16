# Author: Theo Hanon    
# Created : 2024-07-05

import numpy as np
from scipy.integrate import nquad
import matplotlib.animation as animation
from matplotlib import pyplot as plt
import torch
from scipy.linalg import fractional_matrix_power
import tqdm
from scipy.special import logsumexp
from tqdm.autonotebook import tqdm


class Simulation_Parameter:
    """
    A class to handle the initialization and operations of simulation parameters 
    
    Attributes:
    ----------
    nStep : int
        Number of steps in the simulation.
    N : int
        Number of particles or entities in the simulation.
    dt : float
        Time step size.
    n : int
        Dimensionality of x.
    m : int
        Dimensionality of y.
    A : np.ndarray
        Transformation matrix.
    """
    
    def __init__(self, y:float, nStep: int, N: int, dt: float, n: int, m: int, A: np.ndarray, mu_prop: np.ndarray, sigma_prop: np.ndarray, uniform : bool = False, a_unif : float = 2):
        self.y = y
        self.nStep = nStep
        self.N = N
        self.dt = dt    
        self.n = n
        self.m = m
        self.A = A

        self.mu_prop = mu_prop
        self.sigma_prop = sigma_prop
        self.sigma_prop_inv = np.linalg.inv(sigma_prop)
        self.sigma_prop_det = np.linalg.det(sigma_prop)

        self.uniform = uniform
        self.a_unif = a_unif
        self.log_ratios_p0 = None


        if not uniform:
            self.x0 = np.random.multivariate_normal(mu_prop, sigma_prop, N)
            exponents = -0.5 * np.einsum("ij, ij -> i", np.dot(self.x0 - mu_prop, self.sigma_prop_inv), self.x0 - mu_prop)
            self.ratios_q0 = np.exp(exponents[:, None] - exponents[None, :])

        else :
            self.x0 = np.random.uniform([-a_unif]*n, [a_unif]*n, (N, n))
            self.ratios_q0 = np.ones((N, N))
        

    def init(self):
        """
        Initialize the simulation parameters.
        """
        raise NotImplementedError("Not implemented")
    
    def gradient_flow(self):
        """
        Compute the gradient flow of a function V at point x0.

        Parameters:
        ----------
        x0 : np.array
            The initial point.
        y : np.array
            The observation.
        param : Simulation_Parameter
            Parameters of the simulation.

        Returns:
        -------
        np.array
            Particles postion xt following the gradient flow.
        
        """
        N = self.N
        n = self.n
        nStep = self.nStep
        dt = self.dt

        x = np.empty((nStep, N, n))
        x[0] = self.x0

        for i in range(nStep-1):
            x[i+1] = x[i] - self.gradV(x[i]) * dt

        return x
    

    def compute_moment_IP(self, alpha:int, xt, qt):
        """
        Compute the moment of the posterior distribution using importance sampling with normalization.

        Parameters:
        ----------
        alpha : int
            The moment to compute.
        xpred : np.array
            Particles position.
        qt : np.array
            Density function at xt (xpred).
        y : np.array
            The observation.
        param : Simulation_Parameter
            Parameters of the simulation.

        Returns:
        -------
        float
            The moment of the posterior distribution using importance sampling with normalization.    
        """

        def f(x):
            return x**alpha

        p_post = self.p_xy(xt, self.y)
        
        w = (p_post / qt) / np.sum(p_post / qt)
   
  
        return np.sum(f(xt) * w[:, np.newaxis], axis =0)
    

    def compute_moment_IP_with_ratio(self, alpha:list[int], x, wik, IP = True):
        """
        Args:
        alpha: list[int]
        x: (N, n)
        ratios_q: (N, N)
        ratios_p: (N, N)
        """
  
        if IP:
            return (np.sum((x**a) * wik[:, None], axis = 0) for a in alpha)
        
        else:
            return (np.mean(x**a, axis = 0) for a in alpha)
        
        
    def compute_moment_AIS(self, alpha:list[int], wik, xk):
        wik = wik.reshape(-1)
        wik = (wik / np.sum(wik)).reshape(-1, 1)
        xk = xk.reshape(-1, self.n)

        return (np.sum(wik * (xk)**a, axis = 0) for a in alpha)
    

    def compute_ratio_qt_ssf(self, xt2, xt1, D_inv):

        n = self.n
        N = self.N
        dt = self.dt

        diff = xt2[:, None, :] - xt1[None, :, :]
        exponents = -0.5 * np.einsum("ijl, lk, ijk -> ij", diff, D_inv/(2*dt), diff)

        sum_exp = logsumexp(exponents, axis = 1)
        return sum_exp[:, None] - sum_exp[None, :]
    

    def h_MALA(self, x, y, gradVy, sigma, D):
        return 0.5 * np.einsum("ij, ij -> i", y - x - (sigma[:, None]/4)*np.einsum("ikj, ik -> ij", D, gradVy), gradVy)


    def split_step_flow(self, k = 1, phi_n = 0.015, alpha_star = 0.574, burn_in = 500, lam  = 10):

        n = self.n
        N = self.N
        dt = self.dt
        scale = self.scale

        xt = np.empty((self.nStep, N, n))
        zt = np.empty((self.nStep, N, n))
        xt[0, ...] = self.x0
        zt[0, ...] = self.x0

        log_ratios_q = np.log(self.ratios_q0)
        log_ratios_p = self.log_ratios_p0

        m1_IP = np.zeros((self.nStep, n)) 
        m2_IP = np.zeros((self.nStep, n)) 
        m1_NIP = np.zeros((self.nStep, n))
        m2_NIP = np.zeros((self.nStep, n))
        m1_AIP = np.zeros((self.nStep, n))
        m2_AIP = np.zeros((self.nStep, n))

        m1 = self.compute_moment(1)
        m2 = self.compute_moment(2)

        wik = np.zeros((self.nStep, N))
        log_wik = - logsumexp(np.array([log_ratios_p[:, j] + log_ratios_q[j, :] for j in range(N)]), axis = 1)
        wik[0] = np.exp(log_wik)

        m1_IP[0], m2_IP[0] = self.compute_moment_IP_with_ratio([1, 2], zt[0]/scale,   wik[0])
        m1_NIP[0], m2_NIP[0] = self.compute_moment_IP_with_ratio([1, 2], zt[0], wik = None, IP = False)

        sigma_s = 100*dt * np.ones(N)
        sigma_R = 100*dt * np.ones(N)
        R = np.tile(np.eye(n), (N, 1, 1))

        with tqdm(total = self.nStep-1) as pbar:
      
            for i in range(self.nStep-1):

                scale = np.random.normal(self.scale, np.sqrt(.1*self.scale))
                K = np.random.lognormal(k, np.sqrt(.1*k), n)

                
                gradVx = self.gradV_robust(xt[i])

                xsi1 = np.sqrt(sigma_R[:, None]) * np.einsum("ijn, in ->  ij", R, np.random.multivariate_normal(np.zeros(n), np.eye(n), N))
                z = xt[i] - (sigma_R[:, None] / 2) * np.einsum("ijk, ik -> ij", R , np.einsum("ijk, ik -> ij", R.transpose(0, 2, 1), gradVx)) + xsi1

                gradVz = self.gradV_robust(z)
                

                logqzx = self.h_MALA(z, xt[i], gradVx, sigma_R, np.matmul(R, R.transpose(0, 2, 1)))
                logqxz = self.h_MALA(xt[i], z, gradVz, sigma_R, np.matmul(R, R.transpose(0, 2, 1)))

                logpz = self.V(z)
                logpx = self.V(xt[i])

                # print(logpz, logpx, logqzx, logqxz)

                alpha = np.minimum(1, np.exp(np.clip(logqxz - logqzx + logpz - logpx, -700, 700)))


                if i < burn_in:
                    sigma_s *= (1 +phi_n *(alpha - alpha_star))
                    sigma_R = sigma_s

                elif i == burn_in:
                    
                    s_detla = np.sqrt(alpha)[:, None] * (gradVz - gradVx) 
                    inner_s = np.einsum("ij, ij -> i", s_detla, s_detla)
                    r = 1 / (1 + np.sqrt(lam / (lam + inner_s)))

                    R = 1/np.sqrt(lam) * (np.eye(n) - (r / (lam + inner_s))[:, None, None] * np.einsum("ij, il -> ijl", s_detla, s_detla))
                    A = np.sum(np.einsum("ij, il -> ijl", s_detla, s_detla)) + lam * np.eye(n)

                    sigma_s *= (1 + phi_n *(alpha - alpha_star))
                    sigma_R = sigma_s / (1/n * np.einsum("ijk, ijk -> i", R, R.transpose(0, 2, 1)))

                else:

                    s_detla = np.sqrt(alpha)[:, None] * (gradVz - gradVx) 
                    phi = np.einsum("ijk, ij -> ik", R.transpose(0, 2, 1), s_detla)
                    inner_phi = np.einsum("ij, ij -> i", phi, phi)
                    r = 1 / (1 + np.sqrt(1 / (1 + inner_phi)))

                    A = np.sum(np.einsum("ij, il -> ijl", s_detla, s_detla), axis = 0) + lam * np.eye(n)
                    R = R - (r / (1 + inner_phi))[:, None, None] * np.einsum("ij, il -> ijl", np.einsum("ijk, ik -> ij", R, phi), phi)
                    
                    sigma_s *= (1 + phi_n *(alpha - alpha_star))
                    sigma_R = sigma_s / (1/n * np.trace(np.matmul(R, R.transpose(0, 2, 1)), axis1 = 1, axis2 = 2))

    
                u = np.random.uniform(0, 1, N)
                xt[i+1][u < alpha] = z[u < alpha]
                xt[i+1][u >= alpha] = xt[i][u >= alpha]
                
                D_rng = np.random.choice([.1, 0.5, 1, 2, 3], n)
                D_inv =np.diag(1/(K * D_rng))
                D_frac = np.sqrt(K) * np.diag(D_rng**(1/2))
       
                # D_frac = R_mean
                # D_inv = np.linalg.inv(np.matmul(D_frac, D_frac.T))
                # D_logdet = 2*np.linalg.slogdet(D_frac)[1]
                # sigma_D = np.mean(sigma_R)
            
                xsi2 = np.sqrt(2*dt) * D_frac @ np.random.multivariate_normal(np.zeros(n), np.eye(n), N).T

                zt[i+1] = xt[i+1] + xsi2.transpose(1, 0)
                
                vi = self.V(zt[i+1]/scale)

                log_ratios_p = vi[:, None] - vi[None, :]
                log_ratios_q = self.compute_ratio_qt_ssf(zt[i+1], xt[i+1], D_inv = D_inv)

                log_wik = - logsumexp(np.array([log_ratios_p[:, j] + log_ratios_q[j, :] for j in range(N)]), axis = 1)
                max_log_wik = np.max(log_wik)

                wik[i+1] = np.exp(log_wik - max_log_wik) * np.exp(max_log_wik)

                print(f"ite : {i}", np.max(wik[i+1]))
                m1_AIP[i+1], m2_AIP[i+1] = self.compute_moment_AIS([1, 2], wik[:i+1], zt[:i+1]/scale)
                m1_IP[i+1], m2_IP[i+1] = self.compute_moment_IP_with_ratio([1, 2], zt[i+1]/scale, wik[i+1])
                m1_NIP[i+1], m2_NIP[i+1] = self.compute_moment_IP_with_ratio([1, 2], xt[i+1:].reshape(-1, n), wik = None, IP = False)

                err1 = np.mean(np.abs(m1_AIP[i+1] - m1))
                err2 = np.mean(np.abs(m2_AIP[i+1] - m2))

                pbar.update(1)
                pbar.set_postfix( {"E1" : f"{err1:.3e}", "E2" : f"{err2:.3e}"} )

            
        return xt, m1_IP, m2_IP, m1_NIP, m2_NIP, m1_AIP, m2_AIP
    

    def mix_flow(self, k=1) :

        n = self.n
        N = self.N
        dt = self.dt

        xt = np.empty((self.nStep, N, n))
        xt[0, ...] = self.x0
        D = k*np.ones(n)


        for i in range(self.nStep-1):
            if (i+2) % 2 == 0:
                gradVx = self.gradV_robust(xt[i])
                xt[i+1] = xt[i] - gradVx * dt
            else :
                xsi = np.sqrt(2*dt) * np.diag(D**(1/2)) @ np.random.multivariate_normal(np.zeros(n), np.eye(n), N).T
                xt[i+1] = xt[i] + xsi.transpose(1, 0)
   
        return xt

    


    def p_x(self, x):
        raise NotImplementedError("Not implemented")

    def p_yx(self, y, x):
        raise NotImplementedError("Not implemented")

    def p_y(self, y):
        raise NotImplementedError("Not implemented")

    def p_xy(self, x, y):
        raise NotImplementedError("Not implemented")
    

    def laplaceV(self, x, y):
        raise NotImplementedError("Not implemented")

    def gradV(self, x, y):
        raise NotImplementedError("Not implemented")
    
    def V(self, x, y):
        raise NotImplementedError("Not implemented")
    
    def gradV_robust(self, x, y):
        raise NotImplementedError("Not implemented")
    

class Simulation_Parameter_Gaussian(Simulation_Parameter):

    """
    A class to handle the initialization and operations of simulation parameters for a Gaussian model.
    
    Attributes:
    ----------
    var_x : float
        Variance for x.
    var_yx : float
        Variance for y given x.
    """

    def __init__(self, y:float, nStep: int, N: int, dt: float, n: int, m: int, A: np.ndarray, mu_prop : np.ndarray, sigma_prop: np.ndarray, sigma_x: np.ndarray, sigma_yx: np.ndarray, uniform : bool = False, a_unif : float = 2, scale : float = 1):
        super().__init__(y, nStep, N, dt, n, m, A, mu_prop, sigma_prop, uniform, a_unif)
        self.sigma_x = sigma_x
        self.sigma_yx = sigma_yx
        self._init_gaussian()

        self.scale = scale
        v0 = self.V(self.x0/scale)
        self.log_ratios_p0 = v0[:, None] - v0[None, :]

        # self.xt = self.gradient_flow()
        # self.qt = self.compute_qt(self.xt)
    
    def _init_gaussian(self):
        """
        Initialize Gaussian matrices and their inverses and determinants.

        Parameters:
        ----------
        var_x : float
            Variance for x.
        var_yx : float
            Variance for y given x.
        n : int
            Dimensionality of x.
        m : int
            Dimensionality of y.
        """
        self.sigma_x_inv = np.linalg.inv(self.sigma_x)
        self.sigma_x_det = np.linalg.det(self.sigma_x)

        self.sigma_yx_inv = np.linalg.inv(self.sigma_yx) 
        self.sigma_yx_det = np.linalg.det(self.sigma_yx)

        self.sigma_y = self.sigma_yx + self.A @ self.sigma_x @ self.A.T
        self.sigma_y_inv = np.linalg.inv(self.sigma_y)
        self.sigma_y_det = np.linalg.det(self.sigma_y)

        self.sigma_inv = self.sigma_x_inv + self.A.T @ self.sigma_yx_inv @ self.A
        self.sigma = np.linalg.inv(self.sigma_inv)
        self.sigma_det = np.linalg.det(self.sigma)

        self.mu_xy = self.sigma * (self.A).T @ self.sigma_yx_inv @ self.y

    def p_x(self, x):
        p = 1/((2*np.pi)**(self.n/2)) \
                * 1/np.sqrt(self.sigma_x_det) \
                    * np.exp(-0.5 * np.einsum('ij, ij->i', np.dot(x, self.sigma_x_inv), x))
        return  p

    def p_yx(self, y, x):
        diff = y - np.dot(x, self.A.T) #because x is (N, n)
        p = 1/((2*np.pi)**(self.m/2)) \
                * 1/np.sqrt(self.sigma_yx_det) \
                        * np.exp(-0.5 * np.einsum("ij, ij->i",np.dot(diff, self.sigma_yx_inv), diff))

        return p

    def p_y(self, y):
        p = 1/((2*np.pi)**(self.m/2)) \
                * 1/np.sqrt(self.sigma_y_det) \
                        * np.exp(-0.5 * np.dot(np.dot(y, self.sigma_y_inv), y))
        return p

    def p_xy(self, x, y):
        p = self.p_yx(y, x) * self.p_x(x) / self.p_y(y)
        return p
    
    def laplaceV(self, x):
        return self.sigma
    
    def gradV(self, x):
        gradV = np.dot(x, self.sigma_inv.T) - np.dot(self.A.T, np.dot(self.sigma_yx_inv, self.y))

        return gradV

    def V(self, x):
        diff = x - self.mu_xy
        return -0.5 * np.einsum("ij, ij -> i", np.dot(diff, self.sigma_inv), diff)
    
    def compute_moment(self, alpha:int):
        if alpha == 1:
            return self.mu_xy
        
        elif alpha == 2:
            return self.sigma + np.outer(self.mu_xy,  self.mu_xy)

        else :
            raise ValueError("Moment not implemented")

class Simulation_Parameter_Gaussian_Mixture(Simulation_Parameter):
    """ 
    A class to handle the initialization and operations of simulation parameters for a Gaussian mixture model.

    Attributes:
    ----------
    weight : np.ndarray
        Weights of the Gaussian components.
    mu_x : np.ndarray
        Means of the Gaussian components.
    var_x : np.ndarray
        Variances of the Gaussian components.
    var_yx : float
        Variance for y given x.
    """

    def __init__(self, y:float, nStep: int, N: int, dt: float, n: int, m: int, A : np.ndarray, mu_prop:np.ndarray, sigma_prop : np.ndarray, weight: np.ndarray, sigma_x: np.ndarray, mu_yx: np.ndarray, sigma_yx: np.ndarray, uniform : bool = False, a_unif : float = 2, scale : float = 1):

        super().__init__(y, nStep, N, dt, n, m, A, mu_prop, sigma_prop, uniform, a_unif)

        self.weight = weight 
        self.nComp = len(weight)
        self.mu_yx = mu_yx
        self.sigma_yx = sigma_yx

        self.sigma_x = sigma_x
        self.y = y
        self.mu_prop = mu_prop
        self.sigma_prop = sigma_prop

        if not uniform:
            self.x0 = np.random.multivariate_normal(mu_prop, sigma_prop, N)
        else :
            self.x0 = np.random.uniform([-a_unif]*n, [a_unif]*n, (N, n))
        self._init_gaussian_mixture()


        self.scale = scale
        v0 = self.V(self.x0/scale)
        self.log_ratios_p0 =v0[:, None] - v0[None, :]

        # self.xt = self.gradient_flow()
        # self.qt = self.compute_qt(self.xt)


    def _init_gaussian_mixture(self):

        self.sigma_yx_inv = np.array([np.linalg.inv(sigma) for sigma in self.sigma_yx])
        self.sigma_yx_logdet = np.array([np.linalg.slogdet(sigma)[1] for sigma in self.sigma_yx])

    
        self.sigma_x_inv = np.linalg.inv(self.sigma_x)
        self.sigma_x_logdet = np.linalg.slogdet(self.sigma_x)[1]

        self.sigma_y = np.array([sigma_yx + self.A @ self.sigma_x @ self.A.T for sigma_yx in self.sigma_yx])
        self.sigma_y_inv = np.array([np.linalg.inv(sigma) for sigma in self.sigma_y])
        self.sigma_y_logdet = np.array([np.linalg.slogdet(sigma)[1] for sigma in self.sigma_y])

        self.sigma_inv = np.array([self.sigma_x_inv + (self.A).T @ sigma_yx_inv @ self.A for sigma_yx_inv in self.sigma_yx_inv])
        self.sigma = np.array([np.linalg.inv(sigma) for sigma in self.sigma_inv])
        self.sigma_logdet = np.array([np.linalg.slogdet(sigma)[1] for sigma in self.sigma])
        self.mu_xy = np.array([sigma @ (self.A).T @ sigma_yx_inv @ (self.y - mu_yx)  for (sigma, sigma_yx_inv, mu_yx) in zip(self.sigma, self.sigma_yx_inv, self.mu_yx)])
        

    def p_x(self, x):
        exponent = -.5 * np.einsum("ij,ij->i", np.dot(x, self.sigma_x_inv), x)
        log_prefactor = -self.n/2 * np.log(2*np.pi) - 0.5 * self.sigma_x_logdet
        return np.exp(exponent + log_prefactor)

    def p_yx(self, y, x):
        p = np.zeros(x.shape[0])
        for w, mu_yx, sigma_yx_inv, sigma_yx_logdet in zip(self.weight, self.mu_yx, self.sigma_yx_inv, self.sigma_yx_logdet):
            diff = y - np.dot(x, self.A.T) - mu_yx
            exponent = -0.5 * np.einsum("ij,ij->i", np.dot(diff, sigma_yx_inv), diff)
            log_prefactor = -self.m/2 * np.log(2*np.pi) - 0.5 * sigma_yx_logdet
            p += w * np.exp(exponent + log_prefactor)

        return p
    

    def p_y(self, y):
        p = 0
        for w, mu, sigma_y_inv, sigma_y_logdet in zip(self.weight, self.mu_yx, self.sigma_y_inv, self.sigma_y_logdet):
            diff = y + mu
            exponent = -0.5 * np.dot(np.dot(diff, sigma_y_inv), diff)
            log_prefactor = -self.m/2 * np.log(2*np.pi) - 0.5 * sigma_y_logdet
            p += w * np.exp(exponent + log_prefactor)
        return p
    
    def p_xy(self, x, y):  
        logp = np.array([self.logp_xy_i(x, i) + np.log(self.weight[i]) for i in range(self.nComp)])
        return np.exp(logsumexp(logp, axis = 0))
    
    
    def logp_xy_i(self, x, i):
        sigma_inv_i = self.sigma_inv[i]
        sigma_logdet_i = self.sigma_logdet[i]
        mu_xy_i = self.mu_xy[i]
        diff = x - mu_xy_i

        exponent = -0.5 * np.einsum("ij,ij->i", np.dot(diff, sigma_inv_i), diff)
        log_prefactor = -self.n/2 * np.log(2*np.pi) - 0.5 * sigma_logdet_i

        return exponent + log_prefactor


    def p_xy_i(self, x, i):
        sigma_inv_i = self.sigma_inv[i]
        sigma_logdet_i = self.sigma_logdet[i]
        mu_xy_i = self.mu_xy[i]
        diff = x - mu_xy_i

        exponent = -0.5 * np.einsum("ij,ij->i", np.dot(diff, sigma_inv_i), diff)
        log_prefactor = -self.n/2 * np.log(2*np.pi) - 0.5 * sigma_logdet_i


        return np.exp(exponent + log_prefactor)

        
    def laplaceV(self, x):

        """
        laplaceV = - (A*b - C x D) / b**2

        """
        n = self.n
        N = self.N
        nComp = self.nComp
        wi = self.weight
        
        A = np.zeros((N, n, n))
        b = np.zeros(N)
        C = np.zeros((N, n))
        D = np.zeros((N, n))

        for i in range(nComp):

            pi = self.p_xy_i(x, i)
            grad_pi = -pi[:, None] * np.einsum("ij, kj -> ki", self.sigma_inv[i], x-self.mu_xy[i])
            
            A += wi[i] * (pi[:, None, None] * self.sigma_inv[i] + np.einsum("ki, kj -> kij", grad_pi, np.dot(x-self.mu_xy[i],self.sigma_inv[i].T)))
            b += wi[i] * pi
            C += wi[i] * pi[:, None] * np.dot(x-self.mu_xy[i],self.sigma_inv[i].T)
            D += wi[i] * grad_pi

        laplaceV = (A * b[:, None, None] - np.einsum("ki, kj -> kij", C, D)) / np.clip(b[:, None, None]**2, 1e-12, None)

        return laplaceV

                  
    def gradV(self, x):
        num = np.zeros((self.N, self.n))
        den = np.zeros((self.N))

        for i in range(self.nComp):
            pi = self.p_xy_i(x, i)
            grad_pi = -pi[:, None] * np.dot(x - self.mu_xy[i], self.sigma_inv[i].T)
            num += self.weight[i, None] * grad_pi
            den += self.weight[i] * pi

        sign = np.sign(den)
        den = np.clip(np.abs(den), 1e-12, None)
        grad = - num / den[:, None] * sign[:, None]

        return grad
    

    def gradV_robust(self, x):

        n = self.n
        N = self.N
        nComp = self.nComp
        w = self.weight

        num = np.zeros((N, n))
        log_den = np.full(N, -np.inf)

        for i in range(nComp):
            log_pi = self.logp_xy_i(x, i) + np.log(w[i])
            pi = np.exp(log_pi - logsumexp([log_pi]))

            grad_pi = -pi[:, None] * np.dot(x - self.mu_xy[i], self.sigma_inv[i].T)
            num += self.weight[i] * grad_pi
            log_den = np.logaddexp(log_den, log_pi)

        max_den = np.max(log_den)
        den = np.exp(log_den - max_den)
        den = np.where(den < 1e-12, 1e-12, den)
        
        return - (num / den[:, None]) * np.exp(max_den)

    
    def V(self, x):
        log_probs = np.array([self.logp_xy_i(x, i) + np.log(self.weight[i]) for i in range(self.nComp)])
        return logsumexp(log_probs, axis = 0)


    def compute_moment(self, alpha:int):
        if alpha == 1:
            return np.sum([w * mu for w, mu in zip(self.weight, self.mu_xy)], axis = 0)
        
        elif alpha == 2:
            m2 = np.zeros((self.n, self.n))
            for w, sigma, mu in zip(self.weight, self.sigma, self.mu_xy):
                m2 += w * (sigma + np.outer(mu, mu))
            return m2
            # return np.sum([w[None, None] * (sigma + np.outer(mu, mu)) for w, sigma, mu in zip(self.weight, self.sigma, self.mu_xy)], axis = 0)

        else :
            raise ValueError("Moment not implemented")

    
    


    
    
    

