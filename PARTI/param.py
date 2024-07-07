# Author: Theo Hanon    
# Created : 2024-07-05

import numpy as np
from scipy.integrate import quad
from autograd import elementwise_grad, jacobian
import matplotlib.animation as animation
from matplotlib import pyplot as plt
import tqdm

import jax
import jax.numpy as jnp


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
    
    def __init__(self, nStep: int, N: int, dt: float, n: int, m: int, A: np.ndarray):
        self.nStep = nStep
        self.N = N
        self.dt = dt    
        self.n = n
        self.m = m
        self.A = A


    def div(self, f, x):
        """
        Compute the divergence of a function f at point x.
        
        Parameters:
        ----------
        f : function
            The function to compute the divergence of.
        x : np.array
            The point at which to compute the divergence.

        Returns:
        -------
        float
            The divergence of f at x.

        """

        jac_f = jax.jacobian(f)
        jac_matrix = jac_f(x)

        if self.n == 1:
            return np.diag(jac_matrix.squeeze())
        else:
            div = jnp.trace(jac_matrix, axis1=0, axis2=1)
            return div
    

    def gradient_flow(self, x0, y):
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

        n = self.n
        N = self.N
        nStep = self.nStep
        dt = self.dt

        x = np.empty((n, N, nStep))
        x[..., 0] = x0

        for i in range(nStep-1):
            x[..., i+1] = x[..., i] - self.gradV(x[..., i], y) * dt

        return x

    def langevin_sampling(self, x0, y, D = 10):
        """
        Compute the langevin diffusion of a function V at point x0.

        Parameters:
        ----------
        x0 : np.array
            The initial point.
        y : np.array
            The observation.
        param : Simulation_Parameter
            Parameters of the simulation.
        D : float
            The diffusion coefficient.

        Returns:
        -------
        np.array
            Particles postion xt following the langevin diffusion.
        """

        n = self.n
        N = self.N
        nStep = self.nStep
        dt = self.dt

        x = np.empty((n, N, nStep))
        x[..., 0] = x0

        for i in range(nStep-1):
            x[..., i+1] = x[..., i] - self.gradV(x[..., i], y) * dt + np.random.normal(0, 1, (n, N)) * np.sqrt(2*D*dt)

        return x
    

    def compute_qt(self, mu_prop, var_prop, xpred, y):
        """
        Compute the importance weights qt for a given proposal distribution.

        Parameters:
        ----------
        mu_prop : float
            Mean of the proposal distribution.
        var_prop : float
            Variance of the proposal distribution.
        xpred : np.array
            Particles position.
        y : np.array
            The observation.
        param : Simulation_Parameter
            Parameters of the simulation.
        
        Returns:
        -------
        np.array
            Density evalution at the particles position xt (xpred).
        """

        n = self.n
        m = self.m
        N = self.N
        nStep = self.nStep
        dt = self.dt

        f = lambda x:  -self.gradV(x, y) 
        qt = np.zeros((n, N, nStep))

        qt[..., 0] = 1/((2*np.pi)**(n/2)*np.sqrt(var_prop)) * np.exp(-0.5 * (xpred[...,0] - mu_prop)**2 / var_prop)

        I = np.zeros((n, N))

        for k in tqdm.tqdm(range(1, nStep)):
            xold = xpred[..., k-1]
            xnew = xpred[..., k]
        
            diva = self.div(f, xold)
            divb = self.div(f, xnew)

            I += dt * (diva + divb) / 2

            if np.isnan(I).any():
                print("k = ", k)
                # print(diva, divb)

            # print(np.isnan(I).any())
            

            qt[..., k] = qt[..., 0] * np.exp(-I)

        return qt
    
    def compute_moment(self, alpha:int, y):
        """
        Compute the exact moment of the posterior distribution.

        Parameters:
        ----------
        alpha : int
            The moment to compute.
        y : np.array
            The observation.
        param : Simulation_Parameter
            Parameters of the simulation.
        
        Returns:
        -------
        float
            The moment of the posterior distribution.

        """

        f = lambda x: np.array(x**alpha * self.p_xy(np.array([[x]]), y))
        return quad(f, -np.inf, np.inf)[0]
    
    def compute_moment_IP(self, alpha:int, xpred, qt, y):
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

        f = lambda x : x**alpha
        p_post = np.array(self.p_xy(xpred, y))
        qt = np.array(qt)
        # print(np.isnan(qt).any())
        w = (p_post / qt) / np.sum(p_post/qt, axis = 1)
    
    
        return np.sum((f(xpred) * w), axis = 1)
    

    def p_x(self, x):
        raise NotImplementedError("Not implemented")

    def p_yx(self, y, x):
        raise NotImplementedError("Not implemented")

    def p_y(self, y):
        raise NotImplementedError("Not implemented")

    def p_xy(self, x, y):
        raise NotImplementedError("Not implemented")

    def gradV(self, x, y):
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

    def __init__(self, nStep: int, N: int, dt: float, n: int, m: int, A: np.ndarray, var_x: float, var_yx: float):
        super().__init__(nStep, N, dt, n, m, A)
        self.var_x = var_x
        self.var_yx = var_yx

        self._init_gaussian()

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
        self.sigma_x = self.var_x * np.eye(self.n) 
        self.sigma_x_inv = np.linalg.inv(self.sigma_x)
        self.sigma_x_det = np.linalg.det(self.sigma_x)

        self.sigma_yx = self.var_yx * np.eye(self.m) 
        self.sigma_yx_inv = np.linalg.inv(self.sigma_yx) 
        self.sigma_yx_det = np.linalg.det(self.sigma_yx)

        self.sigma_y = self.sigma_yx + self.A @ self.sigma_x @ self.A.T
        self.sigma_y_inv = np.linalg.inv(self.sigma_y)
        self.sigma_y_det = np.linalg.det(self.sigma_y)

        self.sigma_inv = self.sigma_x_inv + self.A.T @ self.sigma_yx_inv @ self.A
        self.sigma = np.linalg.inv(self.sigma_inv)
        self.sigma_det = np.linalg.det(self.sigma)

    def p_x(self, x):
        p = 1/((2*np.pi)**(self.n/2)) \
                * 1/np.sqrt(self.sigma_x_det) \
                    * np.exp(-0.5 * np.einsum("ij, ji->i",x.T, self.sigma_x_inv @ x))
        return  p

    def p_yx(self, y, x):
        p = 1/((2*np.pi)**(self.m/2)) \
                * 1/np.sqrt(self.sigma_yx_det) \
                        * np.exp(-0.5 * np.einsum("ij, ji->i",(y - self.A @ x).T, self.sigma_yx_inv @ (y - self.A @ x)))

        return p


    def p_y(self, y):
        p = 1/((2*np.pi)**(self.m/2)) \
                * 1/np.sqrt(self.sigma_y_det) \
                        * np.exp(-0.5 * np.einsum("ij, ji->i",y.T, self.sigma_y_inv @ y))
        return p


    def p_xy(self, x, y):
        p = self.p_yx(y, x) * self.p_x(x) / self.p_y(y)
        # print(np.info(p))
        return p.reshape(-1)
    
        
    def gradV(self, x, y):
        gradV =  self.sigma_inv @ x - (self.A).T @ self.sigma_yx_inv @ y
        return gradV
    

    def compute_qt(self, mu_prop, var_prop, xpred, y):
        """
        Compute the importance weights qt for a given proposal distribution.

        Parameters:
        ----------
        mu_prop : float
            Mean of the proposal distribution.
        var_prop : float
            Variance of the proposal distribution.
        xpred : np.array
            Particles position.
        y : np.array
            The observation.
        param : Simulation_Parameter
            Parameters of the simulation.
        
        Returns:
        -------
        np.array
            Density evalution at the particles position xt (xpred).
        """

        n = self.n
        m = self.m
        N = self.N
        nStep = self.nStep
        dt = self.dt

        f = lambda x:  -self.gradV(x, y) ## set y = 0 for simplicity
        qt = np.zeros((n, N, nStep))

        qt[..., 0] = 1/((2*np.pi)**(n/2)*np.sqrt(var_prop)) * np.exp(-0.5 * (xpred[...,0] - mu_prop)**2 / var_prop)

        I = np.zeros((n, N))

        for k in tqdm.tqdm(range(1, nStep)):
            xold = xpred[..., k-1]
            xnew = xpred[..., k]
            
            diva = -np.trace(self.sigma_inv)
            divb = -np.trace(self.sigma_inv)

            I += dt * (diva + divb) / 2
            qt[..., k] = qt[..., 0] * np.exp(-I)

        return qt




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

    def __init__(self, nStep: int, N: int, dt: float, n: int, m: int, A : np.ndarray, weight: np.ndarray, mu_x: np.ndarray, var_x: np.ndarray, var_yx: np.ndarray):

        super().__init__(nStep, N, dt, n, m, A)

        self.weight = weight    
        self.mu_x = mu_x
        self.var_yx = var_yx
        self.var_x = var_x
        

        self._init_gaussian_mixture()


    def _init_gaussian_mixture(self):

        self.sigma_x = np.array([var * np.eye(self.n) for var in self.var_x])
        self.sigma_x_inv = np.array([np.linalg.inv(sigma) for sigma in self.sigma_x])
        self.sigma_x_det = np.array([np.linalg.det(sigma) for sigma in self.sigma_x])

        self.sigma_yx = np.eye(self.m) * self.var_yx
        self.sigma_yx_inv = np.linalg.inv(self.sigma_yx)
        self.sigma_yx_det = np.linalg.det(self.sigma_yx)

        self.sigma_y = np.array([self.sigma_yx + self.A @ sigma_x @ self.A.T for sigma_x in self.sigma_x])
        self.sigma_y_inv = np.array([np.linalg.inv(sigma) for sigma in self.sigma_y])
        self.sigma_y_det = np.array([np.linalg.det(sigma) for sigma in self.sigma_y])

    def p_x(self, x):
        p = jnp.zeros_like(x)  
        for w, mu, det, sigma_inv in zip(self.weight, self.mu_x, self.sigma_x_det, self.sigma_x_inv):
            diff = x - mu  
            exponent = -0.5 * jnp.einsum("ij,ji->i", diff.T, sigma_inv @ diff)
            p += w * 1/((2*jnp.pi)**(self.n/2)) * 1/jnp.sqrt(det) * jnp.exp(exponent)
        return p

    def p_yx(self, y, x):
        diff = y - self.A @ x 
        exponent = -0.5 * jnp.einsum("ij,ji->i", diff.T, self.sigma_yx_inv @ diff).reshape(self.n, -1)
        p = 1/((2*jnp.pi)**(self.m/2)) * 1/jnp.sqrt(self.sigma_yx_det) * jnp.exp(exponent)
        return p

    def p_y(self, y):
        p = jnp.zeros_like(y)
        for w, mu, sigma_y_inv, sigma_y_det in zip(self.weight, self.mu_x, self.sigma_y_inv, self.sigma_y_det):
            diff = y - self.A * mu #here mu is 1d : TO CHANGE 
            exponent = -0.5 * jnp.einsum("ij,ji->i", diff.T, sigma_y_inv @ diff)

            p += w * 1/((2*jnp.pi)**(self.m/2)) * 1/jnp.sqrt(sigma_y_det) * jnp.exp(exponent)
        return p
    
    def p_xy(self, x, y):
        p = self.p_yx(y, x) * self.p_x(x) / self.p_y(y)
        return p.reshape(-1)
    
    def gradx_p_x(self, x):
        grad = jnp.zeros_like(x)
        for w, mu, det, sigma_inv in zip(self.weight, self.mu_x, self.sigma_x_det, self.sigma_x_inv):
            diff = x - mu
            grad += w * sigma_inv @ diff * jnp.exp(-0.5 * jnp.einsum("ij,ji->i", diff.T, sigma_inv @ diff)) * 1/((2*jnp.pi)**(self.n/2) * jnp.sqrt(det))
        return grad
    
    def gradx_p_yx(self, y, x):
        grad = -self.A.T@self.sigma_yx_inv@(y - self.A@x) * jnp.exp(-0.5 * jnp.einsum("ij,ji->i", (y - self.A@x).T, self.sigma_yx_inv@(y - self.A@x))) * 1/((2*jnp.pi)**(self.m/2) * jnp.sqrt(self.sigma_yx_det))
        return grad
        
    def gradV(self, x, y):
        num = self.gradx_p_x(x) * self.p_yx(y, x) + self.gradx_p_yx(y, x)*self.p_x(x)
        den = jnp.clip(self.p_yx(y, x)*self.p_x(x), 1e-15, None)

        return num/den
    
    


    
    
    

