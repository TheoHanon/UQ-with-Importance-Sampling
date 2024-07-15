# Author: Theo Hanon    
# Created : 2024-07-05

import numpy as np
from scipy.integrate import nquad
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

        if not uniform:
            self.x0 = np.random.multivariate_normal(mu_prop, sigma_prop, N)
        else :
            self.x0 = np.random.uniform([-a_unif]*n, [a_unif]*n, (N, n))


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

        jac_f = jax.jacfwd(f)
        jac_matrix = jac_f(x)
        # print(jac_matrix.shape)

        if self.n == 1:
            return np.diag(jac_matrix.squeeze())
        else:
            div = jnp.trace(jnp.diagonal(jac_matrix, axis1=0, axis2=2).T, axis1=1, axis2 =2)
            return div
    
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
    
    
    def langevin_sampling(self, D = 10):
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

        x = np.empty((nStep, N, n))
        x[0] = self.x0

        for i in range(nStep-1):
            x[i+1, ...] = x[i, ...] - self.gradV(x[i, ...]) * dt + np.random.normal(0, 1, n) * np.sqrt(2*D*dt)

        return x
    

    def compute_qt(self, xt):
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

    
        f = lambda x:  -self.gradV(x) 
        qt = np.zeros((nStep, N))

        diff = xt[0, :] - self.mu_prop
        if not self.uniform:
            qt[0, :] = 1/((2*np.pi)**(n/2)*np.sqrt(self.sigma_prop_det)) * np.exp(-0.5 * np.einsum('ij, ij->i', np.dot(diff, self.sigma_prop_inv), diff))
        else:
            qt[0, :] = 1/(2*self.a_unif)**(self.n) * np.ones(N)

        I = np.zeros(N)

        for k in tqdm.tqdm(range(1, nStep)):
            xold = xt[k-1, ...]
            xnew = xt[k, ...]
        
            diva = self.div(f, (xold))
            divb = self.div(f, (xnew))

            I += dt * (diva + divb) / 2
            qt[k, :] = qt[0, :] * np.exp(-I)

        return qt
    
    
    def compute_moment(self, alpha:int):
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

        m = np.zeros(self.n)

        for i in range(self.n):
            def f(*x) :
                x_vec = np.array(x)
                return np.array(np.prod(x_vec[i]**alpha) * self.p_xy(np.array([x_vec]), self.y))
            m[i] = nquad(f, [(-np.inf, np.inf)]*self.n)[0]
       
        return m

    
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

    def __init__(self, y:float, nStep: int, N: int, dt: float, n: int, m: int, A: np.ndarray, mu_prop : np.ndarray, sigma_prop: np.ndarray, sigma_x: np.ndarray, sigma_yx: np.ndarray, uniform : bool = False, a_unif : float = 2):
        super().__init__(y, nStep, N, dt, n, m, A, mu_prop, sigma_prop, uniform, a_unif)
        self.sigma_x = sigma_x
        self.sigma_yx = sigma_yx
        self._init_gaussian()

        self.xt = self.gradient_flow()
        self.qt = self.compute_qt(self.xt)


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
    

    def gradV(self, x):
        gradV =  np.dot(x, self.sigma_inv.T) - np.dot(self.A.T, np.dot(self.sigma_yx_inv, self.y))

        return gradV
    
    def compute_qt(self, xt):
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

        qt = np.zeros((nStep, N))
        I = np.zeros(N)

        if not self.uniform:
            diff = xt[0, :] - self.mu_prop
            qt[0, :] = 1/((2*np.pi)**(n/2)*np.sqrt(self.sigma_prop_det)) * np.exp(-0.5 * np.einsum('ij, ij->i', np.dot(diff, self.sigma_prop_inv), diff))
        else:
            qt[0, :] = 1/(2*self.a_unif)**(self.n) * np.ones(N)

        for k in range(1, nStep):
            diva = -np.trace(self.sigma_inv)
            divb = -np.trace(self.sigma_inv)

            I += dt * (diva + divb) / 2
            qt[k, :] = qt[0, :] * np.exp(-I)
            
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

    def __init__(self, y:float, nStep: int, N: int, dt: float, n: int, m: int, A : np.ndarray, mu_prop:np.ndarray, sigma_prop : np.ndarray, weight: np.ndarray, sigma_x: np.ndarray, mu_yx: np.ndarray, sigma_yx: np.ndarray, uniform : bool = False, a_unif : float = 2):

        super().__init__(y, nStep, N, dt, n, m, A, mu_prop, sigma_prop, uniform, a_unif)

        self.weight = weight 

        self.mu_yx = mu_yx
        self.sigma_yx = sigma_yx

        self.sigma_x = sigma_x
        self.y = y
        self.mu_prop = mu_prop
        self.sigma_prop = sigma_prop

        self.x0 = np.random.multivariate_normal(mu_prop, sigma_prop, N)
        self._init_gaussian_mixture()

        self.xt = self.gradient_flow()
        self.qt = self.compute_qt(self.xt)


    def _init_gaussian_mixture(self):

        self.sigma_yx_inv = np.array([np.linalg.inv(sigma) for sigma in self.sigma_yx])
        self.sigma_yx_det = np.array([np.linalg.det(sigma) for sigma in self.sigma_yx])

        self.sigma_x_inv = np.linalg.inv(self.sigma_x)
        self.sigma_x_det = np.linalg.det(self.sigma_x)

        self.sigma_y = np.array([sigma_yx + self.A @ self.sigma_x @ self.A.T for sigma_yx in self.sigma_yx])
        self.sigma_y_inv = np.array([np.linalg.inv(sigma) for sigma in self.sigma_y])
        self.sigma_y_det = np.array([np.linalg.det(sigma) for sigma in self.sigma_y])

        self.sigma_inv = np.array([self.sigma_x_inv + (self.A).T @ sigma_yx_inv @ self.A for sigma_yx_inv in self.sigma_yx_inv])
        self.sigma = np.array([np.linalg.inv(sigma) for sigma in self.sigma_inv])
        self.sigma_det = np.array([np.linalg.det(sigma) for sigma in self.sigma])
        self.mu_xy = np.array([sigma @ (self.A).T @ sigma_yx_inv @ (self.y - mu_yx)  for (sigma, sigma_yx_inv, mu_yx) in zip(self.sigma, self.sigma_yx_inv, self.mu_yx)])
        

    def p_x(self, x):
        exponent = -.5 * jnp.einsum("ij,ij->i", jnp.dot(x, self.sigma_x_inv), x)
        return 1/((2*jnp.pi)**(self.n/2)) * 1/jnp.sqrt(self.sigma_x_det) * jnp.exp(exponent)

    def p_yx(self, y, x):
        p = jnp.zeros(x.shape[0])
        for w, mu_yx, sigma_yx_inv, sigma_yx_det in zip(self.weight, self.mu_yx, self.sigma_yx_inv, self.sigma_yx_det):
            diff = y - jnp.dot(x, self.A.T) - mu_yx
            exponent = -0.5 * jnp.einsum("ij,ij->i", jnp.dot(diff, sigma_yx_inv), diff)
            p += w * 1/((2*jnp.pi)**(self.m/2)) * 1/jnp.sqrt(sigma_yx_det) * jnp.exp(exponent)

        return p
    

    def p_y(self, y):
        p = 0
        for w, mu, sigma_y_inv, sigma_y_det in zip(self.weight, self.mu_yx, self.sigma_y_inv, self.sigma_y_det):
            diff = y + mu
            exponent = -0.5 * jnp.dot(jnp.dot(diff, sigma_y_inv), diff)
            p += (w * 1/((2*jnp.pi)**(self.m/2)) * 1/jnp.sqrt(sigma_y_det) * jnp.exp(exponent))
        
        return p
    
    def p_xy(self, x, y):  
        p = self.p_yx(y, x) * self.p_x(x) / self.p_y(y)
        return p
                      
        
    def gradV(self, x):
        num = jnp.zeros_like(x)
        den = jnp.zeros_like(x)
        for w, mu_xy, sigma_inv, sigma_det in zip(self.weight, self.mu_xy, self.sigma_inv, self.sigma_det):
            p = 1/((2*np.pi)**(self.n/2)) * 1/np.sqrt(sigma_det) * jnp.exp(-0.5 * jnp.einsum('ij, ij->i', jnp.dot(x-mu_xy, sigma_inv), x-mu_xy))
            dp = -jnp.dot(x, sigma_inv.T) + jnp.dot(sigma_inv, mu_xy)
            num += w * p[:, None] * dp
            den += w * p[:, None]

        den = jnp.clip(den, 1e-16, None)

        return - num / den

    
    


    
    
    

