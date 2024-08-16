# Author: Theo Hanon    
# Created : 2024-07-05

import numpy as np
from scipy.integrate import nquad
import matplotlib.animation as animation
from matplotlib import pyplot as plt
import tqdm

import jax
import jax.numpy as jnp
from numba import njit, prange

class Simulation_Parameter:
    """
    A class to handle the initialization and operations of simulation parameters.
    """
    
    def __init__(self, y: float, nStep: int, N: int, dt: float, n: int, m: int, A: np.ndarray, mu_prop: np.ndarray, sigma_prop: np.ndarray):
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

        self.x0 = np.random.multivariate_normal(mu_prop, sigma_prop, N)

    def div(self, f, x):
        """
        Compute the divergence of a function f at point x.
        """
        jac_f = jax.jacfwd(f)
        jac_matrix = jac_f(x)
        if self.n == 1:
            return np.diag(jac_matrix.squeeze())
        else:
            div = jnp.trace(jnp.diagonal(jac_matrix, axis1=0, axis2=2).T, axis1=1, axis2=2)
            return div

    def gradient_flow(self):
        """
        Compute the gradient flow of a function V at point x0.
        """
        N = self.N
        n = self.n
        nStep = self.nStep
        dt = self.dt

        x = np.empty((nStep, N, n))
        x[0] = self.x0

        for i in range(nStep - 1):
            x[i + 1] = x[i] - self.gradV(x[i]) * dt

        return x

    def langevin_sampling(self, D=10):
        """
        Compute the Langevin diffusion of a function V at point x0.
        """
        n = self.n
        N = self.N
        nStep = self.nStep
        dt = self.dt

        x = np.empty((nStep, N, n))
        x[0] = self.x0

        for i in range(nStep - 1):
            x[i + 1, ...] = x[i, ...] - self.gradV(x[i, ...]) * dt + np.random.normal(0, 1, (N, n)) * np.sqrt(2 * D * dt)

        return x

    def compute_qt(self, xt):
        """
        Compute the importance weights qt for a given proposal distribution.
        """
        n = self.n
        N = self.N
        nStep = self.nStep
        dt = self.dt

        f = lambda x: -self.gradV(x)
        qt = np.zeros((nStep, N))

        diff = xt[0, :] - self.mu_prop
        qt[0, :] = 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(self.sigma_prop_det)) * np.exp(
            -0.5 * np.sum(diff @ self.sigma_prop_inv * diff, axis=1))

        I = np.zeros(N)

        for k in tqdm.tqdm(range(1, nStep)):
            xold = xt[k - 1, ...]
            xnew = xt[k, ...]

            diva = self.div(f, xold)
            divb = self.div(f, xnew)

            I += dt * (diva + divb) / 2
            qt[k, :] = qt[0, :] * np.exp(-I)

        return qt

    def compute_moment(self, alpha: int):
        """
        Compute the exact moment of the posterior distribution.
        """
        m = np.zeros(self.n)

        for i in range(self.n):
            def f(*x):
                x_vec = np.array(x)
                return np.prod(x_vec[i] ** alpha) * self.p_xy(np.array([x_vec]), self.y)

            m[i] = nquad(f, [(-np.inf, np.inf)] * self.n)[0]

        return m

    def compute_moment_IP(self, alpha: int, xt, qt):
        """
        Compute the moment of the posterior distribution using importance sampling with normalization.
        """
        def f(x):
            return x ** alpha

        p_post = self.p_xy(xt, self.y)
        w = (p_post / qt) / np.sum(p_post / qt)

        return np.sum(f(xt) * w[:, np.newaxis], axis=0)

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

# Helper functions to be compiled with Numba

@njit
def p_x_gaussian(x, sigma_x_inv, sigma_x_det, n):
    exponent = -0.5 * np.sum(x @ sigma_x_inv * x, axis=1)
    return 1 / ((2 * np.pi) ** (n / 2)) * 1 / np.sqrt(sigma_x_det) * np.exp(exponent)

@njit
def p_yx_gaussian(y, x, A, sigma_yx_inv, sigma_yx_det, m):
    diff = y - x @ A.T
    exponent = -0.5 * np.sum(diff @ sigma_yx_inv * diff, axis=1)
    return 1 / ((2 * np.pi) ** (m / 2)) * 1 / np.sqrt(sigma_yx_det) * np.exp(exponent)

@njit
def p_y_gaussian(y, sigma_y_inv, sigma_y_det, m):
    exponent = -0.5 * y @ sigma_y_inv @ y
    return 1 / ((2 * np.pi) ** (m / 2)) * 1 / np.sqrt(sigma_y_det) * np.exp(exponent)

@njit
def p_xy_gaussian(x, y, sigma_x_inv, sigma_x_det, A, sigma_yx_inv, sigma_yx_det, sigma_y_inv, sigma_y_det, m, n):
    px = p_x_gaussian(x, sigma_x_inv, sigma_x_det, n)
    pyx = p_yx_gaussian(y, x, A, sigma_yx_inv, sigma_yx_det, m)
    py = p_y_gaussian(y, sigma_y_inv, sigma_y_det, m)
    return pyx * px / py

@njit
def gradV_gaussian(x, sigma_inv, A, sigma_yx_inv, y):
    return x @ sigma_inv.T - A.T @ sigma_yx_inv @ y

@njit
def p_x_gaussian_mixture(x, sigma_x_inv, sigma_x_det, n):
    exponent = -0.5 * np.sum(x @ sigma_x_inv * x, axis=1)
    return 1 / ((2 * np.pi) ** (n / 2)) * 1 / np.sqrt(sigma_x_det) * np.exp(exponent)

@njit
def p_yx_gaussian_mixture(y, x, A, weight, mu_yx, sigma_yx_inv, sigma_yx_det, m):
    p = np.zeros(x.shape[0])
    for w, mu, inv, det in zip(weight, mu_yx, sigma_yx_inv, sigma_yx_det):
        diff = y - x @ A.T - mu
        exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)
        p += w * 1 / ((2 * np.pi) ** (m / 2)) * 1 / np.sqrt(det) * np.exp(exponent)
    return p

@njit
def p_y_gaussian_mixture(y, weight, mu_yx, sigma_y_inv, sigma_y_det, m):
    p = 0.0
    for w, mu, inv, det in zip(weight, mu_yx, sigma_y_inv, sigma_y_det):
        diff = y + mu
        exponent = -0.5 * diff @ inv @ diff
        p += w * 1 / ((2 * np.pi) ** (m / 2)) * 1 / np.sqrt(det) * np.exp(exponent)
    return p

@njit
def p_xy_gaussian_mixture(x, y, A, weight, mu_yx, sigma_x_inv, sigma_x_det, sigma_yx_inv, sigma_yx_det, sigma_y_inv, sigma_y_det, m, n):
    px = p_x_gaussian_mixture(x, sigma_x_inv, sigma_x_det, n)
    pyx = p_yx_gaussian_mixture(y, x, A, weight, mu_yx, sigma_yx_inv, sigma_yx_det, m)
    py = p_y_gaussian_mixture(y, weight, mu_yx, sigma_y_inv, sigma_y_det, m)
    return pyx * px / py

@njit
def gradV_gaussian_mixture(x, A, weight, mu_xy, sigma_inv, sigma_det):
    num = np.zeros_like(x)
    den = np.zeros_like(x)
    for w, mu, inv, det in zip(weight, mu_xy, sigma_inv, sigma_det):
        p = 1 / ((2 * np.pi) ** (x.shape[1] / 2)) * 1 / np.sqrt(det) * np.exp(-0.5 * np.sum((x - mu) @ inv * (x - mu), axis=1))
        dp = -x @ inv.T + inv @ mu
        num += w * p[:, None] * dp
        den += w * p[:, None]
    den = np.clip(den, 1e-16, None)
    return -num / den

class Simulation_Parameter_Gaussian(Simulation_Parameter):
    """
    A class to handle the initialization and operations of simulation parameters for a Gaussian model.
    """

    def __init__(self, y: float, nStep: int, N: int, dt: float, n: int, m: int, A: np.ndarray, mu_prop: np.ndarray, sigma_prop: np.ndarray, sigma_x: np.ndarray, sigma_yx: np.ndarray):
        super().__init__(y, nStep, N, dt, n, m, A, mu_prop, sigma_prop)
        self.sigma_x = sigma_x
        self.sigma_yx = sigma_yx
        self._init_gaussian()

        self.xt = self.gradient_flow()
        self.qt = self.compute_qt(self.xt)

    def _init_gaussian(self):
        """
        Initialize Gaussian matrices and their inverses and determinants.
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

        self.mu_xy = self.sigma @ (self.A).T @ self.sigma_yx_inv @ self.y

    def p_x(self, x):
        return p_x_gaussian(x, self.sigma_x_inv, self.sigma_x_det, self.n)

    def p_yx(self, y, x):
        return p_yx_gaussian(y, x, self.A, self.sigma_yx_inv, self.sigma_yx_det, self.m)

    def p_y(self, y):
        return p_y_gaussian(y, self.sigma_y_inv, self.sigma_y_det, self.m)

    def p_xy(self, x, y):
        return p_xy_gaussian(x, y, self.sigma_x_inv, self.sigma_x_det, self.A, self.sigma_yx_inv, self.sigma_yx_det, self.sigma_y_inv, self.sigma_y_det, self.m, self.n)

    def gradV(self, x):
        return gradV_gaussian(x, self.sigma_inv, self.A, self.sigma_yx_inv, self.y)

    def compute_qt(self, xt):
        """
        Compute the importance weights qt for a given proposal distribution.
        """
        n = self.n
        N = self.N
        nStep = self.nStep
        dt = self.dt

        qt = np.zeros((nStep, N))
        I = np.zeros(N)

        diff = xt[0, :] - self.mu_prop
        qt[0, :] = 1 / ((2 * np.pi) ** (n / 2) * np.sqrt(self.sigma_prop_det)) * np.exp(
            -0.5 * np.sum(diff @ self.sigma_prop_inv * diff, axis=1))

        for k in range(1, nStep):
            diva = -np.trace(self.sigma_inv)
            divb = -np.trace(self.sigma_inv)

            I += dt * (diva + divb) / 2
            qt[k, :] = qt[0, :] * np.exp(-I)

        return qt

class Simulation_Parameter_Gaussian_Mixture(Simulation_Parameter):
    """
    A class to handle the initialization and operations of simulation parameters for a Gaussian mixture model.
    """

    def __init__(self, y: float, nStep: int, N: int, dt: float, n: int, m: int, A: np.ndarray, mu_prop: np.ndarray, sigma_prop: np.ndarray, weight: np.ndarray, sigma_x: np.ndarray, mu_yx: np.ndarray, sigma_yx: np.ndarray):
        super().__init__(y, nStep, N, dt, n, m, A, mu_prop, sigma_prop)

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
        self.mu_xy = np.array([sigma @ (self.A).T @ sigma_yx_inv @ (self.y - mu_yx) for (sigma, sigma_yx_inv, mu_yx) in zip(self.sigma, self.sigma_yx_inv, self.mu_yx)])

    def p_x(self, x):
        return p_x_gaussian_mixture(x, self.sigma_x_inv, self.sigma_x_det, self.n)

    def p_yx(self, y, x):
        return p_yx_gaussian_mixture(y, x, self.A, self.weight, self.mu_yx, self.sigma_yx_inv, self.sigma_yx_det, self.m)

    def p_y(self, y):
        return p_y_gaussian_mixture(y, self.weight, self.mu_yx, self.sigma_y_inv, self.sigma_y_det, self.m)

    def p_xy(self, x, y):
        return p_xy_gaussian_mixture(x, y, self.A, self.weight, self.mu_yx, self.sigma_x_inv, self.sigma_x_det, self.sigma_yx_inv, self.sigma_yx_det, self.sigma_y_inv, self.sigma_y_det, self.m, self.n)

    def gradV(self, x):
        return gradV_gaussian_mixture(x, self.A, self.weight, self.mu_xy, self.sigma_inv, self.sigma_det)
