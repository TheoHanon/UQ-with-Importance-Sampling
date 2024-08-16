import numpy as np
import matplotlib.pyplot as plt
from model.gaussian_mixture import GaussianMixture
from model.gaussian import Gaussian
from PARTI.old.param import Simulation_Parameter_Gaussian_Mixture as GaussianMixture_old

m = 1
n = 1
N = 100

y = np.ones(m)
A = np.ones((m, n))
weights = np.array([.5, .25, .5])


sigma_X = 2 * np.eye(n)
mu_YX = np.array([np.random.choice([-3], m), np.zeros(m), np.random.choice([3], m)])
sigma_YX = np.array([np.diag(np.random.choice([.5, 1], n)), np.diag(np.random.choice([1, 2], n)), np.diag(np.random.choice([.1, .5], n))])

GMM = GaussianMixture(y, A, weights, mu_YX, sigma_YX, sigma_X)
G = Gaussian(y, A, mu_YX[1], sigma_YX[1], sigma_X)

GMM_old = GaussianMixture_old(y = y, nStep = 1, N = N, dt = 1e-3, n = n, m = m, A = A, mu_prop = np.zeros(n), sigma_prop = np.eye(n, n), weight = weights, sigma_x = sigma_X, mu_yx = mu_YX, sigma_yx = sigma_YX, uniform = True, a_unif = 2, scale = 1)

x = np.linspace(-5, 5, N).reshape(-1, 1)

P_XY = np.exp(GMM.logP_XY(x, numpy = True))
P_XY_G = np.exp(G.logP_XY(x, numpy = True))
# grad_logP_XY = GMM.grad_logP_XY(x, numpy = True)



plt.plot(x, P_XY, label = "GMM")
plt.plot(x, P_XY_G, label = "G")
# plt.plot(x, P_XY_old, label = "Old")
# plt.plot(x, grad_logP_XY)
plt.legend()
plt.show()


















