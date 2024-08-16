import numpy as np
import tqdm
import matplotlib.pyplot as plt
import scipy.linalg
from matplotlib.animation import FuncAnimation 
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


    
def F(x, t, mu, sigma, D):
    """
    Exact solution to the anisotropic diffusion equation with a Gaussian initial distribution N(mu, Sigma).
    Args:    
    x: (N,n)
    sigma: (n, n)
    t: (1,)
    mu: (n,)
    D : (n,)
    """

    n = x.shape[1]
    Sigma_t = sigma + np.diag(2*t*D)
    Sigma_t_inv = np.linalg.inv(Sigma_t)
    log_Sigma_t_det = np.linalg.slogdet(Sigma_t)[1]
    
    exponent = -1/2 * np.einsum("ij, ji -> i" , x - mu, np.dot(Sigma_t_inv, (x - mu).T))
    log_prefactor =  - 1/2 * log_Sigma_t_det -n/2 * np.log(2*np.pi)

    return np.exp(exponent + log_prefactor) #*(2*np.pi)**(n/2) * np.sqrt(np.prod(2*D*t))

def diffuse(x2, x1, Alpha, dt):
    """
    Approximate the solution to the anisotropic diffusion equation with a Monte carlo integration.
    Args: 
    D:(n,)
    x2: (N,n)clear

    x1: (N,n)
    dt: (1,)
    """

    n = x1.shape[1]
    N = x1.shape[0]
    diff = x2[:, None, :] - x1[None, ...] # (N, N, n)
    Dt = 2*Alpha*dt


    exponent = -1/2 * np.einsum("ijk, k, ijk -> ij", diff, 1/Dt, diff) # (N, N)
    
    max_exponent = np.max(exponent, axis = 1, keepdims = True) # (N, 1)
    w = np.exp(np.clip(-700, 700, exponent - max_exponent)) # (N, N)

    return np.mean(w * np.exp(max_exponent), axis = 1) * (2*np.pi)**(-n/2) * np.prod(2*Alpha*dt)**(-1/2)


def compute_q1q2(xi, xj, z, D, dt):
    """
    Args:
    xi = (n, )
    xj = (n, )
    z = (N, n)
    D = (n,)
    dt = (1,)
    """

    diff1 = xi - z # (N, n) 
    diff2 = xj - z # (N, n)


    exponent1 = -1/2 * np.einsum("ij, j, ij -> i", diff1, 1/(2*D*dt), diff1) # (N,)
    exponent2 = -1/2 * np.einsum("ij, j, ij -> i", diff2, 1/(2*D*dt), diff2) # (N,)

    # print(exponent1, exponent2)

    max_exponent1 = np.max(exponent1)
    max_exponent2 = np.max(exponent2)


    sum_exp1 = np.sum(np.exp(np.clip(exponent1 - max_exponent1, -700, 700)))
    sum_exp2 = np.sum(np.exp(np.clip(exponent2 - max_exponent2, -700, 700)))


    ratio = sum_exp1/sum_exp2 * np.exp(max_exponent1 - max_exponent2)

    return ratio


def compute_q1q2_fast(z, x, D, dt):
    """
    Args:
    x2: (N,n)
    x1: (N,n)
    D: (n,)
    dt: (1,)
    """

    diff = x[:, None, :] - z[None, ...] # (N, N, n)

    exponent = -1/2 * np.einsum("ijk, k, ijk -> ij", diff, 1/(2*D*dt), diff) # (N, N)
    max_exponent = np.max(exponent, axis = 1, keepdims = True) # (N, 1)
    sum_exp = np.sum(np.exp(exponent - max_exponent), axis = 1) # (N,)

    ratios = (sum_exp[:, None] / sum_exp[None, :]) * np.exp(max_exponent - max_exponent.T)
    return ratios






N = 10000
dt = 2
dim = 1

np.random.seed(0)
sigma = 1*np.eye(dim)
mu = np.zeros(dim)

D = 10*np.ones(dim)

x1 = np.random.multivariate_normal(mu, sigma, N)
x2 = x1 + np.sqrt(2*D*dt) * np.random.normal(np.zeros(dim), np.ones(dim), (N, dim))


p = F(x2, dt, mu, sigma, D)
# q = diffuse(x2, x1, D, dt)

ratios_p = p[:, None] / p[None, :]
ratios_q = compute_q1q2_fast(x1, x2, D, dt)

print(np.allclose(ratios_p, ratios_q, atol = 0.1))
# print(ratios_p, ratios_q)
# print(np.allclose(ratios_q, ratios_p, atol = 0.1))
# print(np.allclose(q, p, atol = 0.1))

# M = 50
# dt = 1
# dim  = [1, 5, 10, 15, 20, 30, 40, 50, 100, 1000]
# var = [1, 1e-2, 1e2]

# alphas = np.zeros((len(dim), len(var)))

# for i, d in enumerate(dim):
#     for j, v in enumerate(var):
#         sigma = v*np.eye(d)
#         mu = np.zeros(d)
#         D = 1e0*np.ones(d)

#         x1 = np.random.multivariate_normal(mu, sigma, M)
#         x2 = x1 + np.sqrt(2*D*dt) * np.random.normal(np.zeros(d), np.ones(d), (M, d))

#         ratios = compute_q1q2_fast(x1, x2, D, dt).reshape(-1)
#         mask = (ratios < 1e3) & (ratios > 1e-3) & (ratios != 1)
#         alphas[i, j] = np.sum(mask) / (M*M)


# dims = [str(d) for d in dim]

# fig, axs = plt.subplots(1, 1, figsize=(9, 6))

# bar_width = 0.2  # Width of the bars
# x = np.arange(len(dims))  # the label locations
# offsets = np.abest range(len(var)) * bar_width  # Offsets for each bar group

# for i, v in enumerate(var):
#     axs.bar(x + offsets[i] - bar_width / 2, alphas[:, i], width=bar_width, label=r"$\mathbf{\Sigma}$ = " + str(v) + r"$\mathbf{Id}$")

# axs.set_xticks(x)
# axs.set_xticklabels(dims)

# axs.set_xlabel('Dimension')
# axs.set_ylabel(r'$\alpha$', rotation=0, labelpad=20, fontsize=15)
# axs.set_title(r'$\gamma = 1e^{3},~\xi = 1e^{-5},~D = $'+str(D[0])+r'$\mathbf{Id}$', fontsize=15)
# axs.legend(loc='lower left')
# plt.show()




if False:

    dt = .1
    eps = 1e-16
    N = [50, 100, 200, 500]#, 1000, 2000]
    dim =[1, 5, 10, 15, 20, 30, 40, 50, 1000]



    MSE_g = np.zeros((len(N), len(dim)))

    for j, d in enumerate(dim):
        for k, n in enumerate(N) :
            mu = np.zeros(d)
            sigma = 1*np.eye(d)
            D = 1*np.ones(d)


            x1 = np.random.multivariate_normal(mu, sigma, n)
            x2 = x1 + np.sqrt(2*D) * np.random.normal(np.zeros(d), dt*np.ones(d), (n, d))
            
            q_g = diffuse(x2, x1, Alpha= D, dt = dt)
            p_q = F(x2, dt, mu, sigma, D)
            
            MSE_g[k, j] = np.mean((q_g - p_q)**2)




    fig, axs = plt.subplots(1, 1, figsize = (9, 6))
    fig.suptitle("MSE of the MC integral", fontsize = 16, fontweight = "bold")
    axs.semilogy(np.tile(N, (len(dim), 1)).T, MSE_g, marker = "s", label = dim)
    axs.grid(which="both")
    axs.set_xlabel("N")
    axs.set_ylabel(r"$\mathbf{MSE}$", rotation = 0, labelpad = 20)
    axs.legend(title = "Dimension", loc = "upper right", ncol = 2)

    plt.show()




