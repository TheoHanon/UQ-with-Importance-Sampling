import numpy as np


nStep = 1000
N = 1000
dt = 1e-3
D = 1

sigma = .1
mu = 1


def langevin_diffusion(x0, D):
    x = np.zeros((nStep, N))
    x[0] = x0
    gradV = lambda x : -(x - mu)/sigma
    for i in range(1, nStep):
        x[i] = x[i-1] + gradV(x[i-1]) * dt + np.sqrt(2*D*dt) * np.random.normal(0, 1, N)
    return x
    


def compute_hpred(xt):
    alpha_t = np.zeros(nStep)
    beta_t = np.zeros(nStep)

    alpha_t = np.mean(xt, axis = 1)
    beta_t = np.mean((xt - alpha_t[:, None])**2, axis = 1)

    return alpha_t, beta_t


def compute_qt(xt, beta_t):

    qt = np.zeros((nStep, N))
    qt[0] = 1/np.sqrt(2*np.pi * sigma) * np.exp(-0.5 * (xt[0] - mu)**2/sigma)
    time = np.arange(nStep) * dt
    for i in range(nStep):
        qt[i] = qt[0] * np.exp(- time[i] * (-1/sigma + D / beta_t[i]))

    return qt



xt = langevin_diffusion(np.random.normal(mu, sigma, N), D)
alpha_t, beta_t = compute_hpred(xt)
qt = compute_qt(xt, beta_t)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


fig, ax = plt.subplots()

def update(frame):
    idx = np.argsort(xt[frame])

    ax.clear()
    ax.hist(xt[frame], bins=50, density=True)
    ax.plot(xt[frame, idx], qt[frame, idx], color='red')
    ax.set_title(f"Frame {frame}")
    ax.set_xlabel("x")
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 1)   
    ax.set_ylabel("Probability density")


ani = FuncAnimation(fig, update, frames=nStep, interval=50)
plt.show()