import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Flow import *
from Model import Model
from NN import NN
from estimator import MonteCarlo, ImportanceSampling, AdaptiveImportanceSampling

def g(x):    
    mask = (x >= -1) & (x < 0)
    out = 1/2 * (np.sin(2*np.pi*x)**3 - 1) * mask
    out[~mask] = 1/2 * (np.sin(2*np.pi*x[~mask])**3 + 1)
    return out

gw = NN(input = 1, output = 1, hidden = 1, hidden_units=20, activation = tf.sin)
d = gw.count_params()
M = 20

X_train = np.random.uniform(-1, 1, (32, 1))
Y_train = g(X_train)
 
prior = tfp.distributions.Normal(-10, 10)
likelihood = tfp.distributions.Normal(0, 1e-1)

model = Model(X_train = X_train, Y_train = Y_train, gw = gw, prior = prior, likelihood = likelihood)
f = GradientFlow(M = M, epochs = 500, lr = 1e-2, model = model, q0 = tfp.distributions.Uniform(-2, 2))
w, logq = f.flow_and_distribution()
# w = f.flow()

est = ImportanceSampling(model = model, samples_weight = w, logq = logq)
# est = AdaptiveImportanceSampling(model = model, samples_weight = w, logq = logq)
# est = MonteCarlo(model = model, samples_weight = w)
mean, var = est.estimate(x_pred = np.linspace(-1, 1, 100).reshape(-1, 1))


fig, ax = plt.subplots()

def update(frame):
    ax.clear()
    ax.plot(np.linspace(-1, 1, 100), g(np.linspace(-1, 1, 100)), label = "True", color = "tab:red")
    ax.plot(np.linspace(-1, 1, 100), mean[frame], label = "Predicted", color = "tab:blue")
    ax.fill_between(np.linspace(-1, 1, 100), mean[frame] - np.sqrt(var[frame]), mean[frame] + np.sqrt(var[frame]), alpha = 0.5, color = "tab:blue")
    ax.legend()

    return ax,


ani = animation.FuncAnimation(fig, update, frames = 500, interval = 10)
plt.show()











