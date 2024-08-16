from model import Model
import numpy as np
from flow import *
from importance_sampling import ImportanceSampling
from nn import NN
import matplotlib.pyplot as plt
import matplotlib.animation as animation


n = 1
m = 1
M = 32
N = 50
nStep = 1000

def g(x):    
    mask = (x >= -1) & (x < 0)
    out = 1/2 * (np.sin(2*np.pi*x)**3 - 1) * mask
    out[~mask] = 1/2 * (np.sin(2*np.pi*x[~mask])**3 + 1)
    return out
    
sigma_YX = 0.1
n_YX = np.random.normal(0, np.sqrt(sigma_YX), (M, n), )


x1 = torch.linspace(-.8, -0.2, M//2)
x2 = torch.linspace(0.2,.8, M//2)
X_train = torch.cat((x1, x2)).reshape(-1, 1).double()
# X_train = torch.linspace(-1, 1, M).reshape(-1, 1).double()
Y_train = g(X_train) + n_YX

x_test = torch.linspace(-1, 1, 50, dtype = torch.float64).reshape(-1, 1)

bounds = (-50, 50)
gw = NN(n, m, 20, 1)

print("d = ", gw.get_params_count())

model = Model(Y_train, X_train, gw = gw, prior_bounds = bounds, scale_likelihood=1e0)
flow3 = MixFlow(N = N, nStep = nStep, dt = 1e-1, model = model, pre_compute=True, q0 = torch.distributions.Uniform(-2, 2), k = 5*1e-3)
# flow3 = HamiltonFlow(N = N, nStep = nStep, dt = 1e-1, model = model, pre_compute=True, q0 = torch.distributions.Uniform(-2, 2), k = 1e-3)


def g_opt(x, W):
    x = torch.tensor(x).reshape(-1, 1).float()
    W = W.detach().clone().float()
    return model.gw(x, W).squeeze().detach().numpy()



def plot1():

    flow1 = GradientFlow(N = N, nStep = nStep, dt = 1e-2, model = model, pre_compute=True, q0 = torch.distributions.Uniform(-2, 2))
    flow2 = MixFlow(N = N, nStep = nStep, dt = 1e-1, model = model, pre_compute=True, q0 = torch.distributions.Uniform(-2, 2), k = 5*1e-3)
    # flow3 = SplitStepFlow(N = N, nStep = nStep, dt = 1e-1, model = model, pre_compute=True, q0 = torch.distributions.Uniform(-2, 2), k = 1e-2, burn_in = nStep, sigma0=1e-4)
    flow3 = HamiltonFlow(N = N, nStep = nStep, dt = 1e-1, model = model, pre_compute=True, q0 = torch.distributions.Uniform(-2, 2), k = 1e-3)



    x = np.linspace(-1, 1, 50)

    IS_1 = ImportanceSampling(x_test, flow1, model)
    IS_2 = ImportanceSampling(x_test, flow2, model)
    IS_3 = ImportanceSampling(x_test, flow3, model)


    m1_IS1, m2_IS1 = IS_1.adaptiveImportanceSampling([1, 2])
    var_IS1 = m2_IS1 - m1_IS1**2
    m1_MC1, m2_MC1 =  IS_1.monteCarlo([1, 2])
    var_MC1 = m2_MC1 - m1_MC1**2

    m1_IS2, m2_IS2 = IS_2.adaptiveImportanceSampling([1, 2])
    var_IS2 = m2_IS2 - m1_IS2**2
    m1_MC2, m2_MC2 =  IS_2.monteCarlo([1, 2])
    var_MC2 = m2_MC2 - m1_MC2**2

    m1_IS3, m2_IS3 = IS_3.adaptiveImportanceSampling([1, 2])
    var_IS3 = m2_IS3 - m1_IS3**2
    m1_MC3, m2_MC3 =  IS_3.monteCarlo([1, 2])
    var_MC3 = m2_MC3 - m1_MC3**2



    fig, axs = plt.subplots(2, 2, figsize = (15, 9))
    ax1, ax2, ax3, ax4 = axs.flatten()

    def update(frame):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        

        ax1.plot(x_test.squeeze(), m1_IS1[frame, :, 0], color = 'k', label = r"$\mathbb{E}[y|x]$ (AIS)")
        ax1.plot(x_test.squeeze(), m1_MC1[frame, :, 0], color = 'orange', label = r"$\mathbb{E}[y|x]$ (MC)", alpha = 0.5)
        ax1.plot(X_train.squeeze(), Y_train.squeeze(), 'o', color = 'b', label = "Training data", alpha = 0.5)
        ax1.fill_between(x_test.squeeze(), m1_IS1[frame, :, 0] - var_IS1[frame, :, 0], m1_IS1[frame, :, 0] + var_IS1[frame, :, 0], alpha = 0.5, color = 'grey')
        ax1.fill_between(x_test.squeeze(), m1_MC1[frame, :, 0] - var_MC1[frame, :, 0], m1_MC1[frame, :, 0] + var_MC1[frame, :, 0], alpha = 0.5, color = 'orange')
        
        ax1.set_xlabel("x")
        ax1.set_ylabel(r"$u(x)$")
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-2, 3)
        ax1.plot(x_test.squeeze(), g(x_test.squeeze()), label = "True function", color = "r")
        ax1.legend()
        ax1.set_title("Gradient flow")


        ax2.plot(x_test.squeeze(), m1_IS2[frame, :, 0], color = 'k', label = r"$\mathbb{E}[y|x]$ (AIS)")
        ax2.plot(x_test.squeeze(), m1_MC2[frame, :, 0], color = 'orange', label = r"$\mathbb{E}[y|x]$ (MC)", alpha = 0.5)
        ax2.plot(X_train.squeeze(), Y_train.squeeze(), 'o', color = 'b', label = "Training data", alpha = 0.5)
        ax2.fill_between(x_test.squeeze(), m1_IS2[frame, :, 0] - var_IS2[frame, :, 0], m1_IS2[frame, :, 0] + var_IS2[frame, :, 0], alpha = 0.5, color = 'grey')
        ax2.fill_between(x_test.squeeze(), m1_MC2[frame, :, 0] - var_MC2[frame, :, 0], m1_MC2[frame, :, 0] + var_MC2[frame, :, 0], alpha = 0.5, color = 'orange')
        
        ax2.set_xlabel("x")
        ax2.set_ylabel(r"$u(x)$")
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-2, 3)
        ax2.plot(x_test.squeeze(), g(x_test.squeeze()), label = "True function", color = "r")
        ax2.legend()
        ax2.set_title("Mix flow")

        ax3.plot(x_test.squeeze(), m1_IS3[frame, :, 0], color = 'k', label = r"$\mathbb{E}[y|x]$ (AIS)")
        ax3.plot(x_test.squeeze(), m1_MC3[frame, :, 0], color = 'orange', label = r"$\mathbb{E}[y|x]$ (MC)", alpha = 0.5)
        ax3.plot(X_train.squeeze(), Y_train.squeeze(), 'o', color = 'b', label = "Training data", alpha = 0.5)
        ax3.fill_between(x_test.squeeze(), m1_IS3[frame, :, 0] - var_IS3[frame, :, 0], m1_IS3[frame, :, 0] + var_IS3[frame, :, 0], alpha = 0.5, color = 'grey')
        ax3.fill_between(x_test.squeeze(), m1_MC3[frame, :, 0] - var_MC3[frame, :, 0], m1_MC3[frame, :, 0] + var_MC3[frame, :, 0], alpha = 0.5, color = 'orange')
        
        ax3.set_xlabel("x")
        ax3.set_ylabel(r"$u(x)$")
        ax3.set_xlim(-1, 1)
        ax3.set_ylim(-2, 3)
        ax3.plot(x_test.squeeze(), g(x_test.squeeze()), label = "True function", color = "r")
        ax3.legend()
        ax3.set_title("Split-step flow")

        ax4.set_axis_off()

    ani = animation.FuncAnimation(fig, update, frames=nStep, repeat=False, interval=10)
    plt.show()
        

def plot2(flow):


    IS_1 = ImportanceSampling(x_test, flow, model)

    m1_IS1, m2_IS1 = IS_1.adaptiveImportanceSampling([1, 2])
    var_IS1 = m2_IS1 - m1_IS1**2
    m1_MC1, m2_MC1 =  IS_1.monteCarlo([1, 2])
    var_MC1 = m2_MC1 - m1_MC1**2



    fig, ax1 = plt.subplots(1, 1, figsize = (9, 9))


    def update(frame):
        ax1.clear()
        
        
        ax1.plot(x_test.squeeze(), m1_IS1[frame, :, 0], color = 'k', label = r"$\mathbb{E}[y|x]$ (AIS)")
        ax1.plot(x_test.squeeze(), m1_MC1[frame, :, 0], color = 'orange', label = r"$\mathbb{E}[y|x]$ (MC)", alpha = 0.5)
        ax1.plot(X_train.squeeze(), Y_train.squeeze(), 'o', color = 'b', label = "Training data", alpha = 0.5)
        ax1.fill_between(x_test.squeeze(), m1_IS1[frame, :, 0] - var_IS1[frame, :, 0], m1_IS1[frame, :, 0] + var_IS1[frame, :, 0], alpha = 0.5, color = 'grey')
        ax1.fill_between(x_test.squeeze(), m1_MC1[frame, :, 0] - var_MC1[frame, :, 0], m1_MC1[frame, :, 0] + var_MC1[frame, :, 0], alpha = 0.5, color = 'orange')
        
        ax1.set_xlabel("x")
        ax1.set_ylabel(r"$u(x)$")
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-2, 3)
        ax1.plot(x_test.squeeze(), g(x_test.squeeze()), label = "True function", color = "r")
        ax1.legend()
        ax1.set_title(f"{flow.__class__.__name__}")

    ani = animation.FuncAnimation(fig, update, frames=nStep, repeat=False, interval=10)
    plt.show()



plot2(flow3)


