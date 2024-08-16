import numpy as np
from model.gaussian import Gaussian
from model.gaussian_mixture import GaussianMixture
from flow.gradient_flow import GradientFlow
from flow.mix_flow import MixFlow
from flow.split_step_flow import SplitStepFlow
from flow.hamilton_flow import HamiltonFlow, SplitHamiltonFlow
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import torch



def plot1():
    N = 200
    nStep = 1500
    dt = 1e-1
    n  = 20
    m  = 20


    sigma_x = 3 * np.eye(n)
    y = np.ones(m)
    A = np.eye(m, n)

    mu_yx = np.array([np.random.choice([-2.5], m), np.zeros(m), np.random.choice([2.5], m)])
    sigma_yx = np.array([np.diag(np.random.choice([1], n)), np.diag(np.random.choice([1, 2], n)), np.diag(np.random.choice([1, 5], n))])
    weight = np.array([.3, .4, .3])


    # model = GaussianMixture(y = y, A = A, weights = weight, mu_YX = mu_yx, sigma_YX = sigma_yx, sigma_X = sigma_x)
    model = Gaussian(y = y, A = A, mu_YX = mu_yx[1], sigma_YX = sigma_yx[1], sigma_X = sigma_x)

    m1 = model.compute_moment(1)
    m2 = np.diag(model.compute_moment(2))
    # print(model.model_XY.variance + model.model_XY.mean**2, m2)


    flow_sHMC = SplitHamiltonFlow(N = N, nStep = nStep, dt = 1e-1, model = model, pre_compute=True, epsilon = 1e-1, k = 1,  q0 = torch.distributions.Uniform(-3, 3), M_inv = 1e-2*torch.eye(n))
    xt_sHMC = flow_sHMC.x.detach().numpy()
    m1_sHMC, m2_sHMC = flow_sHMC.adaptiveImportanceSampling([1, 2])
    m1_HMC, m2_HMC = flow_sHMC.monteCarlo([1, 2])
    qt_sHMC = np.exp(flow_sHMC.getlogQ())


    flow_ssf = SplitStepFlow(N = N, nStep = nStep, dt = dt, model = model, pre_compute=True, k = 5, sigma0= 1e-2,  q0 = torch.distributions.Uniform(-3, 3), burn_in=100)
    xt_ssf = flow_ssf.x.detach().numpy()
    qt_ssf = np.exp(flow_ssf.getlogQ())
    m1_sSSF, m2_sSSF = flow_ssf.adaptiveImportanceSampling([1, 2])
    m1_SSF, m2_SSF = flow_ssf.monteCarlo([1, 2])

    flow_mix = MixFlow(N = N, nStep = nStep, dt = dt, model = model, pre_compute=True, k = 1)
    xt_mix = flow_mix.getFlow()
    qt_mix = np.exp(flow_mix.getlogQ())
    m1_mix, m2_mix = flow_mix.adaptiveImportanceSampling([1, 2])


    # print(np.min(m2_HMC), np.max(m2_HMC))

    MSE_m1_sSSF = np.mean((m1_sSSF - m1)**2, axis =1)
    MSE_m2_sSSF = np.mean((m2_sSSF - m2)**2, axis =1)

    MSE_m1_SSF = np.mean((m1_SSF - m1)**2, axis =1)
    MSE_m2_SSF = np.mean((m2_SSF - m2)**2, axis =1)

    MSE_m1_HMC = np.mean((m1_HMC - m1)**2, axis =1)
    MSE_m2_HMC = np.mean((m2_HMC - m2)**2, axis =1)

    MSE_m1_sHMC = np.mean((m1_sHMC - m1)**2, axis =1)
    MSE_m2_sHMC = np.mean((m2_sHMC - m2)**2, axis =1)

    MSE_m1_mix = np.mean((m1_mix - m1)**2, axis =1)
    MSE_m2_mix = np.mean((m2_mix - m2)**2, axis =1)


    fig, ax = plt.subplots(1, 1, figsize = (10, 5))
    fig.suptitle(f"N = {N}, d = {n}")
    ax.loglog(np.arange(1, nStep+1), MSE_m1_sSSF, label = "m1 - Split Step (IS)", color = "darkseagreen")
    ax.loglog(np.arange(1, nStep+1), MSE_m1_mix, label = "m1 - Mix (IS)", color = "cornflowerblue")
    ax.loglog(np.arange(1, nStep+1), MSE_m1_sHMC, label = "m1 - Hamiltonian (IS)", color = "darkturquoise")

    ax.loglog(np.arange(1, nStep+1), MSE_m1_SSF, label = "m1 - Split Step (no IS)", color = "orangered")
    ax.loglog(np.arange(1, nStep+1), MSE_m1_HMC, label = "m1 - Hamiltonian (no IS)", color = "slateblue")

    ax.loglog(np.arange(1, nStep+1), MSE_m2_sSSF, label = "m2 - Split Step (IS)", color = "darkseagreen", linestyle = "--")
    ax.loglog(np.arange(1, nStep+1), MSE_m2_mix, label = "m2 - Mix (IS)", color = "cornflowerblue", linestyle = "--")
    ax.loglog(np.arange(1, nStep+1), MSE_m2_sHMC, label = "m2 - Hamiltonian (IS)", color = "darkturquoise", linestyle = "--")

    ax.loglog(np.arange(1, nStep+1), MSE_m2_SSF, label = "m2 - Split Step (no IS)", color = "orangered", linestyle = "--")
    ax.loglog(np.arange(1, nStep+1), MSE_m2_HMC, label = "m2 - Hamiltonian (no IS)", color = "slateblue", linestyle = "--")

    ax.legend(ncol = 2)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\frac{1}{d} MSE[\hat{m}]$")
    ax.grid(which='both')
    plt.show()


    x1, x2 = np.meshgrid(np.linspace(np.min(xt_ssf[:, :, 0]), np.max(xt_ssf[:, :, 0]), 100), np.linspace(np.min(xt_ssf[:, :, 1]), np.max(xt_ssf[:, :, 1]), 100))
    x = np.zeros((x1.size, n))
    x[:, 0] = x1.flatten()
    x[:, 1] = x2.flatten()
    z = model.logP_XY(x).reshape(100, 100)


    fig, ax = plt.subplots(1, 1, figsize = (8, 8))

    def update(frame):

        ax.clear()
        ax.contour(x1, x2, z, levels = 50, colors = "black", linewidths = .5)
        ax.contourf(x1, x2, z, levels = 50, cmap = "GnBu")

        ax.plot(xt_mix[frame, :, 0], xt_mix[frame, :, 1], '.', markersize = 10, label = "Mix", alpha = .8, color = 'darkorange')
        ax.plot(xt_ssf[frame, :, 0], xt_ssf[frame, :, 1], '.', markersize = 10, label = "Split Step", alpha = .8, color = 'mediumspringgreen')
        ax.plot(xt_sHMC[frame, :, 0], xt_sHMC[frame, :, 1], '.', markersize = 10, label = "Hamiltonian", alpha = .8, color = 'k')
        
        ax.set_xlim(np.min(xt_ssf[:, :, 0]), np.max(xt_ssf[:, :, 0]))
        ax.set_ylim(np.min(xt_ssf[:, :, 1]), np.max(xt_ssf[:, :, 1]))

        ax.legend(loc = "upper right")
        ax.set

        return ax,


    ani = animation.FuncAnimation(fig, update, frames = nStep)
    plt.show()


def plot2():


    N = 200
    nStep = 500
    dt = 1e-2
    n  = 1
    m  = 1


    sigma_x = 3 * np.eye(n)
    y = np.ones(m)
    A = np.eye(m, n)

    mu_yx = np.array([np.random.choice([-2.5], m), np.zeros(m), np.random.choice([2.5], m)])
    sigma_yx = np.array([np.diag(np.random.choice([.5], n)), np.diag(np.random.choice([1], n)), np.diag(np.random.choice([.5], n))])
    weight = np.array([.3, .4, .3])


    model = GaussianMixture(y = y, A = A, weights = weight, mu_YX = mu_yx, sigma_YX = sigma_yx, sigma_X = sigma_x)
    # model = Gaussian(y = y, A = A, mu_YX = mu_yx[1], sigma_YX = sigma_yx[1], sigma_X = sigma_x)

    m1 = model.compute_moment(1)
    m2 = np.diag(model.compute_moment(2))


    flow_sHMC = SplitHamiltonFlow(N = N, nStep = nStep, dt = 1e0, model = model, pre_compute=True, epsilon = 1e-2, k = .5,  q0 = torch.distributions.Uniform(-3, 3))
    xt_sHMC = flow_sHMC.getFlow()

    # print(np.min(xt_sHMC), np.max(xt_sHMC))
    m1_sHMC, m2_sHMC = flow_sHMC.adaptiveImportanceSampling([1, 2])
    m1_HMC, m2_HMC = flow_sHMC.monteCarlo([1, 2])
    qt_sHMC = np.exp(flow_sHMC.getlogQ())


    flow_ssf = SplitStepFlow(N = N, nStep = nStep, dt = dt, model = model, pre_compute=True, k = 2, sigma0= 1e-2,  q0 = torch.distributions.Uniform(-3, 3), burn_in=100)
    xt_ssf = flow_ssf.getFlow()
    qt_ssf = np.exp(flow_ssf.getlogQ())
    m1_sSSF, m2_sSSF = flow_ssf.adaptiveImportanceSampling([1, 2])
    m1_SSF, m2_SSF = flow_ssf.monteCarlo([1, 2])


    flow_mix = MixFlow(N = N, nStep = nStep, dt = dt, model = model, pre_compute=True, k = 1, q0 = torch.distributions.Uniform(-3, 3))
    xt_mix = flow_mix.getFlow()
    qt_mix = np.exp(flow_mix.getlogQ())
    m1_mix, m2_mix = flow_mix.importanceSampling([1, 2])


    MSE_m1_sSSF = np.mean((m1_sSSF - m1)**2, axis =1)
    MSE_m2_sSSF = np.mean((m2_sSSF - m2)**2, axis =1)

    MSE_m1_sHMC = np.mean((m1_sHMC - m1)**2, axis =1)
    MSE_m2_sHMC = np.mean((m2_sHMC - m2)**2, axis =1)

    MSE_m1_mix = np.mean((m1_mix - m1)**2, axis =1)
    MSE_m2_mix = np.mean((m2_mix - m2)**2, axis =1)


    xp = np.linspace(-6, 6, 100).reshape(-1, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))

    def update(frame):

        ax1.clear()
        ax2.clear()

        ax1.plot(xt_mix[frame, :, 0], qt_mix[frame], '.', markersize = 10, label = "Mix")
        ax1.plot(xt_ssf[frame, :, 0], qt_ssf[frame], '.', markersize = 10, label = "Split Step", alpha = .5)
        ax1.plot(xt_sHMC[frame, :, 0], qt_sHMC[frame], '.', markersize = 10, label = "Hamiltonian", alpha = .5)
        # ax1.plot(xt_gf[frame, :, 0], qt_gf[frame], '.', markersize = 10, label = "Gradient Flow")
        ax1.plot(xp.squeeze(), np.exp(model.logP_XY(xp)))
        ax1.set_xlim(-6, 6)
        ax1.set_ylim(0, .7)
        ax1.legend()

        ax2.semilogy(np.arange(1, frame+1), MSE_m1_sSSF[:frame], label = "m1 - Split Step")
        ax2.semilogy(np.arange(1, frame+1), MSE_m1_mix[:frame], label = "m1 - Mix")
        ax2.semilogy(np.arange(1, frame+1), MSE_m1_sSSF[:frame], label = "m1 - Hamiltonian")
        
        ax2.semilogy(np.arange(1, frame+1), MSE_m2_sSSF[:frame], label = "m2 - Split Step")
        ax2.semilogy(np.arange(1, frame+1), MSE_m2_mix[:frame], label = "m2 - Mix")
        ax2.semilogy(np.arange(1, frame+1), MSE_m2_sHMC[:frame], label = "m2 - Hamiltonian")
        ax2.legend()
        return ax1, ax2, 


    ani = animation.FuncAnimation(fig, update, frames = nStep)
    plt.show()



def plot3():

    N = 200
    nStep = 500
    dt = 1e-2
    n  = 2
    m  = 2


    sigma_x = 5 * np.eye(n)
    y = np.zeros(m)
    A = np.eye(m, n)

    mu_yx = np.array([np.array([-4, 0]), np.zeros(m), np.array([0, 4])])
    sigma_yx = np.array([np.array([[.01, 0],[0, 1]]), np.array([[.05, 0],[0, .05]]), np.array([[.05, 0],[0, 1]])])
    weight = np.array([.33, .33, .33])



    model = GaussianMixture(y = y, A = A, weights = weight, mu_YX = mu_yx, sigma_YX = sigma_yx, sigma_X = sigma_x)
    # model = Gaussian(y = y, A = A, mu_YX = mu_yx[1], sigma_YX = sigma_yx[1], sigma_X = sigma_x)

    m1 = model.compute_moment(1)
    m2 = np.diag(model.compute_moment(2))

    # glow_gf = GradientFlow(N = N, nStep = nStep, dt = dt, model = model, pre_compute=True)


    flow_sHMC = SplitHamiltonFlow(N = N, nStep = nStep, dt = 1e-1, model = model, pre_compute=True, epsilon = 1e-2, k = 10,  q0 = torch.distributions.Uniform(-5, 5))
    xt_sHMC = flow_sHMC.x.detach().numpy()


    flow_ssf = SplitStepFlow(N = N, nStep = nStep, dt = dt, model = model, pre_compute=True, k = 5, sigma0= 1e-2,  q0 = torch.distributions.Uniform(-5, 5), burn_in=100)
    xt_ssf = flow_ssf.x.detach().numpy()

    flow_mix = MixFlow(N = N, nStep = nStep, dt = dt, model = model, pre_compute=True, k = 1, q0 = torch.distributions.Uniform(-5, 5))
    xt_mix = flow_mix.getFlow()


    x1, x2 = np.meshgrid(np.linspace(np.min(xt_ssf[:, :, 0]), np.max(xt_ssf[:, :, 0]), 100), np.linspace(np.min(xt_ssf[:, :, 1]), np.max(xt_ssf[:, :, 1]), 100))
    x = np.zeros((x1.size, n))
    x[:, 0] = x1.flatten()
    x[:, 1] = x2.flatten()
    z = model.logP_XY(x).reshape(100, 100)


    fig, ax = plt.subplots(1, 1, figsize = (8, 8))

    def update(frame):

        ax.clear()
        ax.contour(x1, x2, z, levels = 50, colors = "black", linewidths = .5)
        ax.contourf(x1, x2, z, levels = 50, cmap = "GnBu")

        ax.plot(xt_mix[frame, :, 0], xt_mix[frame, :, 1], '.', markersize = 10, label = "Mix", alpha = .8, color = 'darkorange')
        ax.plot(xt_ssf[frame, :, 0], xt_ssf[frame, :, 1], '.', markersize = 10, label = "Split Step", alpha = .8, color = 'mediumspringgreen')
        ax.plot(xt_sHMC[frame, :, 0], xt_sHMC[frame, :, 1], '.', markersize = 10, label = "Hamiltonian", alpha = .8, color = 'k')
        
        ax.set_xlim(np.min(xt_ssf[:, :, 0]), np.max(xt_ssf[:, :, 0]))
        ax.set_ylim(np.min(xt_ssf[:, :, 1]), np.max(xt_ssf[:, :, 1]))

        ax.legend(loc = "upper right")
        ax.set

        return ax,


    ani = animation.FuncAnimation(fig, update, frames = nStep)
    plt.show()


# plot2()
plot1()
# plot3()








            

    
    

    
            




    



    


    













  

    









