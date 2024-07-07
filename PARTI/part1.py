# Author : Theo Hanon
# Created : 2024-07-01

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from param import Simulation_Parameter_Gaussian, Simulation_Parameter_Gaussian_Mixture


def draw_particul_path(mu_prop, var_prop, param):

    """
    Draw the path of a particular sample for gradient flow and langevin diffusion

    Parameters:
    -----------
    mu_prop : float
        Mean of the proposal distribution
    var_prop : float
        Variance of the proposal distribution
    param : Simulation_Parameter
        Parameters of the simulation

    Returns:
    --------
    None

    """

    n = param.n
    N = param.N
    nStep = param.nStep
    dt = param.dt
    A = param.A
    # var_x = param.var_x
    var_yx = param.var_yx
    
    x0 = np.random.normal(mu_prop, var_prop, (n, N))
    xtrue = np.random.normal(0, 1, (n, 1)) # true x
    noise = np.random.normal(0, var_yx, (m, 1))
    y = A @ xtrue + noise # y = Ax + n

    # mu_y = (param.sigma)@A.T@(param.sigma_yx_inv)@y

    xpart1 = param.gradient_flow(x0, y)
    xpart2 = param.langevin_sampling(x0,y,D = .1)

    fix, ax = plt.subplots(1,2, sharey = True, figsize = (10, 5), sharex=True)

    time = np.arange(nStep) * dt
    for ipaRT in range(0, 10):
        cmap20 = plt.colormaps['tab10']
        color = cmap20(ipaRT    )
        ax[0].plot(xpart1[0, ipaRT, 0], 0, marker='o', color=color)
        ax[0].plot(xpart1[0, ipaRT, :], time, linestyle = '-', color = color, linewidth = 0.5)
        ax[0].plot(xpart1[0, ipaRT, -1], nStep * dt, marker = 'o', color =color)
    # ax[0].scatter(mu_y[0], time[-1], marker = 'x', color = "black", s = 100) 

    for ipaRT in range(0, 5):
        cmap20 = plt.colormaps['tab10']
        color = cmap20(ipaRT)
        ax[1].plot(xpart2[0, ipaRT, 0], 0, marker = 'o', color = color)
        ax[1].plot(xpart2[0, ipaRT, :], time, linestyle = '-', color =color, linewidth = 0.5)
        ax[1].plot(xpart2[0, ipaRT, -1], nStep * dt, marker = 'o', color = color)
    # ax[1].scatter(mu_y[0], time[-1], marker = 'x', color = "black", s = 100)
    

    for axi in ax : axi.set_xlabel(r'$x_t$', fontweight = 'bold'); axi.set_ylabel(r'$t$', rotation = 0, fontweight = 'bold')
    ax[0].scatter([], [], marker = 'x', color = 'black', label = r'$\mu_y$', s = 100)
    ax[1].scatter([], [], marker = 'x', color = 'black', label = r'$\mu_y$', s = 100)

    ax[0].set_title('Gradient flow') 
    ax[1].set_title(r'Langevin diffusion $(D = .01)$')
    ax[0].grid()
    ax[1].grid()
    ax[0].legend(loc = 'lower left')
    ax[1].legend(loc = 'lower left')
    plt.suptitle(f"Path of a particular sample (y = {y})")
    plt.show()


def qt_and_moment_error_ani(mu_prop, var_prop, param):
    """
    Animation of the density function qt vs the exact posterior distribution p(x|y). We also display
    the error in the moment estimation (alpha = 1, 2) using gradient flow and langevin diffusion.

    Parameters:
    -----------
    mu_prop : float
        Mean of the proposal distribution
    var_prop : float
        Variance of the proposal distribution
    param : Simulation_Parameter
        Parameters of the simulation

    Returns:
    --------
    None

    """

    # if ~isinstance(param, Simulation_Parameter_Gaussian):
    #     raise ValueError("The parameter must be a Simulation_Parameter_Gaussian object")
    
    n = param.n    
    m = param.m    
    N = param.N 
    nStep = param.nStep
    dt = param.dt
    var_x = 1#param.var_x
    var_yx = param.var_yx
    A = param.A

    x0 = np.random.normal(mu_prop, np.sqrt(var_prop), (n, N))
    xtrue = np.random.normal(0, np.sqrt(var_x), (n, 1)) 
    noise = np.random.normal(0, np.sqrt(var_yx), (m, 1))
    y = A @ xtrue + noise 

    xpred_gf = param.gradient_flow(x0,y)
    qt_gf = param.compute_qt(mu_prop, var_prop, xpred_gf, y)
    mc_1_gf = np.array([param.compute_moment_IP(1, xpred_gf[..., k], qt_gf[..., k], y) for k in range(nStep)])
    mc_2_gf = np.array([param.compute_moment_IP(2, xpred_gf[..., k], qt_gf[..., k], y) for k in range(nStep)])
   
  
    xpred_ld = param.langevin_sampling(x0,y,D = .001)
    qt_ld = param.compute_qt(mu_prop, var_prop, xpred_gf, y)
    mc_1_ld = np.array([param.compute_moment_IP(1, xpred_ld[..., k], qt_ld[..., k], y) for k in range(nStep)])
    mc_2_ld = np.array([param.compute_moment_IP(2, xpred_ld[..., k], qt_ld[..., k], y) for k in range(nStep)])
    
    # true_qt = np.array([qt(xpred_gf[..., k], y, k * dt, mu_prop, var_prop, param) for k in range(nStep)])
    print(np.max(np.abs(mc_1_gf - mc_1_ld)))

    m1 = param.compute_moment(1, y)
    m2 = param.compute_moment(2, y)

    # mu_y = (param.sigma)@A.T@(param.sigma_yx_inv)@y

    fig, ax = plt.subplots(3, 2, figsize = (15, 9))
    line1, = ax[0, 0].plot([], [], lw=2)
    line2, = ax[0, 1].plot([], [], lw=2)
    line3, = ax[1, 0].plot([], [], lw=2)
    line4, = ax[1, 1].plot([], [], lw=2)
    line5, = ax[2, 0].plot([], [], lw=2)
    line6, = ax[2, 1].plot([], [], lw=2)

    def update(frame):
        for axi in ax.ravel(): 
            axi.cla()

        # Sort data for gradient flow
        sorted_indices_gf = np.argsort(xpred_gf[0, :, frame])
        sorted_xpred_gf = xpred_gf[0, sorted_indices_gf, frame]
        sorted_qt_gf = qt_gf[0, sorted_indices_gf, frame]


        # Sort data for langevin diffusion
        sorted_indices_ld = np.argsort(xpred_ld[0, :, frame])
        sorted_xpred_ld = xpred_ld[0, sorted_indices_ld, frame]
        sorted_qt_ld = qt_ld[0, sorted_indices_ld, frame]

    
        ax[0, 0].plot(sorted_xpred_gf, sorted_qt_gf, marker = "x", label = r"$\tilde{q}_t(x_t)$")
        ax[0, 1].plot(sorted_xpred_ld, sorted_qt_ld, marker = "x", label = r"$\tilde{q}_t(x_t)$")
        ax[0, 0].plot(np.linspace(-3, 3), param.p_xy(np.linspace(-3, 3).reshape(1, -1), y), color = 'red', label = r"$p(x|y)$")
        ax[0, 1].plot(np.linspace(-3, 3), param.p_xy(np.linspace(-3, 3).reshape(1, -1), y), color = 'red', label = r"$p(x|y)$")
        # ax[0, 0].plot(sorted_xpred_gf, true_qt[frame, sorted_indices_gf], color = 'green', label = r"$q_t(x_t)$", linestyle = "--")
        # ax[0, 1].plot(sorted_xpred_ld, true_qt[frame, sorted_indices_ld], color = 'green', label = r"$q_t(x_t)$", linestyle = "--")
    
        ax[0, 0].set_title(f"Gradient flow - Time: {frame * dt:.4f}")
        ax[0, 1].set_title(f"Langevin diffusion - Time: {frame * dt:.4f}")

        # ax[0, 0].vlines(mu_y, 0, 10, color = 'black', label = r'$\mu_y$', linestyle = "--")
        # ax[0, 1].vlines(mu_y, 0, 10, color = 'black', label = r'$\mu_y$', linestyle = "--")
        ax[0, 0].set_xlim(-3, 3)   
        ax[0, 1].set_xlim(-3, 3)  
        ax[0, 0].set_ylim(0, 1) 
        ax[0, 1].set_ylim(0, 1)
        ax[0, 0].legend()
        ax[0, 1].legend()

        # print(mc_1_gf[..., frame], m1)
        ax[1, 0].plot(dt * np.arange(frame), np.abs(mc_1_gf[:frame] - m1))
        ax[1, 0].set_title("Error in moment 1 estimation")
        ax[1, 0].set_xlabel("Number of iterations")
        ax[1, 0].set_ylabel(r"$\ell_1$-error")

        ax[1, 1].plot(dt * np.arange(frame), np.abs(mc_1_ld[:frame] - m1))
        ax[1, 1].set_title("Error in moment 1 estimation")
        ax[1, 1].set_xlabel("Number of iterations")
        ax[1, 1].set_ylabel(r"$\ell_1$-error")

        ax[2, 0].plot(dt * np.arange(frame), np.abs(mc_2_gf[:frame] - m2))
        ax[2, 0].set_title("Error in moment 2 estimation")
        ax[2, 0].set_xlabel("Number of iterations")
        ax[2, 0].set_ylabel(r"$\ell_1$-error")
        # ax[2, 1].grid()

        ax[2, 1].plot(dt * np.arange(frame), np.abs(mc_2_ld[:frame] - m2))
        ax[2, 1].set_title("Error in moment 2 estimation")
        ax[2, 1].set_xlabel("Number of iterations")
        ax[2, 1].set_ylabel(r"$\ell_1$-error")
        # ax[2, 1].grid()
        
        return line1, line2, line3, line4, line5, line6

    ani = animation.FuncAnimation(fig, update, frames=nStep, blit=False, interval = 10, repeat = True)
    plt.show()

    return 


def qt(x, y, t, mu_prop, sigma_prop, param):
    """
    Compute the exact density function qt at time t in the case x is normaly distributed.
    
    Parameters:
    ----------
    x : np.array
        The particles position.
    y : np.array
        The observation.
    t : float
        The time.
    mu_prop : float
        Mean of the proposal distribution.
    sigma_prop : float
        Variance of the proposal distribution.
    param : Simulation_Parameter
        Parameters of the simulation.

    Returns:
    -------
    np.array
        Density evalution at the particles position xt.    
    """

    mu_y  = (param.sigma)@param.A.T@(param.sigma_yx_inv)@y

    sigma = np.exp(-t*param.sigma_inv) * sigma_prop * np.exp(-t*param.sigma_inv)
    mu = mu_y + np.exp(-t*param.sigma_inv) * (mu_prop - mu_y)

    return 1/((2*np.pi)**(param.n/2)*np.sqrt(np.linalg.det(sigma))) * np.exp(-0.5 * np.einsum("ij, ji->i", (x - mu).T, np.linalg.inv(sigma) @ (x - mu)))


def get_sample_qt(t, y, mu_prop, sigma_prop, param):
    """
    Compute a sample from the density function qt at time t in the case x is normaly distributed.

    Parameters:
    ----------
    t : float
        The time.
    y : np.array
        The observation.
    mu_prop : float
        Mean of the proposal distribution.
    sigma_prop : float
        Variance of the proposal distribution.
    param : Simulation_Parameter
        Parameters of the simulation.
    
    Returns:
    -------
    np.array
        A sample from the density function qt at time t.
    
    """

    mu_y  = (param.sigma)@param.A.T@(param.sigma_yx_inv)@y  
    sigma = np.exp(-t*param.sigma_inv) * sigma_prop * np.exp(-t*param.sigma_inv)
    mu = mu_y + np.exp(-t*param.sigma_inv) * (mu_prop - mu_y)

    return np.random.normal(mu, np.sqrt(sigma), (param.n, param.N))

def merged_IP_ratio(mu_prop, var_prop, param):

    """
    Comparison between the moment estimation using different methods and 
    calculation of the ratio plot E_{q^t_y}[(p(x|y) / q_t(x_t))^2]

    Parameters:
    -----------
    mu_prop : float
        Mean of the proposal distribution
    var_prop : float
        Variance of the proposal distribution
    param : Simulation_Parameter
        Parameters of the simulation
    
    Returns:
    --------
    None

    """

    n = param.n    
    m = param.m    
    N = param.N 
    nStep = param.nStep
    dt = param.dt

    x0 = np.random.normal(mu_prop, np.sqrt(var_prop), (n, N))
    xtrue = np.random.normal(0, np.sqrt(param.var_x), (n, 1))
    noise = np.random.normal(0, np.sqrt(param.var_yx), (m, 1))
    y = param.A @ xtrue + noise

    mu_y = (param.sigma) @ param.A.T @ (param.sigma_yx_inv) @ y

    x_sample = np.empty((n, N, nStep))
    x_pred = param.gradient_flow(x0, y)

    qt_sample = np.empty((n, N, nStep))
    qt_pred = param.compute_qt(mu_prop, var_prop, x_pred, y, param)
    qt_no_resampling = np.empty((n, N, nStep))

    ratio = np.empty((n, nStep))

    for k in range(nStep):
        x_sample[..., k] = get_sample_qt(k * dt, y, mu_prop, var_prop, param)
        qt_sample[..., k] = qt(x_sample[..., k], y, k * dt, mu_prop, var_prop, param)
        qt_no_resampling[..., k] = qt(x_pred[..., k], y, k * dt, mu_prop, var_prop, param)
        ratio[..., k] = np.mean((param.p_xy(x_sample[..., k], y) / qt_sample[..., k]) ** 2, axis=1)

    m1 = param.compute_moment(1, y)
    m2 = param.compute_moment(2, y)

    m1_mc = np.array([param.compute_moment_IP(1, x_sample[..., k], qt_sample[..., k], y) for k in range(nStep)])
    m2_mc = np.array([param.compute_moment_IP(2, x_sample[..., k], qt_sample[..., k], y) for k in range(nStep)])

    m1_mc_pred = np.array([param.compute_moment_IP(1, x_pred[..., k], qt_pred[..., k], y) for k in range(nStep)])
    m2_mc_pred = np.array([param.compute_moment_IP(2, x_pred[..., k], qt_pred[..., k], y) for k in range(nStep)])

    m1_mc_no_resampling = np.array([param.compute_moment_IP(1, x_pred[..., k], qt_no_resampling[..., k], y) for k in range(nStep)])
    m2_mc_no_resampling = np.array([param.compute_moment_IP(2, x_pred[..., k], qt_no_resampling[..., k], y) for k in range(nStep)])

    fig, ax = plt.subplots(1, 2)

    ax[ 0].scatter(dt * np.arange(nStep), np.abs(m1_mc - m1), marker=".", label=r"Estimator using resampling on the ground truth $q_t(\cdot)$")
    ax[ 0].plot(dt * np.arange(nStep), np.abs(m1_mc_pred - m1), color="red", label=r"Estimator using the particles $x_t$ and $\tilde{q}_t(x_t)$")
    ax[ 0].plot(dt * np.arange(nStep), np.abs(m1_mc_no_resampling - m1), color="k", linestyle="--", label=r"Estimator using the particles $x_t$ and the ground truth $q_t(\cdot)$")
    ax[ 0].set_title("Moment 1")
    ax[ 0].set_xlabel("Time")
    ax[ 0].set_ylabel(r"$\ell_1$-error")
    ax[ 0].grid(which='both')
    ax[ 0].legend()

    ax[1].scatter(dt * np.arange(nStep), np.abs(m2_mc - m2), marker=".", label=r"Estimator using resampling on the ground truth $q_t(\cdot)$")
    ax[1].plot(dt * np.arange(nStep), np.abs(m2_mc_pred - m2), color="red", label=r"Estimator using the particles $x_t$ and $\tilde{q}_t(x_t)$")
    ax[1].plot(dt * np.arange(nStep), np.abs(m2_mc_no_resampling - m2), color="k", linestyle="--", label=r"Estimator using the particles $x_t$ and the ground truth $q_t(\cdot)$")
    ax[1].set_title("Moment 2")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel(r"$\ell_1$-error")
    ax[1].grid(which='both')
    ax[1].legend()

    # ax[1, 0].scatter(dt * np.arange(nStep), ratio[0, :], marker=".")
    # ax[1, 0].set_title(r"$E_{x \sim q^t_y}[(p(x|y) / q_y^t(x))^2]$")
    # ax[1, 0].set_xlabel("Number of iterations")
    # ax[1, 0].set_ylabel("Ratio")
    # ax[1, 0].grid(which='both')

    # ax[1, 1].remove()

    # plt.tight_layout()
    plt.show()


def compare_moment(mu_prop, var_prop, param):

    if ~isinstance(param, Simulation_Parameter_Gaussian):
        raise ValueError("The parameter must be a Simulation_Parameter_Gaussian object")
    
    n = param.n    
    m = param.m    
    N = param.N 
    nStep = param.nStep
    dt = param.dt
    var_x = param.var_x
    var_yx = param.var_yx
    A = param.A

    x0 = np.random.normal(mu_prop, np.sqrt(var_prop), (n, N))
    xtrue = np.random.normal(0, np.sqrt(var_x), (n, 1)) # true x
    noise = np.random.normal(0, np.sqrt(var_yx), (m, 1))
    y = A @ xtrue + noise # y = Ax + n

    xpred = param.gradient_flow(x0, y)
    # xpred = langevin_sampling(x0, y, param, D = .05)
    qt = param.compute_qt(mu_prop, var_prop, xpred, y)
    

    m1 = param.compute_moment(1, y)
    m2 = param.compute_moment(2, y)

    m2_mc = np.array([param.compute_moment_IP(2, xpred[..., k], qt[..., k], y) for k in range(nStep)])
    m1_mc = np.array([param.compute_moment_IP(1, xpred[..., k], qt[..., k], y) for k in range(nStep)])

    
    KL_div = KL_curve(xpred, qt, y, param)  
    # true_KL = true_KL_curve(param, mu_prop, var_prop, y)

    fig, ax = plt.subplots(2, 2, figsize = (15, 9))

    ax[0, 0].plot(dt*np.arange(nStep), np.abs(m1_mc - m1), marker = ".")
    # ax[0, 0].plot(dt*np.arange(nStep), np.abs(m1_mc_pred - m1)*np.ones(nStep), marker = ".", color = "r")
    # ax[0, 0].semilogx(m1_mc, marker = ".")
    ax[0, 0].set_title("Moment 1")
    ax[0, 0].set_xlabel("Number of iterations")
    ax[0, 0].set_ylabel(r"$\ell_1$-error")
    ax[0, 0].grid(which = 'both')
    ax[0, 1].plot(dt * np.arange(nStep),np.abs(m2_mc - m2), marker = ".")
    # ax[0, 1].plot(dt * np.arange(nStep),np.abs(m2_mc_pred - m2)*np.ones(nStep), marker = ".", color = "r")
    # ax[0, 1].semilogx(m2_mc, marker = ".")
    ax[0, 1].set_title("Moment 2")
    ax[0, 1].set_xlabel("Number of iterations")
    ax[0, 1].set_ylabel(r"$\ell_1$-error")
    ax[0, 1].grid(which = 'both')

    ax[1, 0].plot(dt*np.arange(nStep), KL_div, marker = ".")  
    # ax[1, 0].plot(dt*np.arange(nStep), true_KL, color = "r"
    # ax[1, 0].semilogx(np.arange(nStep), true_KL, color = "r")
    ax[1, 0].set_title(r"$D\left(p(x|y) || q_t(x_t)\right)$")
    ax[1, 0].set_xlabel("Number of iterations") 
    ax[1, 0].set_ylabel("KL divergence")
    ax[1, 0].grid(which = 'both')
    ax[1, 1].remove()
    plt.suptitle("Error in moment estimation")
    print("Error min moment 1 :" , np.min(np.abs(m1_mc - m1)))
    print("Error min moment 2 :" , np.min(np.abs(m2_mc - m2)))
    print("y = ", y)
    print("xtrue = ", xtrue)
    plt.show()


  
def KL_divergence_1D(mu_p, mu_q, var_p, var_q): 
    return .5 * (var_p/var_q + (mu_q - mu_p)**2/var_q - 1 + np.log(var_q/var_p))


def KL_curve(xpred, qt, y, param):
    nStep = xpred.shape[-1] 
    mu_qt = np.array([np.mean(xpred[..., k] * qt[..., k], axis = 1) for k in range(nStep)])
    var_qt = np.array([np.mean((xpred[..., k] - mu_qt[k])**2 * qt[..., k], axis = 1) for k in range(nStep)])

    var = 1 / (1/param.var_x + 1/param.var_yx) 
    mu_xy = A @ y * var/param.var_yx 

    var_y = A @ A.T * param.var_x + param.var_yx * np.eye(m)
    var_xy = param.var_yx * param.var_x/var_y

    KL_div = np.array([KL_divergence_1D(mu_xy, mu_qt[k], var_xy, var_qt[k]) for k in range(nStep)])

    return KL_div[:, 0, 0]


def true_KL_curve(param, mu_prop, var_prop, y):
    dt = param.dt
    t = (np.arange(param.nStep) * dt).reshape(-1, 1)
    mu_y = (param.sigma) @ param.A.T @ (param.sigma_yx_inv) @ y


    # Reshape t to match dimensions for broadcasting if necessary
    exp_term = np.exp(t * param.sigma_inv)
    
    KL_curve = 1/2 * (-2 * t * np.trace(param.sigma_inv) + np.log(np.linalg.det(param.sigma)) - np.log(np.linalg.det(param.sigma_x)) +
                      np.einsum("ij, ji->i", (mu_prop - mu_y).T, param.sigma_inv @ (mu_prop - mu_y)) +
                         exp_term * exp_term * param.sigma * 1/var_prop - param.m)

    return KL_curve


def display_qt(mu_prop, var_prop, param):
    n = param.n    
    m = param.m    
    N = param.N 
    nStep = param.nStep
    dt = param.dt
    var_x = 1#param.var_x
    var_yx = param.var_yx
    A = param.A

    x0 = np.random.normal(mu_prop, np.sqrt(var_prop), (n, N))
    xtrue = np.random.normal(0, np.sqrt(var_x), (n, 1)) 
    noise = np.random.normal(0, np.sqrt(var_yx), (m, 1))
    y = A @ xtrue + noise 

    xt = param.gradient_flow(x0, y)
    qt = param.compute_qt(mu_prop, var_prop, xt, y)


    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    line, = ax.plot([], [], lw=2)

    def update(frame):
        ax.cla()
        sorted_idx = np.argsort(xt[0, :, frame], axis = 0)
        
        qt_sorted = qt[0, sorted_idx, frame]
        xt_sorted = xt[0, sorted_idx, frame]

        ax.plot(xt_sorted, qt_sorted, marker = "x", color = "k", label = r"$\tilde{q}_y^t(x_t)$ ")
        ax.set_title("t = {:.3f}".format(param.dt*frame))
        ax.set_xlim(-3, 3)
        ax.legend()
        return line


    anim = animation.FuncAnimation(fig, update, frames=nStep, interval=200)
    plt.show()


if __name__ == "__main__":

    nStep = 10
    N = 50
    dt = 1e-2   
    n = 1
    m = 1
    var_x = 1.0/np.sqrt(2)
    var_yx = 1.0/np.sqrt(2)  
    A = np.eye(m, n)
    param1 = Simulation_Parameter_Gaussian(nStep, N, dt, n, m, A, var_x, var_yx)
    
    # display_qt(0, 1, param)

    weight = np.array([.25, .5, .25])
    mu = np.array([-1, 0, 1])
    sigma = np.array([.1, .2, .1])

    param2 = Simulation_Parameter_Gaussian_Mixture(nStep, N, dt, n, m, A, weight, mu, sigma, var_yx)
    # print(param2.compute_moment(0, np.zeros((m, 1))))
    qt_and_moment_error_ani(0, np.sqrt(2), param2)
    # print(param2.gradV(100*np.ones((m, 1)), np.zeros((m, 1))))

    # draw_particul_path(0, 1, param)
    # display_qt(0, 1, param2)
    # x = np.linspace(-5, 5, 100).reshape(1, -1)
    # p1 = param1.p_xy(x, np.zeros((m, 1)))
    # p2 = param2.p_xy(x, np.zeros((m, 1)))
    # print(p1.shape, p2.shape)

    # print(compute_moment(0, np.ones((m, 1)), param2))

    # plt.plot(x.reshape(-1), p.reshape(-1))
    # plt.show()


    # f = lambda x: x[0]**2 + x[1]**2
    # x = np.array([[1, 1], [2, 2]], dtype = np.float64).reshape(2, 3)
# 
    # div = param.div(f, x)
    # plt.plot(x.reshape(-1), div.reshape(-1))
    # plt.hlines(0, -5, 5, color = 'red')
    # plt.show()

# 
    # x = np.linspace(-5, 5, 100).reshape(1, -1)
    # gradV =  param.gradV(x, np.zeros((m, 1)))
    # gradV2 = param.gradV2(x, np.zeros((m, 1)))
    # p = param.p_yx(np.zeros((m, 1)), x) * param.p_x(x)
    
    # plt.plot(x.reshape(-1), gradV.reshape(-1))
    # plt.plot(x.reshape(-1), gradV2.reshape(-1))
    # plt.hlines(0, -5, 5, color = 'red')
    # plt.show()

    # display_qt(0, 2, param)
    # perfect_IP(0, 1, param)
    # merged_IP_ratio(0, 1, param)