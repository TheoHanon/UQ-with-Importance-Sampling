# Author : Theo Hanon
# Created : 2024-07-01

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from PARTI.old.param  import Simulation_Parameter_Gaussian, Simulation_Parameter_Gaussian_Mixture
from scipy.stats import gaussian_kde
import seaborn as sns
from  matplotlib.lines import Line2D
import pandas as pd
import warnings
import copy
import scipy.linalg as sc



def filter_qt_1D(xt, qt, dx = 1e-1):
    """
    Filter the density function qt to remove the polution on the projection of qt along the given axis. 
    The xt and qt should be already sorted along the axis.

    Parameters:
    -----------
    xt : np.array
        The particles position.
    qt : np.array
        The density function qt.
    
    Returns:
    --------
    np.array
        The filtered density function qt.
    
    """


    interval = np.arange(xt[0], xt[-1] + dx, dx)
    interval_idx = np.digitize(xt, interval) - 1 

    filtered_qt = []
    filtered_xt = []

    for i in range(len(interval) -1):
        mask  = (interval_idx == i)
        if np.any(mask):
            max_idx = np.argmax(qt[mask])
            filtered_qt.append(np.mean(qt[mask][max_idx]))
            filtered_xt.append(np.mean(xt[mask][max_idx]))

    return np.array(filtered_xt), np.array(filtered_qt)


def qt_1D(param, time_frame, component=0, dx = 1e-1, save_as=None, ax = None):
    """
    Plot the density function qt at a specific time frame.

    Parameters:
    -----------
    param : Simulation_Parameter
        Parameters of the simulation
    time_frame : int
        Time frame at which to visualize the density function qt
    component : int, optional
        Component index for component to display parameter (default is 0)
    save_as : str, optional
        File path to save the plot (default is None)

    Returns:
    --------
    None
    """

    if param.n != 1 and component < 0 or component >= param.n:
        raise ValueError("The component must be between 0 and the dimension of the parameter")
        
    if param.n == 1 and component != 0:
        raise ValueError("The component must be 1 for a 1D")


    xt = np.array(param.xt)
    qt = np.array(param.qt)

    xt = np.where(np.isinf(xt), np.nan, xt)
    qt = np.where(np.isinf(qt), np.nan, qt)

    idx_sort = np.argsort(xt[time_frame, :, component])
    xt = xt[time_frame, idx_sort, component]
    qt = qt[time_frame, idx_sort]

    if param.n != 1:
        xt, qt = filter_qt_1D(xt, qt, dx = dx)

    if param.n == 1:
        x = np.linspace(-3, 3, 1000)
        z = param.p_xy(x.reshape(-1, 1), param.y)
    else :
        x = np.linspace(-3, 3, 1000)
        grid = np.tile(np.zeros(n), (1000, 1))
        grid[:, component] = x
        z = param.p_xy(grid, param.y)

    if ax is None:
        fig, axp = plt.subplots(1, 1, figsize=(5, 5))
    else :
        axp = ax

    cmap_OrRd = plt.get_cmap('OrRd')

    axp.plot(x, z, label=r"$p(x|y)$", color = cmap_OrRd(.8))
    axp.plot(xt, qt, label=r"$q_t(x_t)$", linestyle="--", color = 'k')
    axp.scatter(xt, qt, marker='D', label=rf'$x(t = {param.dt*time_frame:.3f})$', s=5, color="k")

    axp.set_xlabel(r"$x$")
    axp.set_ylabel(r"$z$", rotation = 0, labelpad = 10)
    axp.legend()
    axp.set_xlim(-2,2)


    if save_as is not None:
        plt.savefig(save_as)

    elif ax is not None:
        return axp
    
    else:
        plt.show()


def qt_2D(param, time_frame, component1=0, component2=1, dx=1e-1, kde=False, save_as=None, ax = None):
    """
    Plot a 2D visualization of the parameter `param` at a specific time frame for specified components.

    Parameters:
    - param: An object representing the parameter.
    - time_frame: The time frame at which to visualize the parameter.
    - component1: The first component index to project on.
    - component2: The second component index to project on.
    - dx: Interval size for filtering.
    - kde: Use KDE for density estimation.
    - save_as: Optional. If provided, the plot will be saved as the specified file.

    Raises:
    - ValueError: If the dimension of the parameter is less than 2.

    Returns:
    None
    """

    if param.n < 2:
        raise ValueError("The dimension of the parameter must be at least 2")
    
    if kde:
        warnings.warn("You are using KDE estimation for the density function. The representation may not be accurate.")
    
    xt = np.array(param.xt)
    qt = np.array(param.qt)

    xt = np.where(np.isinf(xt), np.nan, xt)
    qt = np.where(np.isinf(qt), np.nan, qt)

    x1, x2 = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    x = np.zeros((x1.size, param.n))
    x[:, component1] = x1.flatten()
    x[:, component2] = x2.flatten()
    z = param.p_xy(x, param.y)
    
    if ax is None:
        g = sns.JointGrid(x=xt[time_frame, :, component1], y=xt[time_frame, :, component2])
        g.figure.figsize = (6, 6)
        g.ax_joint.set_xlim(-1, 1)
        g.ax_joint.set_ylim(-1, 1)
    else : 
        g = ax
    
    g.ax_joint.tricontourf(x1.flatten(), x2.flatten(), z, levels=100, cmap="Reds")
    g.ax_joint.tricontour(x1.flatten(), x2.flatten(), z, levels=10, colors="k", linewidths=.5)

    sns.scatterplot(
        x=xt[time_frame, :, component1], y=xt[time_frame, :, component2], 
        marker='D', label=rf'$x(t = {param.dt*time_frame:.3f})$', s=10, color="k",
        ax=g.ax_joint
    )
    
    cmap_OrRd = plt.get_cmap('OrRd')

    xt_x1 = np.linspace(-3, 3, 1000)
    xt_x2 = np.linspace(-3, 3, 1000)

    grid_x1 = np.tile(np.zeros(param.n), (1000, 1))
    grid_x2 = np.tile(np.zeros(param.n), (1000, 1))

    grid_x1[:, component1] = xt_x1
    grid_x2[:, component2] = xt_x2
    
    z_x1_weights = param.p_xy(grid_x1, param.y)
    z_x2_weights = param.p_xy(grid_x2, param.y)

    g.ax_marg_x.plot(xt_x1, z_x1_weights, color=cmap_OrRd(.8), linestyle="-")
    g.ax_marg_y.plot(z_x2_weights, xt_x2, color=cmap_OrRd(.8), linestyle="-")

    # Marginal KDE
    if not kde:
        idx_sort_x = np.argsort(xt[time_frame, :, component1])
        idx_sort_y = np.argsort(xt[time_frame, :, component2])

        xt_sorted_x = xt[time_frame, idx_sort_x, component1]
        xt_sorted_y = xt[time_frame, idx_sort_y, component2]

        qt_sorted_x = qt[time_frame, idx_sort_x]
        qt_sorted_y = qt[time_frame, idx_sort_y]

        xt_x1, qt_x1 = filter_qt_1D(xt_sorted_x, qt_sorted_x, dx=dx)
        xt_x2, qt_x2 = filter_qt_1D(xt_sorted_y, qt_sorted_y, dx=dx)
    else:
        kde_x1 = gaussian_kde(xt[time_frame, :, component1])
        kde_x2 = gaussian_kde(xt[time_frame, :, component2])

        qt_x1 = kde_x1(xt_x1)
        qt_x2 = kde_x2(xt_x2)

    g.ax_marg_x.plot(xt_x1, qt_x1, color='k', linestyle="--")
    g.ax_marg_y.plot(qt_x2, xt_x2, color='k', linestyle="--")

    g.ax_joint.vlines(0, -2, -1.8, color='black', linestyle="-.", linewidth=.7, alpha=.8)
    g.ax_joint.vlines(0, 1.8, 2, color='black', linestyle="-.", linewidth=.7, alpha=.8)
    g.ax_joint.hlines(0, -2, -1.8, color='black', linestyle="-.", linewidth=.7, alpha=.8)
    g.ax_joint.hlines(0, 1.8, 2, color='black', linestyle="-.", linewidth=.7, alpha=.8)
    g.ax_joint.annotate("", xytext=(-1.9, -0.2), xy=(-1.9, 0), arrowprops=dict(facecolor='k', arrowstyle='-|>'))
    g.ax_joint.annotate("", xytext=(1.9, -0.2), xy=(1.9, 0), arrowprops=dict(facecolor='k', arrowstyle='-|>'))
    g.ax_joint.annotate("", xytext=(-.2, 1.9), xy=(0, 1.9), arrowprops=dict(facecolor='k', arrowstyle='-|>'))
    g.ax_joint.annotate("", xytext=(-.2, -1.9), xy=(0, -1.9), arrowprops=dict(facecolor='k', arrowstyle='-|>'))

    # Adjusting the axes with Matplotlib
    g.ax_joint.set_xlabel(f'x{component1}')
    g.ax_joint.set_ylabel(f'x{component2}')
    g.ax_marg_x.set_xlim(-2, 2)
    g.ax_marg_y.set_ylim(0, 1)

    g.ax_marg_y.set_ylim(-2, 2)
    g.ax_marg_y.set_xlim(0, 1)

    g.ax_marg_x.set_yticklabels([])
    g.ax_marg_y.set_xticklabels([])

    handles, labels = g.ax_joint.get_legend_handles_labels()
    g.ax_joint.legend(handles=handles, labels=labels, loc='upper right')

    lines = [
        Line2D([0], [0], color=cmap_OrRd(.8), linewidth=2, linestyle="-", label=r'$p(x|y)$'),
        Line2D([0], [0], color='k', linewidth=2, linestyle="--", label=r'$q_t(x_t)$'),
        Line2D([0], [0], marker='D', color='k', label=rf'$x(t = {param.dt*time_frame:.3f})$', markersize=3, linestyle="", markerfacecolor='k'),
    ]
    g.ax_joint.legend(handles=lines)

    if save_as is not None:
        g.savefig(save_as)

    elif ax is not None:
        return g
    else:
        plt.show()

def qt_1D_slider(param, component=0, dx=1e-1, kde=False):
    """
    Plot a 1D visualization of the density function qt at different time frames using a slider.

        Parameters:
        -----------
        param : Simulation_Parameter
            Parameters of the simulation.
        component : int, optional
            Component index for component to display parameter (default is 0).
        dx : float, optional
            Interval size for filtering (default is 1e-1).
        kde : bool, optional
            Use KDE for density estimation (default is False).

        Returns:
        --------
        None
    """

    fig, ax = plt.subplots(1, 1)
    ax = qt_1D(param, time_frame=0, component=component, dx=dx, ax=ax)
    ax.set_ylim(0, 2)
    ax.set_xlim(-3, 3)

    fig.subplots_adjust(bottom=0.25)
    
    slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax=slider_ax, label='Time', valmin=0, valmax=param.nStep - 1, valinit=0, valstep=1)

    def update(val): 
        nonlocal ax   
        ax.clear()
        ax = qt_1D(param, int(slider.val), component=component, dx=dx, ax=ax)
        fig.canvas.draw_idle()
        ax.set_ylim(0, 2)
        ax.set_xlim(-3, 3)

    slider.on_changed(update)
    plt.show()


def qt_2D_slider(param, component1=0, component2=1, dx=1e-1, kde=False):

    g = sns.JointGrid()
    g = qt_2D(param, time_frame=0, component1=component1, component2=component2, dx=dx, kde=kde, ax=g)

    g.figure.subplots_adjust(bottom=0.25)

    slider_ax = g.figure.add_axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax=slider_ax, label='Time', valmin=0, valmax=param.nStep - 1, valinit=0, valstep=1)

    def update(val):
        nonlocal g
        g.ax_joint.clear()
        g.ax_marg_x.clear()
        g.ax_marg_y.clear()
        g = qt_2D(param, int(slider.val), component1=component1, component2=component2, dx=dx, kde=kde, ax=g)
        g.figure.canvas.draw_idle()

        g.ax_marg_x.set_xticklabels([])
        g.ax_marg_y.set_yticklabels([])
        # g.ax_marg_x.set_xlim(-2, 2)


    slider.on_changed(update)
    plt.show()


def qt_1D_animation(param, component=0, dx=1e-1, kde=False, interval=100, save_as  = None):
    """
    Create an animation of the 1D visualization of the density function qt over time.
    """
    fig, ax = plt.subplots()
    ax.set_ylim(0, 2)
    ax.set_xlim(-3, 3)

    def update(frame):
        nonlocal ax
        ax.clear()
        ax = qt_1D(param, time_frame=frame, component=component, dx=dx, ax=ax)
        ax.set_ylim(0, 2)
        ax.set_xlim(-3, 3)
        return ax,

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, param.nStep), interval=interval)

    if save_as is not None:
        ani.save(save_as, writer='ffmpeg')
    else :
        plt.show()

def qt_2D_animation(param, component1=0, component2=1, dx=1e-1, kde=False, interval=100, save_as = None):
    """
    Create an animation of the 2D visualization of the density function qt over time.
    """
    g = sns.JointGrid()
    g.figure.subplots_adjust(bottom=0.25)

    def update(frame):
        nonlocal g
        g.ax_joint.clear()
        g.ax_marg_x.clear()
        g.ax_marg_y.clear()
        g = qt_2D(param, time_frame=frame, component1=component1, component2=component2, dx=dx, kde=kde, ax=g)
        return g.ax_joint,

    ani = animation.FuncAnimation(g.figure, update, frames=np.arange(0, param.nStep), interval=interval)
    if save_as is not None:
        ani.save(save_as, writer='ffmpeg')
    else :
        plt.show()



def particle_path_1D(param, time_frame = -1, n_particles = 10, component = 0, save_as = None, ax = None):


    if ax is None:
        fig, axp = plt.subplots(1, 1, figsize=(10, 5))
    else :
        axp = ax

    cmap20 = plt.get_cmap('tab20')

    time = np.arange(param.nStep) * param.dt

    axp.hlines(0, np.min(param.xt[0, :n_particles, component]), np.max(param.xt[0, :n_particles, component]), color='black', linestyle="--", linewidth=0.5)
    for i in range(n_particles):
        axp.plot(param.xt[0, i, component], time[0], color='grey', marker = ".", markersize = 10)
        axp.plot(param.xt[:time_frame, i, component], time, color='k', linewidth=0.5)
        axp.plot(param.xt[time_frame, i, component], time[-1],  color='r', marker = ".", markersize = 10)

    

    axp.set_xlabel(r"$x_0$")
    axp.set_ylabel(f"t", rotation = 0)

    lines = [Line2D([0], [0], color='k', linewidth=1, linestyle="-", label=r"$x_t$"), 
             Line2D([0], [0], color='r', marker=".", markersize=10, linestyle="", label=r"$x_(t=t_{end})$"),
             Line2D([0], [0], color='grey', marker=".", markersize=10, linestyle="", label=r"$x(t=0)$")]
    axp.legend(handles=lines)
    axp.grid()
    if save_as is not None:
        plt.savefig(save_as)
    else:
        plt.show()


def particle_path_1D_slider(param, component=0, n_particles = 10):

    fig, ax = plt.subplots(1, 1)
    ax = particle_path_1D(param, time_frame=0, component=component, ax=ax)
    ax.set_ylim(0, 2)
    ax.set_xlim(-3, 3)

    fig.subplots_adjust(bottom=0.25)
    
    slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax=slider_ax, label='Time', valmin=0, valmax=param.nStep - 1, valinit=0, valstep=1)

    def update(val): 
        nonlocal ax   
        ax.clear()
        ax = particle_path_1D(param, int(slider.val), component=component, ax=ax, n_particles = n_particles)
        fig.canvas.draw_idle()
        ax.set_ylim(0, 2)
        ax.set_xlim(-3, 3)

    slider.on_changed(update)
    plt.show()

def particle_path_1D_animation(param, component=0, interval=100, save_as = None, n_particles = 10):
    """
    Create an animation of the 1D particle path visualization over time.
    """
    fig, ax = plt.subplots()
    ax.set_ylim(0, 2)
    ax.set_xlim(-3, 3)

    def update(frame):
        nonlocal ax
        ax.clear()
        ax = particle_path_1D(param, time_frame=frame, component=component, ax=ax, n_particles = n_particles)
        ax.set_ylim(0, 2)
        ax.set_xlim(-3, 3)
        return ax,

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, param.nStep), interval=interval)
    if save_as is not None:
        ani.save(save_as, writer='ffmpeg')
    else :
        plt.show()


def particle_path_2D(param, time_frame = -1, n_particles = 10, component1 = 0, component2 = 1, save_as = None, ax = None):


    if ax is None:
        fig, axp = plt.subplots(1, 1, figsize=(9, 9))
    else :
        axp = ax
    

    for i in range(n_particles):
        axp.plot(param.xt[0, i, component1], param.xt[0, i, component2], color="grey", marker = ".", markersize = 10)
        axp.plot(param.xt[:time_frame, i, component1], param.xt[:time_frame, i, component2], color='k', linewidth=0.5)
        axp.plot(param.xt[time_frame, i, component1], param.xt[time_frame, i, component2], color='r', marker = ".", markersize = 10)


    x1, x2 = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    x = np.zeros((x1.size, param.n))
    x[:, component1] = x1.flatten()
    x[:, component2] = x2.flatten()
    z = param.p_xy(x, param.y)



    axp.tricontourf(x1.flatten(), x2.flatten(), z, levels=100, cmap="Reds")
    axp.tricontour(x1.flatten(), x2.flatten(), z, levels=10, colors="k", linewidths=.5)

    axp.set_xlabel(f"$x_{component1}$")
    axp.set_ylabel(f"$x_{component2}$", rotation = 0)
    axp.grid()

    lines = [Line2D([0], [0], color='k', linewidth=1, linestyle="-", label=r"$x_t$"), 
             Line2D([0], [0], color='r', marker=".", markersize=10, linestyle="", label=r"$x_(t=t_{end})$"),
             Line2D([0], [0], color='grey', marker=".", markersize=10, linestyle="", label=r"$x(t=0)$")]

    axp.legend(handles=lines)
    
    if save_as is not None:
        plt.savefig(save_as)
    elif ax is not None:
        return axp
    else:
        plt.show()

def particle_path_2D_slider(param, component1=0, component2=1, n_particles = 10):
    
        fig, ax = plt.subplots(1, 1)
        ax = particle_path_2D(param, time_frame=0, component1=component1, component2=component2, ax=ax)
        ax.set_ylim(-3, 3)
        ax.set_xlim(-3, 3)
    
        fig.subplots_adjust(bottom=0.25)
        
        slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax=slider_ax, label='Time', valmin=0, valmax=param.nStep - 1, valinit=0, valstep=1)
    
        def update(val): 
            nonlocal ax   
            ax.clear()
            ax = particle_path_2D(param, int(slider.val), component1=component1, component2=component2, ax=ax, n_particles = n_particles)
            fig.canvas.draw_idle()
            ax.set_ylim(-3, 3)
            ax.set_xlim(-3, 3)
    
        slider.on_changed(update)
        plt.show()


def particle_path_2D_animation(param, component1=0, component2=1, interval=100, n_particles=10,  save_as = None):
    """
    Create an animation of the 2D particle path visualization over time.
    """
    fig, ax = plt.subplots()
    ax.set_ylim(-3, 3)
    ax.set_xlim(-3, 3)

    def update(frame):
        nonlocal ax
        ax.clear()
        ax = particle_path_2D(param, time_frame=frame, component1=component1, component2=component2, ax=ax, n_particles=n_particles)
        ax.set_ylim(-3, 3)
        ax.set_xlim(-3, 3)
        return ax,

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, param.nStep), interval=interval)

    if save_as is not None:
        ani.save(save_as, writer='ffmpeg')
    else :
        plt.show()



def particle_path_3D(param, time_frame = -1, n_particles = 10, component1 = 0, component2 = 1, component3 = 2, save_as = None, ax = None):


    if param.n < 3:
        raise ValueError("The dimension of the parameter must be at least 3")
    

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        axp = fig.add_subplot(111, projection='3d')
    else:
        axp = ax



    for i in range(n_particles):
        axp.plot(param.xt[0, i, component1], param.xt[0, i, component2], param.xt[0, i, component3], color="grey", marker = ".", markersize = 10)
        axp.plot(param.xt[:time_frame, i, component1], param.xt[:time_frame, i, component2], param.xt[:time_frame, i, component3], color='k', linewidth=0.5)
        axp.plot(param.xt[time_frame, i, component1], param.xt[time_frame, i, component2], param.xt[time_frame, i, component3], color='r', marker = ".", markersize = 10)

    axp.set_xlabel(f"$x_{component1}$")
    axp.set_ylabel(f"$x_{component2}$")
    axp.set_zlabel(f"$x_{component3}$")

    lines = [Line2D([0], [0], color='k', linewidth=1, linestyle="-", label=r"$x_t$"), 
             Line2D([0], [0], color='r', marker=".", markersize=10, linestyle="", label=r"$x_(t=t_{end})$"),
             Line2D([0], [0], color='grey', marker=".", markersize=10, linestyle="", label=r"$x(t=0)$")]

    axp.legend(handles=lines)
    
    if save_as is not None:
        plt.savefig(save_as)

    elif ax is not None:
        return axp
    else:
        plt.show()

def particle_path_3D_slider(param, component1=0, component2=1, component3=2, n_particles = 10):

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax = particle_path_3D(param, time_frame=0, component1=component1, component2=component2, component3=component3, ax=ax)

    fig.subplots_adjust(bottom=0.25)
    
    slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax=slider_ax, label='Time', valmin=0, valmax=param.nStep - 1, valinit=0, valstep=1)

    def update(val): 
        nonlocal ax   
        ax.clear()
        ax = particle_path_3D(param, int(slider.val), component1=component1, component2=component2, component3=component3, ax=ax, n_particles = n_particles)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

def particle_path_3D_animation(param, component1=0, component2=1, component3=2, interval=100, n_particles=10, save_as = None):
    """
    Create an animation of the 3D particle path visualization over time.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        nonlocal ax
        ax.clear()
        ax = particle_path_3D(param, time_frame=frame, component1=component1, component2=component2, component3=component3, ax=ax, n_particles=n_particles)
        return ax,

    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, param.nStep), interval=interval)

    if save_as is not None:
        ani.save(save_as, writer='ffmpeg')
    else :
        plt.show()


def MSE(param, n_estimator=1, N=[50, 100, 500, 1000, 10000, 20000], save_as=None):
    if len(N) != 6:
        raise ValueError(f"The number of sample sizes must be 6 (N={len(N)}!=6)")

    m1 = param.compute_moment(1)
    m2 = param.compute_moment(2)

    mse = np.zeros((n_estimator, len(N), 2, param.nStep))
    param_temp = copy.copy(param)

    for i in range(n_estimator):
        for j, n_sample in enumerate(N):
            param_temp.N = n_sample
            if not param_temp.uniform:
                param_temp.x0 = np.random.multivariate_normal(param_temp.mu_prop, param_temp.sigma_prop, n_sample)
            else:
                param_temp.x0 = np.random.uniform(-param_temp.a_unif, param_temp.a_unif, (n_sample, param_temp.n))

            xt_temp = param_temp.gradient_flow()
            qt_temp = param_temp.compute_qt(xt_temp)

            mc1 = np.array([param_temp.compute_moment_IP(1, xt_temp[k, ...], qt_temp[k, ...]) for k in range(param_temp.nStep)])
            mc2 = np.array([param_temp.compute_moment_IP(2, xt_temp[k, ...], qt_temp[k, ...]) for k in range(param_temp.nStep)])

            mse[i, j, 0, :] = np.linalg.norm(mc1 - m1, ord=2, axis=1)**2
            mse[i, j, 1, :] = np.linalg.norm(mc2 - m2, ord=2, axis=1)**2    

    mse = np.mean(mse, axis=0)

    fig1, axs1 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)  # Adjusted the shape here
    fig2, axs2 = plt.subplots(3, 2, figsize=(12, 9), sharex=True)  # Adjusted the shape here

    fig1.suptitle("Moment 1", fontweight='bold', fontsize=18)
    fig2.suptitle("Moment 2", fontweight='bold', fontsize=18)

    axs1 = axs1.ravel()
    axs2 = axs2.ravel()

    for i in range(6):
        axs1[i].plot(dt*np.arange(nStep),mse[i, 0, :], color='k', linestyle="-", marker=".", markersize=2)
        axs2[i].plot(dt*np.arange(nStep),mse[i, 1, :], color='k', linestyle="-", marker=".", markersize=2)

        axs1[i].set_title(f"N = {N[i]}", fontweight='bold', fontsize=15)
        axs2[i].set_title(f"N = {N[i]}", fontweight='bold', fontsize=15)

        if i in [4, 5]:
            axs1[i].set_xlabel("t")
            axs2[i].set_xlabel("t")

        if i in [0, 2, 4]:
            axs1[i].set_ylabel("MSE", rotation=0, labelpad=15, fontweight='bold')
            axs2[i].set_ylabel("MSE", rotation=0, labelpad=15, fontweight='bold')

        axs1[i].grid()
        axs2[i].grid()

    fig1.tight_layout()
    fig2.tight_layout()

    if save_as is not None: 
        fig1.savefig(save_as.replace(".pdf", "") + "_m1.pdf")
        fig2.savefig(save_as.replace(".pdf", "") + "_m2.pdf")
    else :
        plt.show()


def compare_ssf_and_mix(param):

    nStep = param.nStep
    xt_ssf, _, _, _, _, _, _ = param.split_step_flow(k = 1e-2, burn_in = 50)
    xt_mix = param.mix_flow(k = .01)

    x1, x2 = np.meshgrid(np.linspace(np.min(xt_ssf[:, :, 0]), np.max(xt_ssf[:, :, 0]), 100), np.linspace(np.min(xt_ssf[:, :, 1]), np.max(xt_ssf[:, :, 1]), 100))
    x = np.zeros((x1.size, param.n))
    x[:, 0] = x1.flatten()
    x[:, 1] = x2.flatten()
    z = param.V(x)

    fig, ax = plt.subplots(1, 1, figsize=(15, 9))
    cax = ax.contourf(x1, x2, z.reshape(100, 100), levels=100, cmap="YlGnBu")
    ax.contour(x1, x2, z.reshape(100, 100), levels=10, colors="k", linewidths=.5)
    colorbar = plt.colorbar(cax, ax=ax, label=r"$V(x)$", orientation="vertical")

    def update(frame):
        ax.clear()
        ax.contourf(x1, x2, z.reshape(100, 100), levels=100, cmap="YlGnBu")
        ax.contour(x1, x2, z.reshape(100, 100), levels=10, colors="k", linewidths=.5)
        ax.scatter(xt_ssf[frame, :, 0], xt_ssf[frame, :, 1], color='k', label = r"$x_t$ (Precondionned Langevin)")
        ax.scatter(xt_mix[frame, :, 0], xt_mix[frame, :, 1], color='r', label = r"$x_t$ (Alternance Convection-Diffusion)")
        ax.set_xlim(np.min(xt_ssf[:, :, 0]), np.max(xt_ssf[:, :, 0]))
        ax.set_ylim(np.min(xt_ssf[:, :, 1]), np.max(xt_ssf[:, :, 1]))
        ax.legend(loc = "upper left")
        return ax

    ani = animation.FuncAnimation(fig, update, frames=range(1, nStep), interval=100)
    ani.save("comparison_ssf_mix.mp4", writer='ffmpeg')
    plt.show()


def split_step_exp(param):

    nStep = param.nStep
    xt_ssf, m1_t_IP, m2_t_IP, m1_t_NIP, m2_t_NIP, m1_AIP, m2_AIP = param.split_step_flow(k = 1e-1, burn_in = param.N)

    # x1, x2 = np.meshgrid(np.linspace(np.min(xt_ssf[:, :, 0]), np.max(xt_ssf[:, :, 0]), 100), np.linspace(np.min(xt_ssf[:, :, 1]), np.max(xt_ssf[:, :, 1]), 100))
    # x = np.zeros((x1.size, param.n))
    # x[:, 0] = x1.flatten()
    # x[:, 1] = x2.flatten()
    # z = param.V(x)
    # z = param.p_xy(x, param.y)

    m1 = param.compute_moment(1)
    m2 = param.compute_moment(2)
    m2 = np.diag(m2)

    MSE_m1_IP = np.mean((m1 - m1_t_IP)**2, axis = 1)
    MSE_m2_IP = np.mean((m2 - m2_t_IP)**2, axis = 1)
    MSE_m1_NIP = np.mean((m1 - m1_t_NIP)**2, axis = 1)
    MSE_m2_NIP = np.mean((m2 - m2_t_NIP)**2, axis = 1)
    MSE_m1_AIP = np.mean((m1 - m1_AIP)**2, axis = 1)
    MSE_m2_AIP = np.mean((m2 - m2_AIP)**2, axis = 1)

    print("M2 : pred:", m2_AIP[-1],"true", m2)
    print("M1 : pred:", m1_AIP[-1],"true", m1)

    fig, ax = plt.subplots(1, 1, figsize=(15, 9))
    fig.suptitle(f"N = {param.N}, d = {param.n}")
    ax.loglog(np.arange(nStep), MSE_m1_IP, color='r', linestyle="-", markersize=2, label = "m1 with IS")
    ax.loglog(np.arange(nStep), MSE_m2_IP, color='r', linestyle="--", markersize=2, label = "m2 with IS")
    ax.loglog(np.arange(nStep), MSE_m1_NIP, color='b', linestyle="-", markersize=2, label = "m1 without IS")
    ax.loglog(np.arange(nStep), MSE_m2_NIP, color='b', linestyle="--", markersize=2, label = "m2 without IS")
    ax.loglog(np.arange(nStep), MSE_m1_AIP, color='k', linestyle="-", markersize=2, label = "m1 with AIS")
    ax.loglog(np.arange(nStep), MSE_m2_AIP, color='k', linestyle="--", markersize=2, label = "m2 with AIS")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\frac{1}{d} MSE[\hat{m}]$", rotation = 0, labelpad = 20)
    ax.legend()
    ax.grid(which='both')
    # plt.savefig(f"split_step_exp_d_{param.n}_GMM.pdf")
    plt.show()

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 9))

    # def update(frame):
    #     ax1.clear()
    #     ax2.clear()

    #     ax1.contourf(x1, x2, z.reshape(100, 100), levels=100, cmap="Reds")
    #     ax1.contour(x1,  x2, z.reshape(100, 100), levels=10, colors="k", linewidths=.5)
    #     ax1.scatter(xt_ssf[frame, :, 0], xt_ssf[frame, :, 1], color='k', label = r"$x_t (SSF)$")
    #     ax1.set_xlim(np.min(xt_ssf[:, :, 0]), np.max(xt_ssf[:, :, 0]))
    #     # ax1.set_xlim(-4, 4)
    #     ax1.set_ylim(np.min(xt_ssf[:, :, 1]), np.max(xt_ssf[:, :, 1]))
    #     # ax1.set_ylim(-4, 4)
    #     ax1.legend()

    #     ax2.loglog(np.arange(frame), MSE_m1_IP[:frame], color='r', linestyle="-", markersize=2, label = "m1 with IP")
    #     ax2.loglog(np.arange(frame), MSE_m2_IP[:frame], color='r', linestyle="--", markersize=2, label = "m2 with IP")
    #     ax2.loglog(np.arange(frame), MSE_m1_NIP[:frame], color='b', linestyle="-", markersize=2, label = "m1 without IP")
    #     ax2.loglog(np.arange(frame), MSE_m2_NIP[:frame], color='b', linestyle="--", markersize=2, label = "m2 without IP")
    #     ax2.loglog(np.arange(frame), MSE_m1_AIP[:frame], color='k', linestyle="-", markersize=2, label = "m1 with AIP")
    #     ax2.loglog(np.arange(frame), MSE_m2_AIP[:frame], color='k', linestyle="--", markersize=2, label = "m2 with AIP")
    #     ax2.set_xlabel("iteration")
    #     ax2.set_ylabel(r"$\frac{1}{d} MSE[\hat{m}]$", rotation = 0, labelpad = 20)
    #     ax2.legend()
    #     return ax1, ax2

    # ani = animation.FuncAnimation(fig, update, frames=range(1, nStep), interval=10)
    # plt.show()


if __name__ == "__main__":
    
    nStep = 2000
    N = 100
    dt = 1e-3
    n = 2
    m = 2
    sigma_x = 1 * np.eye(n)
    sigma_yx = np.diag(np.random.choice([0.5, 1,  2], m)) + 0.1 * np.ones((m, m)) 

    mu_prop = np.array(np.random.choice([0, 1, -1], n))  
    sigma_prop = 2 * np.eye(n)
    y = np.ones(m)
    A = np.eye(m, n)

    param1 = Simulation_Parameter_Gaussian(y, nStep, N, dt, n, m, A, mu_prop, sigma_prop, sigma_x, sigma_yx, uniform=True, a_unif=5)

    # split_step_exp(param1)

    # np.random.seed(1)

    nStep = 500#000
    N = 10
    dt = 1e-2
    n = 500
    m = 500

    sigma_x = 2 * np.eye(n)

    mu_prop = np.zeros(n)  
    y = np.zeros(m)#np.random.choice([-1, 1], m)
    sigma_prop = 3 * np.eye(n)

    # weight = np.array([.25, .50, .25])
    # mu_yx = np.array([[-6, -3],[0, 0], [3, 6]])
    # sigma_yx = np.array([np.eye(n), .01*np.eye(n), np.eye(n)])

    mu_yx = np.array([np.random.choice([1, -1], m), np.zeros(m), np.random.choice([-1, 1], m)])
    sigma_yx = np.array([np.diag(np.random.choice([.5, 1], n)), np.diag(np.random.choice([1, 2], n)), np.diag(np.random.choice([.1, .5], n))])
    weight = np.array([.25, .50, .25])
    # mu_yx = np.array([np.random.choice([-1, 1], m)])
    # sigma_yx = np.array([np.diag(np.random.choice([4, 5], n))])
   
    A = np.eye(m, n)
    param2 = Simulation_Parameter_Gaussian_Mixture(y, nStep, N, dt, n, m, A, mu_prop, sigma_prop, weight, sigma_x, mu_yx, sigma_yx, uniform=False, a_unif=5, scale = 1e2)
    

    split_step_exp(param2)










