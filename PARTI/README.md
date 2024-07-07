# Importance Sampling for Deep Ensemble: Part I

## Problem

This part of the project is interested in the following problem:
$$
\begin{equation}
y = Ax + n
\end{equation}
$$
where \(A \in \mathbb{R}^{n \times m},~ y \in \mathbb{R}^m,~ x \in \mathbb{R}^n\), and finally,
$$
n \sim \mathcal{N}(0, \Sigma_{y|x}), ~~~ \Sigma_{y|x} \in \mathbb{R}^{m \times m}
$$
We will consider two different scenarios for the distribution of \(x\):
1. \(x \sim \mathcal{N}(0, \Sigma_x)\) with \(\Sigma_x \in \mathbb{R}^{n \times n}\)
2. \(x \sim \sum_i \alpha_i \mathcal{N}(\mu_i, \Sigma_{x, i})\) with \(\Sigma_{x, i} \in \mathbb{R}^{n \times n}\) and \(\mu_i \in \mathbb{R}^n\)

The goal is to leverage importance sampling on a probability function \(q_t\) to compute the moments of the posterior distribution \(p(x|y)\). The samples \(x_t\) are driven by a gradient flow, and we compute the density at the particles \(q_t(x_t)\) by solving the Fokker-Planck equation with the method of characteristics (cf. notes for more information).  

## Structure of the Project

The project is separated into two files.

#### `param.py`
----
We have created a parent class `Parameter_Simulation` which implements all the necessary functions to perform the importance sampling, i.e.:
* `div` : computes the divergence of \(\mu(x)\) (see notes) with automatic differentiation     
* `gradient_flow` : performs the gradient flow 
* `langevin_sampling` : performs the Langevin diffusion 
* `compute_qt` : computes the density probability function evaluated at the particles
* `compute_moment_IP` : performs importance sampling with the given \(q_t(x_t)\)
* `compute_moment` : computes the moment of the posterior with an integral

We then have two classes that inherit from the former: `Parameter_Simulation_Gaussian` and `Parameter_Simulation_Gaussian_Mixture`. 

They both implement their own versions of:
* `gradV` : gradient of \(v(x)\) (see notes)
* `p_x`, `p_xy`, `p_yx`, `p_y` : the necessary density functions

The strength of this approach is that we can re-implement the functions of the Parent Class. For example, we don't need to use automatic differentiation for the Gaussian case as we have a closed-form formula. 

#### `part1.py`
----
This file contains all the functions for the plots and animations. The plotting functions should work (**I haven't tested yet if they work for all cases**) for any instance of the `Parameter_Simulation` class.  

The interesting plotting functions right now are:
1. `draw_particular_path` : Displays the path of the particle \(x_t\)
2. `qt_and_moment_error_ani` : Animation of the evolution of \(\tilde{q}_t(\cdot)\) vs \(p(x|y)\) with the error on the moment estimation.  
3. `merged_IP_ratio` (!!only for the Gaussian case!!) : Compares the estimators using:
    1. the exact function \(q_t(\cdot)\) with resampling (not with \(x_t\))
    2. the exact function \(q_t(\cdot)\) with the \(x_t\)
    3. our approximation \(\tilde{q}_t(\cdot)\) with the \(x_t\)
4. `display_qt` : Displays the evolution of \(\tilde{q}_t(\cdot)\) alone
5. `error_moment_vs_N` : Compares the error curves for a growing number of samples. 

/!\ Although the code has been written to work in higher dimensions, we can't ensure that it will work properly right now, especially the plotting functions.
 




  








