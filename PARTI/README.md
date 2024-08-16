# Importance Sampling for Deep Ensemble: Part I

## Problem

This part of the project is interested in the following problem:
$\begin{equation}
y = Ax + n
\end{equation}$
where $A \in \mathbb{R}^{n \times m},~ y \in \mathbb{R}^m,~ x \in \mathbb{R}^n$, and finally,
$\begin{equation}
x \sim \mathcal{N}(0, \Sigma_{x}), ~~~ \Sigma_{x} \in \mathbb{R}^{m \times m}
\end{equation}$
We will consider two different scenarios for the distribution of $n$:
1. $n \sim \mathcal{N}(0, \Sigma_{x|y})$ with $\Sigma_{x|y} \in \mathbb{R}^{n \times n}$
2. $n \sim GMM(\mu_i, \Sigma_{y|x, i})$ with $\Sigma_{x|y, i} \in \mathbb{R}^{n \times n}$ and $\mu_i \in \mathbb{R}^n$

## Structure of the Project
The project file contains :
* `flow` : directory containing the implementation of the "flow" i.e. Gradient, Mix, Langevin, Hamilonian
* `Model` : directory containing the implementation of the models for the noise i.e. Gaussian or GMM as described above
* `uq.py` : script to run the experiment








  








