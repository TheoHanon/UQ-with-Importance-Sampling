from .flow import MCMC
import tensorflow as tf
import tensorflow_probability as tfp


class HMC(MCMC):

    def __init__(self, num_leapfrog_step, *args, **kwargs):

        self.num_leapfrog_step = num_leapfrog_step
        super(HMC, self).__init__(*args, **kwargs)

    def set_kernel(self):
        self.kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn = self.model.logP,
            step_size = self.lr,
            num_leapfrog_steps = self.num_leapfrog_step,
        )
    
class MALA(MCMC):

    def set_kernel(self):
        self.kernel = tfp.mcmc.MetropolisAdjustedLangevinAlgorithm(
            target_log_prob_fn = self.model.logP,
            step_size = self.lr,
        )


class Langevin(MCMC):

    def set_kernel(self):
        self.kernel = tfp.mcmc.UncalibratedLangevin(
            target_log_prob_fn = self.model.logP,
            step_size = self.lr,
        )


class GradHMC(HMC) :

    """
    HMC with gradient step for burn-in
    """

    def burn_in_step(self, w0):
        grad = self.compute_grad(w0)
        w0 += self.lr * grad
        kernel = None
        return w0, kernel
    

class GradMALA(MALA):

    """
    MALA with gradient step for burn-in
    """

    def burn_in_step(self, w0):
        grad = self.compute_grad(w0)
        w0 += self.lr * grad
        kernel = None
        return w0, kernel
    

class GradLangevin(Langevin):
    
    """
    Langevin with gradient step for burn-in
    """
    
    def burn_in_step(self, w0):
        grad = self.compute_grad(w0)
        w0 += self.lr * grad
        kernel = None
        return w0, kernel
    



    



        

    



