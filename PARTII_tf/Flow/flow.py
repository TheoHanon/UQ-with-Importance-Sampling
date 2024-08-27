import tensorflow as tf
import tensorflow_probability as tfp
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from Model import model



class Flow(ABC):

    def __init__(self, M:int, epochs:int, lr:float, model : model.Model, q0 : tfp.distributions.Distribution):
        """
        Args:
            N: int Number of particles
            nStep: int Number of steps
            dt: float Time step
            q0: Distribution Initial distribution
        """

        self.M = M
        self.lr = lr
        self.epochs = epochs

        self.model = model
        self.q0 = q0

        self._w = tf.Variable(tf.zeros((self.epochs, self.M, self.model.d)), trainable=False)
        self._logq = None

    
    @tf.function()
    def compute_grad(self, w):
        """
        Args:
            w: tf.Tensor, shape (M, d) : Weights

        Returns:
            grad: tf.Tensor, shape (M, d) : Gradient
        """

        with tf.GradientTape() as tape:
            tape.watch(w)
            logP = self.model.logP(w)
        grad = tape.gradient(logP, w)
        grad = tf.clip_by_value(grad, clip_value_min=-1e0, clip_value_max=1e0)

        return grad


    @tf.function()
    def compute_grad_and_lap(self, w):
        """
        Args:
            w: tf.Tensor, shape (M, d) : Weights

        Returns:
            grad: tf.Tensor, shape (M, d) : Gradient
            lap: tf.Tensor, shape (M,) : Laplacian
        """

        with tf.GradientTape() as tape2:
            tape2.watch(w)
            with tf.GradientTape() as tape1:
                tape1.watch(w)
                logp = self.model.logP(w)

            grad = tape1.gradient(logp, w) 
            grad = tf.clip_by_value(grad, clip_value_min=-1e0, clip_value_max=1e0)
    
        lap = tf.linalg.trace(tape2.batch_jacobian(grad, w))
        
        return grad, lap

    
    def get_flow(self):
        return self._w if self._logq is None else (self._w, self._logq)
    
    def set_flow(self, w, q = None):
        self._w = w
        self._logq = q

    def flow_and_distribution(self):
        pass

    @abstractmethod
    def flow(self):
        pass


class MCMC(Flow):

    def __init__(self, burn_in, *args, **kwargs):

        self.burn_in = burn_in
        self.kernel = None
        super(MCMC, self).__init__(*args, **kwargs)

    
    @abstractmethod
    def set_kernel(self):
        pass

    def burn_in_step(self, w0):
        return self.kernel.one_step(
            current_state = w0,
            previous_kernel_results = self.kernel.bootstrap_results(w0),
        )
    
    def flow(self):

        self.set_kernel()
        kernel_results = None

        progbar = tf.keras.utils.Progbar(target=self.epochs + self.burn_in)

        for i in range(self.burn_in):
            next_state, kernel_results = self.burn_in_step(self._w[0])
            self._w[0].assign(next_state)
            progbar.update(i)

        if kernel_results is None:
            kernel_results = self.kernel.bootstrap_results(self._w[0])

        for i in range(1, self.epochs):
            next_state, kernel_results = self.kernel.one_step(
                current_state = self._w[i-1],
                previous_kernel_results = kernel_results,
            )

            self._w[i].assign(next_state)
            progbar.update(i + 1 + self.burn_in)

    
        return self._w









    
    



    
