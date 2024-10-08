from .flow import Flow
from Model import Model
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class MixFlow(Flow):

    def __init__(self, M: int, epochs: int, lr: float, model: Model, q0: tfp.distributions.Distribution, k : float, burn_in : int = 0):
        super().__init__(M, epochs, lr, model, q0, burn_in)
        self.k = k
        self._logq = tf.Variable(tf.zeros((self.epochs, self.M)), trainable=False)


    def flow(self):

        
        self._w[0].assign(self.q0.sample((self.M, self.model.d)))
        D_square_root = tf.sqrt(self.k) * tf.eye(self.model.d)
        diffusion_noise = tfp.distributions.MultivariateNormalDiag(loc = tf.zeros(self.model.d), scale_diag = tf.ones(self.model.d))

        progbar = tf.keras.utils.Progbar(target=self.epochs+self.burn_in)

        for i in range(self.burn_in):
            # Diffusion step

            noise = tf.sqrt(self.lr) * D_square_root @ tf.transpose(diffusion_noise.sample((self.M,)))
            self._w[0].assign(self._w[0] + tf.transpose(noise))
            
            # Gradient step

            grad = self.compute_grad(self._w[0]) 
            self._w[0].assign(self._w[0] + (self.lr/2)*grad)
            progbar.update(i+1)

        
        for i in range(1, self.epochs):

            # Diffusion step

            noise = tf.sqrt(self.lr) * D_square_root @ tf.transpose(diffusion_noise.sample((self.M,)))
            self._w[i].assign(self._w[i-1] + tf.transpose(noise))
            
            # Gradient step

            grad = self.compute_grad(self._w[i]) 
            self._w[i].assign( self._w[i] + (self.lr/2)*grad)
            progbar.update(i+self.burn_in+1)
        
        return self._w
    
    def flow_and_distribution(self):

        self._w[0].assign(self.q0.sample((self.M, self.model.d)))
        self._logq[0].assign(tf.reduce_sum(self.q0.log_prob(self._w[0]), axis =1))
        z = tf.Variable(tf.zeros(self.M, self.model.d), trainable = False)

        D_square_root = tf.sqrt(self.k) * tf.eye(self.model.d)
        D_inv = 1/self.k * tf.ones(self.model.d)
        diffusion_noise = tfp.distributions.MultivariateNormalDiag(loc = tf.zeros(self.model.d), scale_diag = tf.ones(self.model.d))

        progbar = tf.keras.utils.Progbar(target=self.epochs+self.burn_in)

        for i in range(self.burn_in):

            # Diffusion step
            z = self._w[0]
            noise = tf.sqrt(self.lr) * D_square_root @ tf.transpose(diffusion_noise.sample((self.M,)))
            self._w[0].assign(z + tf.transpose(noise))

            diff = self._w[0, :, None, :] - z
            exponents = -0.5 * tf.einsum("mij, j, mij -> mi", diff, D_inv / (self.lr), diff) 

            self._logq[0].assign(tf.reduce_logsumexp(exponents, axis = 1) - self.model.d/2 * tf.math.log(2*tf.constant(np.pi)*self.lr*self.k*self.M))
            grad, lap_old = self.compute_grad_and_lap(self._w[0])

            # Gradient step

            self._w[0].assign(self._w[0] + (self.lr/2)*grad)
            _, lap_new = self.compute_grad_and_lap(self._w[0])
            self._logq[0].assign(self._logq[0] - self.lr/4 * (lap_old + lap_new))

            progbar.update(i+1)

        
        for i in range(1, self.epochs):

            # Diffusion step

            noise = tf.sqrt(self.lr) * D_square_root @ tf.transpose(diffusion_noise.sample((self.M,)))
            self._w[i].assign(self._w[i-1] + tf.transpose(noise))

            diff = self._w[i, :, None, :] - self._w[i-1, None, :, :]
            exponents = -0.5 * tf.einsum("mij, j, mij -> mi", diff, D_inv / (self.lr), diff) 

            self._logq[i].assign(tf.reduce_logsumexp(exponents, axis = 1) - self.model.d/2 * tf.math.log(2*tf.constant(np.pi)*self.lr*self.k*self.M))
            grad, lap_old = self.compute_grad_and_lap(self._w[i])

            # Gradient step

            self._w[i].assign(self._w[i] + (self.lr/2)*grad)
            _, lap_new = self.compute_grad_and_lap(self._w[i])
            self._logq[i].assign(self._logq[i] - self.lr/4 * (lap_old + lap_new))

            progbar.update(i+1+self.burn_in)

        
        return self._w, self._logq