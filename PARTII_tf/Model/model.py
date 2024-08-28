import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from typing import Union, Tuple

class Model:

    def __init__(self, Y_train: Union[np.ndarray, tf.Tensor], X_train: Union[np.ndarray, tf.Tensor], gw: tf.keras.Model, prior : tfp.distributions.Distribution = tfp.distributions.Uniform(-1, 1), likelihood_std : float = 1):
        """
        Args:
        X_train, Y_train : Union[np.ndarray, tf.Tensor], size (M, n), (M, m) : Training sample
        gw : tf.Module :  Neural network
        prior_bounds : Union[Union[np.ndarray, tf.Tensor], Tuple], size (d, 2) or tuple :  Bounds on the uniform prior distribution
        """

        if isinstance(Y_train, np.ndarray):
            Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)

        if isinstance(X_train, np.ndarray):
            X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)

        self.m = Y_train.shape[1]  # size of y
        self.n = X_train.shape[1]  # size of x
        self.N = X_train.shape[0]  # number of training samples
        self.d = gw.count_params() # dimension of the parameter space

        self.X_train = X_train
        self.Y_train = Y_train
        self.gw = gw

        self.prior = prior
        self.likelihood_std = likelihood_std

    def logP(self, w: tf.Tensor) -> tf.Tensor:
        
        prior = tf.reduce_sum(self.prior.log_prob(w), axis = 1)
        likelihood = - tf.reduce_mean(tf.reduce_sum((self.Y_train[None, ...] - self.gw(self.X_train, w))**2, axis = 2), axis = 1) / (2 * self.likelihood_std**2)
        return likelihood + prior


