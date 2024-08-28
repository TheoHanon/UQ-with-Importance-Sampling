import numpy as np
import tensorflow as tf
from Model import Model 
from abc import ABC, abstractmethod


class Estimator(ABC):

    def __init__(self, samples_weight: tf.Tensor, model : Model):
        """
        Args:
            sample_weight: tf.Tensor, shape (nStep, M, d) : Weights
            model: Model : Model
        """
        self.model = model

        if len(samples_weight.shape) != 3:
            samples_weight = samples_weight[None, ...]

        self.nStep = samples_weight.shape[0]
        self.M = samples_weight.shape[1]
        self.d = samples_weight.shape[2]

        self.samples_weight = samples_weight


    @abstractmethod
    def estimate(self):
        pass


class MonteCarlo(Estimator):


    def estimate(self, x_pred: tf.Tensor):
        """
        Args:
            x_pred: tf.Tensor, shape (B, n) : Input

        Returns:
            tf.Tensor, shape (nStep, B, m) : Mean of the samples
        """
        mean = np.zeros((self.nStep, x_pred.shape[0], self.model.m))
        var  = np.zeros((self.nStep, x_pred.shape[0], self.model.m)) 
        
        for t in range(self.nStep):

            y_pred = self.model.gw(x_pred, self.samples_weight[t])
            mean[t] = tf.reduce_mean(y_pred, axis = 0)
            var[t]  = tf.reduce_mean((y_pred - mean[t][None, ...])**2, axis = 0)

        mean = mean.squeeze()
        var = var.squeeze()

        return mean, var
    

class WeightedMonteCarlo(MonteCarlo):

    def estimate(self, x_pred: tf.Tensor):
        """
        Args:
            x_pred: tf.Tensor, shape (B, n) : Input

        Returns:
            tf.Tensor, shape (nStep, B, m) : Mean of the samples
        """
        mean = np.zeros((self.nStep, x_pred.shape[0], self.model.m))
        var  = np.zeros((self.nStep, x_pred.shape[0], self.model.m)) 
        
        for t in range(self.nStep):

            y_pred = self.model.gw(x_pred, self.samples_weight[t])
            logp = self.model.logP(self.samples_weight[t])
            weights = tf.exp(logp - tf.reduce_logsumexp(logp, axis = 0))

            mean[t] = tf.reduce_sum(y_pred * weights[:, None, None], axis = 0)
            var[t]  = tf.reduce_sum((y_pred - mean[t][None, ...])**2 * weights[:, None, None], axis = 0)

        mean = mean.squeeze()
        var = var.squeeze()

        return mean, var
    

class ImportanceSampling(Estimator):

    def __init__(self, logq: tf.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(logq.shape) == 1:
            logq = logq[None, ...]
        elif len(logq.shape) != 2:
            raise ValueError("logq must have shape (nStep, M)")

        self.logq = logq
        self.importance_weight = np.zeros((self.nStep, self.M))

    def estimate(self, x_pred : tf.Tensor):

        mean = np.zeros((self.nStep, x_pred.shape[0], self.model.m))
        var  = np.zeros((self.nStep, x_pred.shape[0], self.model.m))

        for t in range(self.nStep):
            y_pred = self.model.gw(x_pred, self.samples_weight[t]) # (M, B, m)

            logp = self.model.logP(self.samples_weight[t])
            log_ratio = logp - self.logq[t]
            self.importance_weight[t] = tf.exp(tf.clip_by_value(log_ratio - tf.reduce_logsumexp(log_ratio, axis = 0), clip_value_min = -500, clip_value_max = 500))
            
            mean[t] = tf.reduce_sum(y_pred * self.importance_weight[t][:, None, None], axis = 0)
            var[t]  = tf.reduce_sum((y_pred - mean[t][None, ...])**2 * self.importance_weight[t][:, None, None], axis = 0)

        mean = mean.squeeze()
        var = var.squeeze()

        return mean, var
    

class AdaptiveImportanceSampling(ImportanceSampling):

    def estimate(self, x_pred: tf.Tensor):
        """
        Args:
            x_pred: tf.Tensor, shape (B, n) : Input

        Returns:
            tf.Tensor, shape (nStep, n) : Mean of the samples
            tf.Tensor, shape (nStep, n) : Variance of the samples
        """

        mean = np.zeros((self.nStep, x_pred.shape[0], self.model.m))
        var  = np.zeros((self.nStep, x_pred.shape[0], self.model.m))

        weights = np.zeros((self.nStep, self.M))
        y_pred  = np.zeros((self.nStep, self.M, x_pred.shape[0], self.model.m))

    
        for t in range(self.nStep):
            y_pred[t] = self.model.gw(x_pred, self.samples_weight[t])

            logp = self.model.logP(self.samples_weight[t])
            log_ratio = logp - self.logq[t]
            weights[t] = tf.exp(tf.clip_by_value(log_ratio - tf.reduce_logsumexp(log_ratio, axis = 0), clip_value_min = -500, clip_value_max = 500))

            W = weights[:t+1].reshape(-1)
            W = W / np.sum(W) # (t*M,)
            y = y_pred[:t+1].reshape(-1, x_pred.shape[0], self.model.m) # (t*M, B, m)

            mean[t] = tf.reduce_sum(y * W[:, None, None], axis = 0)
            var[t] = tf.reduce_sum((y - mean[t][None, ...])**2 * W[:, None, None], axis = 0)

        mean = mean.squeeze()
        var = var.squeeze()

        return mean, var
            
    




