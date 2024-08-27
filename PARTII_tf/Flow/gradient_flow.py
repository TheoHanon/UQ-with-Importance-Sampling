from Model.model import Model
from .flow import Flow
import tensorflow as tf 
import tensorflow_probability as tfp
from typing import Tuple


class GradientFlow(Flow):

    def __init__(self, M: int, epochs: int, lr: float, model: Model, q0: tfp.distributions.Distribution):
        super().__init__(M, epochs, lr, model, q0)
        self._logq = tf.Variable(tf.zeros((self.epochs, self.M)), trainable=False)


    def flow(self) -> tf.Tensor:

        self._w[0].assign(self.q0.sample((self.M, self.model.d)))
        grad = self.compute_grad(self._w[0])

        progbar = tf.keras.utils.Progbar(target=self.epochs)
        
        for i in range(1, self.epochs):

            self._w[i].assign(self._w[i-1] + self.lr*grad)
            grad = self.compute_grad(self._w[i])

            progbar.update(i+1)

        return self._w
        
    def flow_and_distribution(self) -> Tuple[tf.Tensor, tf.Tensor]:

        self._w[0].assign(self.q0.sample((self.M, self.model.d)))
        self._logq[0].assign(tf.reduce_sum(self.q0.log_prob(self._w[0]), axis = 1))

        grad, lap_new = self.compute_grad_and_lap(self._w[0])

        progbar = tf.keras.utils.Progbar(target=self.epochs)
        
        for i in range(1, self.epochs):

            self._w[i].assign(self._w[i - 1] + self.lr*grad)

            lap_old = lap_new
            grad, lap_new = self.compute_grad_and_lap(self._w[i])
            self._logq[i].assign(self._logq[i-1] - self.lr/2 * (lap_old + lap_new))
            progbar.update(i+1)

        return self._w, self._logq