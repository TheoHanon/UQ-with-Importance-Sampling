from .flow import Flow
from Model.model import Model
import tensorflow as tf

class OptimizerFlow(Flow):

    def __init__(self, 
                 M: int, 
                 epochs: int,
                 lr : float, 
                 model: Model, 
                 initializer: tf.keras.initializers.Initializer, 
                 optimizer: tf.keras.optimizers.Optimizer, 
                 burn_in: int = 500):

        super().__init__(M, epochs, lr, model, q0 = None, burn_in=burn_in)

        self.optimizer = optimizer(learning_rate = lr)
        self.initializer = initializer
        self._training_w = tf.Variable(tf.zeros((self.M, self.model.d)), trainable=True)
        
    def flow(self) -> tf.Tensor:

        self._training_w.assign(self.initializer(shape = (self.M, self.model.d)))
        grad = self.compute_grad(self._training_w)

        progbar = tf.keras.utils.Progbar(target=self.epochs + self.burn_in)

        for i in range(self.burn_in):
        
            # Perform an optimization step using the optimizer
            self.optimizer.apply_gradients([(-grad, self._training_w)])
            grad = self.compute_grad(self._training_w)

            progbar.update(i + 1)

        
        for i in range(self.epochs):
            self._w[i].assign(self._training_w)

            self.optimizer.apply_gradients([(-grad, self._training_w)])
            grad = self.compute_grad(self._training_w)
            progbar.update(i + self.burn_in + 1)

        return self._w
