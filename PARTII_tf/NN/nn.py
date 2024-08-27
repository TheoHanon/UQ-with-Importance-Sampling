import tensorflow as tf

class Linear(tf.keras.layers.Layer):

    def __init__(self, input_shape, output_shape):

        super(Linear, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    
    def call(self, x : tf.Tensor, W : tf.Tensor):
        """                
        Args:
        x : tf.Tensor, shape (M, B, input_features) : Input
        W : tf.Tensor, shape (M, input_features*output_features + output_features) : Weights and bias

        Returns:
        tf.Tensor, shape (M, B output_features) : Output
        """
        w, b = W[..., :self.input_shape*self.output_shape], W[..., self.input_shape*self.output_shape:]
        return tf.einsum("mbi, moi -> mbo", x, tf.reshape(w, [-1, self.output_shape, self.input_shape])) + tf.expand_dims(b, axis = 1)
    
    def count_params(self):
        return self.input_shape*self.output_shape + self.output_shape


class NN(tf.keras.Model):

    def __init__(self, input : int, output : int, hidden : int, hidden_units : int, activation : tf.function = tf.tanh):

        super(NN, self).__init__()
        
        self.model = []

        self.model.append(Linear(input, hidden_units))

        for _ in range(hidden):
            self.model.append(Linear(hidden_units, hidden_units))

        self.model.append(Linear(hidden_units, output))

        self.activation = activation
        
             
    def call(self, x: tf.Tensor, W: tf.Tensor):
        """
        Args:
        x : tf.Tensor, shape (B, input_features) : Input
        W : tf.Tensor, shape (M, d) : Weights and bias

        Returns:
        tf.Tensor, shape (B, M, output_features) : Output
        """
        idx = 0
        x = tf.tile(tf.expand_dims(x, 0), [W.shape[0], 1, 1])

        for layer in self.model[:-1]:
            n_params = layer.count_params()
            w = W[..., idx:idx+n_params]
            x = self.activation(layer(x, W = w))
            idx += n_params

        return self.model[-1](x, W = W[..., idx:])

    def count_params(self):
        return sum([layer.count_params() for layer in self.model])
            
        
    
