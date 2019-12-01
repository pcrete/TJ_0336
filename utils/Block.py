import tensorflow as tf


class Block(tf.keras.Model):
    def __init__(self, n_units):
        super().__init__()
        self.hidden = tf.keras.layers.Dense(n_units, dtype='float64')
        self.batch_norm = tf.keras.layers.BatchNormalization(dtype='float64')
        self.activation = tf.keras.layers.ReLU(dtype='float64')
        self.dropout = tf.keras.layers.Dropout(0.5, dtype='float64')
        
    def call(self, X):
        X = self.hidden(X)
        X = self.batch_norm(X)
        X = self.activation(X)
        X = self.dropout(X)
        return X
    
class Model(tf.keras.Model):
    def __init__(self, hiddens=[32, 32], out=1):
        super().__init__()
        self.hiddens = [ Block(unit) for unit in hiddens]    
        self.out = tf.keras.layers.Dense(out, dtype='float64')
        
    def call(self, X):
        for h in self.hiddens:
            X = h(X)
        out = self.out(X)
        return out